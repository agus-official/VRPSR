import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import random
import codecsimulator
import pyiqa
import tqdm
from utils import util_image
import shutil

CODEC2IDX = {"libx264": 0, "libx265": 1, "libx266": 2}


def build_temb(codec, bpp=None, device=None):
    # codec(onehot) (+ bpp) -> temb
    # B
    if isinstance(codec, (list, tuple)):
        B = len(codec)
    elif torch.is_tensor(codec):
        B = int(codec.numel())
        codec = [str(x) for x in codec.view(-1).tolist()]
    else:
        B = 1
        codec = [str(codec)]

    # device
    if device is None:
        device = (bpp.device if torch.is_tensor(bpp) and bpp.device.type != "cpu"
                  else torch.device("cpu"))

    # codec -> onehot (B,3)
    idx = torch.tensor([CODEC2IDX[c] for c in codec], device=device, dtype=torch.long)
    onehot = F.one_hot(idx, num_classes=3).float()

    def to_col(x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(device=device, dtype=torch.float32).view(-1)
        if x.numel() == 1 and B > 1:
            x = x.expand(B)
        return x.view(B, 1)
    bpp_col = to_col(bpp)

    if bpp_col is None:
        raise ValueError("build_temb: need `bpp`.")

    parts = [onehot]
    if bpp_col is not None: parts.append(bpp_col)
    return torch.cat(parts, dim=1)


def proc_h26x(img, qp, codec_class, codec_name, width=512, height=512):
    codec = codec_class(width, height)
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img.cpu())
        img_np = np.array(img)
    elif isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    out_np = np.zeros_like(img_np)

    packet_size = codec.Encode(img_np, out_np, qp)

    rec = Image.fromarray(out_np, 'RGB')
    
    bpp = packet_size / 1000

    return img, rec, bpp

CODEC_CLASSES = {
    "libx264": (codecsimulator.PYRGBx264Codec, "h264"),
    "libx265": (codecsimulator.PYRGBx265Codec, "h265"),
    "libx266": (codecsimulator.PYRGBx266Codec, "h266"),
}
def proc(img, codec_type, qp):
    codec_data = CODEC_CLASSES.get(codec_type)

    if codec_data is None:
        raise ValueError(f"Unsupported codec type: {codec_type}")

    codec_class, codec_name = codec_data
    return proc_h26x(img, qp, codec_class, codec_name)


def find_best_qp(tgt_bpp, tgt_codec, input_img, device, qp_min, qp_max):
    best_qps = []
    best_bpps = []
    best_x_codec_preds = []

    for i in range(input_img.size(0)):
        input_img[i] = torch.clamp(input_img[i], -1, 1)
        input_img[i] = input_img[i] * 0.5 + 0.5

        left, right = qp_min, qp_max
        best_qp = qp_min
        best_bpp = float("inf")  
        best_x_codec_pred = None  

        while left <= right: 
            qp_mid = (left + right) // 2
            _, x_codec_pred, bpp = proc(input_img[i], codec_type=tgt_codec[i], qp=qp_mid)

            if isinstance(x_codec_pred, Image.Image):
                x_codec_pred = transforms.ToTensor()(x_codec_pred)

            if abs(bpp - tgt_bpp) < abs(best_bpp - tgt_bpp):
                best_bpp = bpp
                best_qp = qp_mid
                best_x_codec_pred = x_codec_pred

            if bpp > tgt_bpp:
                left = qp_mid + 1
            else:
                right = qp_mid - 1

        best_qps.append(best_qp)
        best_bpps.append(best_bpp)
        best_x_codec_preds.append(best_x_codec_pred)

    best_qps = torch.tensor(best_qps, dtype=torch.int)
    best_bpps = torch.tensor(best_bpps, dtype=torch.float32)
    best_x_codec_preds = torch.stack(
        best_x_codec_preds
    )

    best_x_codec_preds = best_x_codec_preds * 2.0 - 1.0
    best_x_codec_preds = torch.clamp(best_x_codec_preds, -1, 1)

    return best_qps, best_bpps, best_x_codec_preds.to(device)


def real_codec_proc(tgt_codec, tgt_qp, input_img, device):

    x_codec_preds = []
    bpps = []

    for i in range(input_img.size(0)):
        img = input_img[i] * 0.5 + 0.5
        img = torch.clamp(img, 0, 1)
        
        img = transforms.ToPILImage()(img.cpu())
        _, x_codec_pred, bpp = proc(img, codec_type=tgt_codec[i], qp=tgt_qp[i].cpu().item())

        if isinstance(x_codec_pred, Image.Image):
            x_codec_pred = transforms.ToTensor()(x_codec_pred)

        bpps.append(bpp)
        x_codec_preds.append(x_codec_pred)

    bpps = torch.tensor(bpps, dtype=torch.float32)
    x_codec_preds = torch.stack(
        x_codec_preds
    )

    x_codec_preds = x_codec_preds * 2.0 - 1.0
    x_codec_preds = torch.clamp(x_codec_preds, -1, 1)

    return bpps, x_codec_preds.to(device)

def _lowercase_ext_view(src_dir: Path) -> Path:
    src_dir = Path(src_dir)
    tmp_dir = src_dir.parent / f"_fid_tmp_{src_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if p.is_file():
            dst = tmp_dir / (p.stem + p.suffix.lower())
            try:
                os.symlink(p.resolve(strict=False), dst)
            except Exception:
                shutil.copy2(p, dst)
    return tmp_dir


def evaluate(in_path, ref_path, bpp, ntest):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_paired_dict = {}

    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()

    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]

        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)

    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]

    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None

        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w

        if ref_path is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()

    if ref_path is not None:
        src_for_fid = _lowercase_ext_view(in_path)
        ref_for_fid = _lowercase_ext_view(ref_path)

        try:
            fid_metric = pyiqa.create_metric('fid')
            result['fid'] = fid_metric(str(src_for_fid), str(ref_for_fid))
        finally:
            shutil.rmtree(src_for_fid, ignore_errors=True)
            shutil.rmtree(ref_for_fid, ignore_errors=True)

    if bpp is not None:
        result['average bpp'] = bpp

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
            print_results.append(f"{key}: {res/len(lr_path_list):.5f}")
    return print_results