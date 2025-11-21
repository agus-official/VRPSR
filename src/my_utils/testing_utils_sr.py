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

import hashlib

import codecsimulator
from pathlib import Path


CODEC_CLASSES = {
    "libx264": (codecsimulator.PYRGBx264Codec, "h264"),
    "libx265": (codecsimulator.PYRGBx265Codec, "h265"),
    "libx266": (codecsimulator.PYRGBx266Codec, "h266"),
}

def parse_args_paired_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=["s3diff", "realesrgan"], required=True)
    parser.add_argument("--qp_min", type=int, default=25)
    parser.add_argument("--qp_max", type=int, default=51)

    parser.add_argument("--fixed_qp", type=int, default=None)
    parser.add_argument("--fixed_codec_type", type=str, default=None)

    parser.add_argument("--rec_save_dir", type=str, default=None, help="Directory to save 264 rec images")
    parser.add_argument("--ref_path", type=str, default=None)
    parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    parser.add_argument("--sd_path")
    parser.add_argument("--de_net_path")
    parser.add_argument("--pretrained_sr_path", type=str, default=None)
    parser.add_argument("--pretrained_sim_path", type=str, default=None)
    parser.add_argument("--pretrained_post_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--align_method", type=str, default="adain")
    
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--output_combined_dir", type=str, default='output_combined/')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

class PairedDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, root, fixed_qp=None, fixed_codec_type=None):
        super(PairedDataset, self).__init__()
        self.fixed_qp = fixed_qp
        self.fixed_codce_type = fixed_codec_type

        self.opt = opt
        if "crop_size" in opt:
            self.crop_size = opt["crop_size"]
        else:
            self.crop_size = 512

        self.root = root
        self.images = []
        if root[-3:] == "txt":
            f = open(root, "r")
            lines = f.readlines()
            for line in lines:
                self.images.append(line.strip())
        else:
            # self.images = sorted(glob(root + "/**/*.png", recursive=True))
            ex = {'.png', '.jpg', '.jpeg'}
            self.images = sorted(
                str(p) for p in Path(root).rglob('*')
                if p.suffix.lower() in ex
            )
        self.transform = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
            ]
        )
        self.totensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # blur settings for the first degradation
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]  # a list for each kernel probability
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt[
            "betag_range"
        ]  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt["betap_range"]  # betap used in plateau blur kernels
        self.sinc_prob = opt["sinc_prob"]  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]

        # a final sinc filter
        self.final_sinc_prob = opt["final_sinc_prob"]

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def set_seed_from_name(self, image_name):
        seed = int(hashlib.sha256(image_name.encode('utf-8')).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def __getitem__(self, index):
        try:
            img_gt = Image.open(self.images[index]).convert("RGB")
        except Exception as e:
            print("bad image {}".format(self.images[index]))
            return self.__getitem__(0)

        image_name = self.images[index].split("/")[-1]
        seed = self.set_seed_from_name(image_name)

        if self.transform is not None:
            img_gt = self.transform(img_gt)


        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob2"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = self.totensor(img_gt)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        codec_type = self.fixed_codce_type
        qp = self.fixed_qp

        image_name = os.path.basename(self.images[index])

        return_d = {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "codec_type": codec_type,
            "qp": qp,
            "seed": seed,
            "image_name": image_name
        }
        return return_d

    def __len__(self):
        return len(self.images)


def degradation_proc(
    configs, batch, device, seed=None, val=False, use_usm=False, resize_lq=True, random_size=False
):
    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    jpeger = DiffJPEG(
        differentiable=False
    ).cuda()  # simulate JPEG compression artifacts
    usm_sharpener = USMSharp().cuda()  # do usm sharpening

    im_gt = batch["gt"].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch["kernel1"].cuda()
    kernel2 = batch["kernel2"].cuda()
    sinc_kernel = batch["sinc_kernel"].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
        ["up", "down", "keep"],
        configs.degradation["resize_prob"],
    )[0]
    if updown_type == "up":
        scale = random.uniform(1, configs.degradation["resize_range"][1])
    elif updown_type == "down":
        scale = random.uniform(configs.degradation["resize_range"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = configs.degradation["gray_noise_prob"]
    if random.random() < configs.degradation["gaussian_noise_prob"]:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs.degradation["noise_range"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            seed=seed
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs.degradation["poisson_scale_range"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            seed=seed
        )
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation["jpeg_range"])
    out = torch.clamp(
        out, 0, 1
    )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < configs.degradation["second_blur_prob"]:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
        ["up", "down", "keep"],
        configs.degradation["resize_prob2"],
    )[0]
    if updown_type == "up":
        scale = random.uniform(1, configs.degradation["resize_range2"][1])
    elif updown_type == "down":
        scale = random.uniform(configs.degradation["resize_range2"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(
        out,
        size=(int(ori_h / configs.sf * scale), int(ori_w / configs.sf * scale)),
        mode=mode,
    )
    # add noise
    gray_noise_prob = configs.degradation["gray_noise_prob2"]
    if random.random() < configs.degradation["gaussian_noise_prob2"]:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs.degradation["noise_range2"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            seed=seed
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs.degradation["poisson_scale_range2"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            seed=seed
        )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(ori_h // configs.sf, ori_w // configs.sf),
            mode=mode,
        )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(
            *configs.degradation["jpeg_range2"]
        )
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(
            *configs.degradation["jpeg_range2"]
        )
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(ori_h // configs.sf, ori_w // configs.sf),
            mode=mode,
        )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = configs.degradation["gt_size"]
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, configs.sf)
    lq, gt = im_lq, im_gt
    ori_lq = im_lq

    if resize_lq:
        lq = F.interpolate(
            lq,
            size=(gt.size(-2), gt.size(-1)),
            mode="bicubic",
        )

    if (
        random.random() < configs.degradation["no_degradation_prob"]
        or torch.isnan(lq).any()
    ):
        lq = gt

    # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
    lq = (
        lq.contiguous()
    )  # for the warning: grad and param do not obey the gradient layout contract

    prompts, blur_264, bpp = get_prompt(lq, batch["codec_type"], batch["qp"])

    lq = lq * 2 - 1.0
    gt = gt * 2 - 1.0

    lq = torch.clamp(lq, -1.0, 1.0)
    gt = torch.clamp(gt, -1.0, 1.0)

    return {
        "lq": lq.to(device),
        "gt": gt.to(device),
        "ori_lq": ori_lq.to(device),
        "codec_bpp_prompts": prompts,
        "blur_264": blur_264,
        "bpp": bpp,
    }


def get_prompt(blur_img, codec_type, qp):
    prompts = []
    blur_264_imgs = []
    bpps = []

    for i in range(blur_img.size(0)):
        img_pil = transforms.ToPILImage()(blur_img[i].cpu())
        _, blur_264, bpp = proc(img_pil, codec_type[i], qp[i].cpu().item())

        # prompts.append("A {0} (qp={1}) {2:.4f} bpp compressed image.".format(codec_type[i], qp[i], bpp))
        prompts.append("A {0} {1:.4f} bpp compressed image.".format(codec_type[i], bpp))
        blur_264_imgs.append(blur_264)
        bpps.append(bpp)

    to_tensor = transforms.ToTensor()
    blur_264_imgs = [to_tensor(img) for img in blur_264_imgs]
    blur_264_imgs = torch.stack(blur_264_imgs)
    bpps = torch.tensor(bpps, dtype=torch.float32)

    return prompts, blur_264_imgs, bpps


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

    # bpp = packet_size * 8 / (width * height)
    bpp = packet_size / 1000

    return img, rec, bpp


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

        target_bpp = tgt_bpp[i].item() if torch.is_tensor(tgt_bpp) else float(tgt_bpp)

        left, right = qp_min, qp_max
        best_qp = qp_min
        best_bpp = float("inf")
        best_x_codec_pred = None 

        while left <= right:
            qp_mid = (left + right) // 2
            _, x_codec_pred, bpp = proc(input_img[i], codec_type=tgt_codec[i], qp=qp_mid)

            if isinstance(x_codec_pred, Image.Image):
                x_codec_pred = transforms.ToTensor()(x_codec_pred)

            if abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
                best_bpp = bpp
                best_qp = qp_mid
                best_x_codec_pred = x_codec_pred

            if bpp > target_bpp:
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

