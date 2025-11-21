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

def parse_args_paired_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_with_no_meta", action="store_true")
    parser.add_argument("--rate_loss", action="store_true")

    parser.add_argument("--sf", type=int, default=1,)
    parser.add_argument("--fixed_codec_type", type=str, default=None,)
    parser.add_argument("--fixed_qp", type=int, default=None,)
    parser.add_argument("--rec_save_dir", type=str, default=None, help="Directory to save 26x rec images")
    parser.add_argument("--ref_path", type=str, default=None,)
    parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--de_net_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--pretrained_sim_path", type=str, default=None,)
    parser.add_argument("--pretrained_sr_path", type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=96, help="Chopping stride.")
    parser.add_argument("--padding_offset", type=int, default=32, help="padding offset.")

    parser.add_argument("--align_method", type=str, default="adain")

    parser.add_argument("--output_dir", type=str, default='output/')
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



CODEC_CLASSES = {
    "libx264": (codecsimulator.PYRGBx264Codec, "h264"),
    "libx265": (codecsimulator.PYRGBx265Codec, "h265"),
    "libx266": (codecsimulator.PYRGBx266Codec, "h266"),
}

def proc_h26x(img, qp, codec_class, codec_name, width=512, height=512):
    codec = codec_class(width, height)
    img_np = np.asarray(img)
    out_np = np.zeros_like(img_np)

    packet_size = codec.Encode(img_np, out_np, qp)

    rec = Image.fromarray(out_np, 'RGB')

    return img, rec, packet_size

def proc(img, codec_type, qp):
    codec_data = CODEC_CLASSES.get(codec_type)

    if codec_data is None:
        raise ValueError(f"Unsupported codec type: {codec_type}")

    codec_class, codec_name = codec_data
    return proc_h26x(img, qp, codec_class, codec_name)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, codec_type=None, qp=None):
        self.root = root
        self.codec_type = codec_type
        self.qp = qp
        self.images = []
        if root[-3:] == 'txt':
            f = open(root, 'r')
            lines = f.readlines()
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(glob(root + '/**/*.png', recursive=True))
        self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
        ])
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.codec_types = ["libx264", "libx265", "libx266"]
        self.qps = [i for i in range(31, 52)]


    def __getitem__(self, index):
        try:
            img_path = self.images[index]
            filename = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("bad image {}".format(self.images[index]))
            return self.__getitem__(0)

        if self.transform is not None:
            img = self.transform(img)

        codec_type = self.codec_type if self.codec_type is not None else random.choice(self.codec_types)
        qp = self.qp if self.qp is not None else random.choice(self.qps)

        img, rec, packet_size = proc(img, codec_type, qp)
        # bpp = packet_size * 8 / (512 * 512)
        packet_size = packet_size / 1000
        img = self.totensor(img)
        rec = self.totensor(rec)
        prompt = "A {0} {1:.4f} bpp compressed image.".format(self.codec_type, packet_size)

        return {
            "x_src": img,
            "x_tgt": rec,
            "name": filename,
            "bpp": packet_size,
            "codec_type": codec_type,
            "qp": qp,
            "prompt": prompt
        }

    def __len__(self):
        return len(self.images)
