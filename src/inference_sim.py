import os
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import torchvision
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

import diffusers
import utils.misc as misc

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from de_net import DEResNet
from s3diff_meta_embed import S3DiffMetaEmbed

from pathlib import Path
from utils import util_image
from utils.wavelet_color import wavelet_color_fix, adain_color_fix
from my_utils.testing_utils_sim import parse_args_paired_testing, ImageDataset

from my_utils.vrpsr_utils import build_temb, evaluate


def main(args):
    if args.pretrained_path is None:
        from huggingface_hub import hf_hub_download
        pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    else:
        pretrained_path = args.pretrained_path

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # initialize net
    net_sr = S3DiffMetaEmbed(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=pretrained_path,
    )
    net_sr.set_eval()

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_val = ImageDataset(args.ref_path, codec_type=args.fixed_codec_type, qp=args.fixed_qp)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)

    # Prepare everything with our `accelerator`.
    net_sr, net_de = accelerator.prepare(net_sr, net_de)

    if args.rec_save_dir is not None:
        os.makedirs(args.rec_save_dir, exist_ok=True)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    for step, batch in enumerate(dl_val):
        im_lr = batch["x_src"].cuda()
        im_lr = im_lr.to(memory_format=torch.contiguous_format).float()
        im_lr_ori_size = im_lr * 0.5 + 0.5      

        name = batch["name"]
        rec_img = (batch["x_tgt"] * 0.5 + 0.5).cpu().detach()
        rec_pil = transforms.ToPILImage()(rec_img[0])
        rec_path = os.path.join(args.rec_save_dir, f"{name[0]}.png")
        rec_pil.save(rec_path)

        tgt_codec = batch["codec_type"]
        tgt_bpp = batch["bpp"]
        prompt = batch["prompt"]
        print(f"prompt: {prompt}")  

        # save prompt to prompt.txt
        prompt_path = os.path.join(args.rec_save_dir, "prompt.txt")
        with open(prompt_path, "a", encoding="utf-8") as f:
            f.write(f"{name[0]}: {prompt}\n")

        with torch.no_grad():
            # forward pass
            deg_score = net_de(im_lr_ori_size).detach()            
            
            # --- build temb ---
            temb = build_temb(codec=tgt_codec, bpp=tgt_bpp, device=accelerator.device)

            if temb is not None:
                x_tgt_pred, _ = accelerator.unwrap_model(net_sr)(
                    im_lr, deg_score, prompt, meta_vec=temb
                )
            else:
                x_tgt_pred, _ = accelerator.unwrap_model(net_sr)(
                    im_lr, deg_score, prompt
                )

        outf = os.path.join(args.output_dir, f"{name[0]}.png")
        torchvision.utils.save_image(x_tgt_pred.cpu().detach(), outf, normalize=True, value_range=(-1,1))

        name = batch["name"][0]
        print(f"name: {name}")
    sim_results = evaluate(args.output_dir, args.rec_save_dir, None, None)
    sim_t = os.path.join(args.output_dir, 'results.txt')
    with open(sim_t, 'w', encoding='utf-8') as f:
        for item in sim_results:
            f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
