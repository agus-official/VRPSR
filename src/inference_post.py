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

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from de_net import DEResNet
from s3diff_meta_embed import S3DiffMetaEmbed
from pathlib import Path
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

from my_utils.testing_utils_sr import parse_args_paired_testing, PairedDataset, degradation_proc
from my_utils.vrpsr_utils import build_temb, find_best_qp, evaluate


def main(args):
    config = OmegaConf.load(args.base_config)

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

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
        os.makedirs(os.path.join(args.output_combined_dir), exist_ok=True)

    # initialize networks
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    net_sr = S3DiffMetaEmbed(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_sr_path,
    )
    net_sr.set_eval()

    net_post = S3DiffMetaEmbed(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_post_path,
    )
    net_post.set_eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
            net_post.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()
        net_post.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare data
    dataset_val = PairedDataset(
        config.validation,
        args.ref_path,
        args.fixed_qp,
        args.fixed_codec_type,
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_post.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)

    # Prepare everything with our `accelerator`.
    net_sr, net_post, net_de = accelerator.prepare(net_sr, net_post, net_de)
    bpp = 0

    for step, batch_val in enumerate(dl_val):

        # Step 0: Degradation --> x_src
        res = degradation_proc(config, batch_val, accelerator.device, seed=batch_val["seed"].item())
        x_src, x_tgt, x_ori_size_src = res["lq"], res["gt"], res["ori_lq"]
        
        prompt = res["codec_bpp_prompts"]
        print(f"prompt: {prompt}")

        with torch.no_grad():
            # forward pass
            # Step 1: Step 1: SR net: x_src --> x_sr_pred
            deg_score = net_de(x_ori_size_src.detach())
            tgt_codec = batch_val["codec_type"]
            tgt_bpp = res["bpp"]

            # --- build temb ---
            temb = build_temb(codec=tgt_codec, bpp=tgt_bpp, device=accelerator.device)
            
            # --- net_sr ---
            x_sr_pred, _ = accelerator.unwrap_model(net_sr)(
                x_src.clone(), deg_score, prompt, meta_vec=temb
            )

            # Step 2: Real H264: x_sr_pred --> x_codec_pred 
            _, best_bpp, x_codec_pred = find_best_qp(
                tgt_bpp, tgt_codec, x_sr_pred, accelerator.device, args.qp_min, args.qp_max
            )
            bpp += best_bpp.item()
            
            # Step 3: Postprocess: x_codec_pred --> x_post_pred
            deg_score_codec = net_de(x_codec_pred.detach() * 0.5 + 0.5)
            x_post_pred, _ = accelerator.unwrap_model(net_post)(
                x_codec_pred.clone(), deg_score_codec, prompt, meta_vec=temb
            )

            x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
            x_tgt = torch.clamp(x_tgt, 0.0, 1.0)
            x_src = x_src.cpu().detach() * 0.5 + 0.5
            x_src = torch.clamp(x_src, 0.0, 1.0)
            x_sr_pred = x_sr_pred.cpu().detach()
            x_sr_pred = torch.clamp(x_sr_pred, 0.0, 1.0)
            x_codec_pred = x_codec_pred.cpu().detach() * 0.5 + 0.5
            x_codec_pred = torch.clamp(x_codec_pred, 0.0, 1.0)
            x_post_pred = x_post_pred.cpu().detach() * 0.5 + 0.5
            x_post_pred = torch.clamp(x_post_pred, 0.0, 1.0)

            out_img = x_post_pred

            if args.align_method == 'nofix':
                out_img = out_img
            else:
                x_src_fix = transforms.ToPILImage()(x_src[0].cpu().detach().float())
                out_img_fix = transforms.ToPILImage()(out_img[0].cpu().detach().float())
                
                if args.align_method == 'wavelet':
                    out_img_fix = wavelet_color_fix(out_img_fix, x_src_fix)
                elif args.align_method == 'adain':
                    out_img_fix = adain_color_fix(out_img_fix, x_src_fix)
                
                out_img = ToTensor()(out_img_fix)
                out_img = out_img.unsqueeze(0)
            
            combined = torch.cat(
                [
                    x_tgt,
                    x_src,
                    x_sr_pred,
                    x_codec_pred,
                    out_img,
                ],
                dim=3,
            )

        output_pil = transforms.ToPILImage()(out_img[0])

        combined = torch.clamp(combined, 0.0, 1.0)
        output_pil_combined = transforms.ToPILImage()(combined[0])

        name = batch_val["image_name"][0]
        print(f"name: {name}")
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname+'.png')
        output_pil.save(outf)

        outf = os.path.join(args.output_combined_dir, fname+'.png')
        output_pil_combined.save(outf)

    print_results = evaluate(args.output_dir, args.ref_path, bpp, None)
    out_t = os.path.join(args.output_dir, 'results.txt')
    with open(out_t, 'w', encoding='utf-8') as f:
        for item in print_results:
            f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
