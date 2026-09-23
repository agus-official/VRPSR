from vrpsr_training.options import parse_sr_args as parse_args_paired_training
if __name__ == '__main__':
    args = parse_args_paired_training()
from vrpsr_training.evaluation import evaluate
from vrpsr_training.codec import build_temb, find_best_qp, real_codec_proc_with_prompt
from vrpsr_training.data import SRTrainDataset as DatasetTrain, SRValidationDataset as DatasetVal, degrade_train as degradation_proc_train, degrade_validation as degradation_proc_val
import os
import json
import importlib.metadata
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
import torchvision
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from de_net import DEResNet
from s3diff_meta_embed import S3DiffMetaEmbed
from unet_with_meta import ResnetBlock2DWithMeta
import wandb
import datetime

def verify_meta_mlp_trainable(unet_with_meta, resnet):
    mlps = [m.mlp for m in unet_with_meta.modules() if isinstance(m, resnet)]
    assert len(mlps) > 0, 'No ResnetBlock2DWithMeta found! Check model conversion.'
    for i, mlp in enumerate(mlps):
        for name, param in mlp.named_parameters():
            assert param.requires_grad, f'MLP {i}.{name} is not trainable!'
    print(f'✅ All {len(mlps)} meta-mlps are trainable.')

def toggle_optimizer_grad(optimizer, flag: bool, verbose: bool=False):
    for group_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group['params']):
            p.requires_grad = flag
            if verbose:
                print(f"[{optimizer.__class__.__name__}] group{group_idx}/param{p_idx} shape={tuple(p.shape)} → {('ON' if flag else 'OFF')}")

def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=None)
    if args.wandb and accelerator.is_main_process:
        wandb.init(project='vrpsr_sr', name=f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", config=args, mode='offline')
    config = OmegaConf.load(args.base_config)
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id='stabilityai/sd-turbo')
    else:
        sd_path = args.sd_path
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.output_dir, 'environment.txt'), 'w') as f:
            for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get('Name', '').lower()):
                f.write(f"{dist.metadata.get('Name', 'unknown')}=={dist.version}\n")
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()
    net_de.requires_grad_(False)
    net_sim = S3DiffMetaEmbed(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=args.pretrained_sim_path)
    net_sr = S3DiffMetaEmbed(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=args.pretrained_sr_path)
    for p in net_sim.parameters():
        p.requires_grad_(False)
    net_sim.set_eval()
    net_sr.set_train()
    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        net_sr.unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    import vision_aided_loss

    def build_disc():
        d = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=None, device='cuda').cuda()
        d.disc_loss = vision_aided_loss.cv_losses.losses_list(args.gan_loss_type)
        d.requires_grad_(True)
        d.cv_ensemble.requires_grad_(False)
        d.train()
        return d
    net_disc_sr = build_disc()
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.eval()
    net_lpips.requires_grad_(False)
    resnet = ResnetBlock2DWithMeta
    verify_meta_mlp_trainable(net_sr.unet, resnet)
    net_sim.to(accelerator.device, dtype=weight_dtype)
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=torch.float32)
    net_disc_sr.to(accelerator.device, dtype=weight_dtype)
    sr_params = []
    for attr in ['vae_block_embeddings', 'unet_block_embeddings', 'vae_de_mlp', 'unet_de_mlp', 'vae_block_mlp', 'unet_block_mlp', 'vae_fuse_mlp', 'unet_fuse_mlp']:
        if hasattr(net_sr, attr):
            sr_params += list(getattr(net_sr, attr).parameters())
    for n, _p in net_sr.unet.named_parameters():
        if 'lora' in n:
            sr_params.append(_p)
    if hasattr(net_sr.unet, 'conv_in'):
        sr_params += list(net_sr.unet.conv_in.parameters())
    for n, _p in net_sr.vae.named_parameters():
        if 'lora' in n:
            sr_params.append(_p)
    sr_meta_mlp_params = []
    for m in net_sr.unet.modules():
        if isinstance(m, ResnetBlock2DWithMeta):
            sr_meta_mlp_params.extend(list(m.mlp.parameters()))
    sr_meta_mlp_params = [p for p in sr_meta_mlp_params if p.requires_grad]
    assert len(sr_meta_mlp_params) > 0, 'No meta-mlp parameters found!'
    sr_params += sr_meta_mlp_params
    optimizer_sr = torch.optim.AdamW(sr_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler_sr = get_scheduler(args.lr_scheduler, optimizer=optimizer_sr, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    optimizer_disc_sr = torch.optim.AdamW(net_disc_sr.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler_disc_sr = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc_sr, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * 1000 * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    dataset_train = DatasetTrain(config.train, args.train_data, qp=args.fixed_qp, codec_type=args.fixed_codec_type)
    dataset_val = DatasetVal(config.validation, args.val_data, fixed_qp=args.fixed_qp_val, fixed_codec_type=args.fixed_codec_type_val)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, persistent_workers=True if args.dataloader_num_workers > 0 else False)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    disc_list = [net_disc_sr]
    for d in disc_list:
        for name, module in d.named_modules():
            if 'attn' in name:
                module.fused_attn = False
    net_sim = accelerator.prepare(net_sim)
    net_sr, dl_train, optimizer_sr, lr_scheduler_sr, net_disc_sr, optimizer_disc_sr, lr_scheduler_disc_sr = accelerator.prepare(
        net_sr, dl_train, optimizer_sr, lr_scheduler_sr, net_disc_sr, optimizer_disc_sr, lr_scheduler_disc_sr
    )
    net_de, net_lpips = accelerator.prepare(net_de, net_lpips)

    def infinite_loader(dataloader):
        it = iter(dataloader)
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(dataloader)
    train_iter = infinite_loader(dl_train)
    progress_bar = tqdm(total=args.max_train_steps + 1, initial=1, desc='MainSteps', disable=not accelerator.is_local_main_process)
    global_step = 0
    while global_step < args.max_train_steps:
        toggle_optimizer_grad(optimizer_sr, True)
        accelerator.unwrap_model(net_sim).set_eval()
        accelerator.unwrap_model(net_sr).set_train()
        for _ in range(min(1000, args.max_train_steps - global_step)):
            batch = next(train_iter)
            x_src, x_tgt, x_ori_size_src, pos_prompt, _, tgt_bpp = degradation_proc_train(config, batch, accelerator.device)
            B, C, H, W = x_src.shape
            sim_prompt = pos_prompt
            with torch.no_grad():
                deg_score = net_de(x_ori_size_src.detach()).detach()
            acc_targets = [net_sr] + [net_disc_sr]
            optimizer_sr.zero_grad(set_to_none=True)
            temb = build_temb(batch['codec_type'], tgt_bpp, device=accelerator.device)
            with accelerator.accumulate(*acc_targets):
                x_sr_pred, _ = net_sr(x_src, deg_score, pos_prompt, meta_vec=temb)
                x_sr_pred_ori_size = (x_sr_pred + 1.0) / 2.0
                with torch.no_grad():
                    sr_deg = net_de(x_sr_pred_ori_size).detach()
                x_sim_pred, _ = net_sim(x_sr_pred, sr_deg, sim_prompt, meta_vec=temb)
                qp_minus = (batch['qp'].to(torch.int32) - 10).clamp_(0, 51)
                qp_minus_bpp, x_reg, qp_minus_prompt = real_codec_proc_with_prompt(batch['codec_type'], qp_minus, x_tgt.detach(), accelerator.device)
                with torch.no_grad():
                    x_tgt_ori_size = (x_tgt + 1.0) / 2.0
                    tgt_deg = net_de(x_tgt_ori_size).detach()
                temb_qp_minus_sim = build_temb(batch['codec_type'], qp_minus_bpp, device=accelerator.device)
                x_reg, _ = net_sim(x_tgt, tgt_deg, qp_minus_prompt, meta_vec=temb_qp_minus_sim)
                with torch.cuda.amp.autocast(enabled=False):
                    lpips_ori = net_lpips(x_sim_pred.float(), x_reg.detach().float()).mean() * args.lambda_lpips
                loss_ori = F.mse_loss(x_sim_pred, x_reg) * args.lambda_l2 + lpips_ori
                loss_sr = loss_ori
                x_pred_for_gan = x_sim_pred
                fake_out_G_sr = net_disc_sr(x_pred_for_gan)
                lossG_sr = accelerator.unwrap_model(net_disc_sr).disc_loss(fake_out_G_sr, for_G=True).mean() * args.lambda_gan
                loss_sr += lossG_sr
                accelerator.backward(loss_sr, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_params, args.max_grad_norm)
                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad(set_to_none=args.set_grads_to_none)
                x_real_for_gan = x_reg
                disc_out_sr = net_disc_sr(torch.cat([x_real_for_gan.detach(), x_pred_for_gan.detach()], dim=0), detach=True)[0]
                batch_out_sr = [torch.chunk(disc_out_i, dim=0, chunks=2) for disc_out_i in disc_out_sr]
                real_out_sr, fake_out_sr = [list(x) for x in zip(*batch_out_sr)]
                lossD_real_sr = accelerator.unwrap_model(net_disc_sr).disc_loss([real_out_sr], for_real=True).mean() * args.lambda_gan
                lossD_fake_sr = accelerator.unwrap_model(net_disc_sr).disc_loss([fake_out_sr], for_real=False).mean() * args.lambda_gan
                lossD_sr = lossD_real_sr + lossD_fake_sr
                accelerator.backward(lossD_sr.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc_sr.parameters(), args.max_grad_norm)
                optimizer_disc_sr.step()
                lr_scheduler_disc_sr.step()
                optimizer_disc_sr.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.is_local_main_process:
                progress_bar.update(1)
        prev_global_step = global_step
        global_step += min(1000, args.max_train_steps - global_step)
        if accelerator.is_main_process and args.wandb:
            base_logs = {'lr/sr': lr_scheduler_sr.get_last_lr()[0], 'main_step': global_step}
            try:
                base_logs.update({'loss/sr': float(loss_sr.detach().item())})
            except Exception:
                pass
            wandb.log(base_logs, step=global_step)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and prev_global_step // args.checkpointing_steps < global_step // args.checkpointing_steps:
            ckpt_dir = os.path.join(args.output_dir, 'checkpoints', f'step_{global_step}')
            os.makedirs(ckpt_dir, exist_ok=True)
            accelerator.unwrap_model(net_sr).save_model(os.path.join(ckpt_dir, f'sr_{global_step}.pkl'))
        if accelerator.is_main_process and args.eval_freq > 0 and prev_global_step // args.eval_freq < global_step // args.eval_freq:
            accelerator.unwrap_model(net_sr).set_eval()
            sr_pred_dir = os.path.join(args.output_dir, 'sr_test', f'{global_step}_val', 'pred')
            sr_gt_dir = os.path.join(args.output_dir, 'sr_test', f'{global_step}_val', 'gt')
            sr_reg_dir = os.path.join(args.output_dir, 'sr_test', f'{global_step}_val', 'reg')
            sr_dir = os.path.join(args.output_dir, 'sr_test', f'{global_step}_val', 'sr')
            os.makedirs(sr_pred_dir, exist_ok=True)
            os.makedirs(sr_gt_dir, exist_ok=True)
            os.makedirs(sr_reg_dir, exist_ok=True)
            os.makedirs(sr_dir, exist_ok=True)
            l2_vals, lpips_vals = ([], [])
            for i, batch_val in enumerate(dl_val):
                if i >= args.num_samples_eval:
                    break
                with torch.no_grad():
                    x_src, x_tgt, x_ori_size_src, pos_prompt, _, tgt_bpp = degradation_proc_val(config, batch_val, accelerator.device, seed=batch_val['seed'].item())
                    B, C, H, W = x_src.shape
                    deg_score = net_de(x_ori_size_src.detach())
                    temb = build_temb(batch_val['codec_type'], tgt_bpp, device=accelerator.device)
                    x_sr_pred, _ = accelerator.unwrap_model(net_sr)(x_src.to(accelerator.device), deg_score, pos_prompt, meta_vec=temb)
                    best_qp, best_bpp, x_codec_pred = find_best_qp(tgt_bpp, batch_val['codec_type'], x_sr_pred.clone(), accelerator.device)
                    qp_minus = (batch_val['qp'].to(torch.int32) - 10).clamp_(0, 51)
                    qp_minus_bpp, x_reg, qp_minus_prompt = real_codec_proc_with_prompt(batch_val['codec_type'], qp_minus, x_tgt.detach(), accelerator.device)
                    x_tgt_ori_size = (x_tgt + 1.0) / 2.0
                    tgt_deg = net_de(x_tgt_ori_size).detach()
                    temb_qp_minus_sim = build_temb(batch_val['codec_type'], qp_minus_bpp, device=accelerator.device)
                    x_reg, _ = net_sim(x_tgt, tgt_deg, qp_minus_prompt, meta_vec=temb_qp_minus_sim)
                torchvision.utils.save_image(x_codec_pred.cpu().detach(), os.path.join(sr_pred_dir, f"{batch_val['image_name'][0]}"), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(x_tgt.cpu().detach(), os.path.join(sr_gt_dir, f"{batch_val['image_name'][0]}"), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(x_reg.cpu().detach(), os.path.join(sr_reg_dir, f"{batch_val['image_name'][0]}"), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(x_sr_pred.cpu().detach(), os.path.join(sr_dir, f"{batch_val['image_name'][0]}"), normalize=True, value_range=(-1, 1))
                with torch.cuda.amp.autocast(enabled=False):
                    lp = net_lpips(x_codec_pred.float(), x_tgt.float()).mean().item()
                lpips_vals.append(lp)
                l2_vals.append(F.mse_loss(x_codec_pred, x_tgt).item())
            sr_results = evaluate(sr_pred_dir, sr_gt_dir, None, accelerator.device)
            with open(os.path.join(sr_pred_dir, 'results.txt'), 'w', encoding='utf-8') as f:
                for item in sr_results:
                    f.write(f'{item}\n')
            if args.wandb:
                wandb.log({'val_sr/L2': float(np.mean(l2_vals)) if len(l2_vals) else 0.0, 'val_sr/LPIPS': float(np.mean(lpips_vals)) if len(lpips_vals) else 0.0}, step=global_step)
            accelerator.unwrap_model(net_sr).set_train()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(net_sr).save_model(os.path.join(args.output_dir, 'checkpoints', 'sr_final.pkl'))
    if args.wandb and accelerator.is_main_process:
        wandb.finish()
if __name__ == '__main__':
    main(args)
