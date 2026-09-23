from vrpsr_training.options import parse_sim_args as parse_args_paired_training
if __name__ == '__main__':
    args = parse_args_paired_training()
from vrpsr_training.evaluation import evaluate
from vrpsr_training.codec import build_temb
from vrpsr_training.data import CodecDataset
import os
import json
import importlib.metadata
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import lpips
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from de_net import DEResNet
from s3diff_meta_embed import S3DiffMetaEmbed
from unet_with_meta import ResnetBlock2DWithMeta
import wandb
import datetime
import torchvision

def verify_meta_mlp_trainable(unet_with_meta):
    """验证所有 ResnetBlock2DWithMeta 的 mlp 是否已启用梯度"""
    mlps = [m.mlp for m in unet_with_meta.modules() if isinstance(m, ResnetBlock2DWithMeta)]
    assert len(mlps) > 0, 'No ResnetBlock2DWithMeta found! Check model conversion.'
    for i, mlp in enumerate(mlps):
        for name, param in mlp.named_parameters():
            assert param.requires_grad, f'MLP {i}.{name} is not trainable!'
    print(f'✅ All {len(mlps)} meta-mlps are trainable.')

def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=None)
    if args.wandb and accelerator.is_main_process:
        wandb.init(project='vrpsr_simulator', name=f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", config=args, mode='offline')
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id='stabilityai/sd-turbo')
    else:
        sd_path = args.sd_path
    print('Downloaded model path:', sd_path)
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
        os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()
    net_de.requires_grad_(False)
    net_sim = S3DiffMetaEmbed(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=args.pretrained_path, random_unet=False)
    net_sim.set_train()
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sim.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError('xformers is not available, please install it by running `pip install xformers`')
    if args.gradient_checkpointing:
        net_sim.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    import vision_aided_loss
    net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=None, device='cuda')
    net_disc.disc_loss = vision_aided_loss.cv_losses.losses_list(args.gan_loss_type)
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    verify_meta_mlp_trainable(net_sim.unet)
    layers_to_opt = []
    layers_to_opt = layers_to_opt + list(net_sim.vae_block_embeddings.parameters()) + list(net_sim.unet_block_embeddings.parameters())
    layers_to_opt = layers_to_opt + list(net_sim.vae_de_mlp.parameters()) + list(net_sim.unet_de_mlp.parameters()) + list(net_sim.vae_block_mlp.parameters()) + list(net_sim.unet_block_mlp.parameters()) + list(net_sim.vae_fuse_mlp.parameters()) + list(net_sim.unet_fuse_mlp.parameters())
    for n, _p in net_sim.unet.named_parameters():
        if 'lora' in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_sim.unet.conv_in.parameters())
    for n, _p in net_sim.vae.named_parameters():
        if 'lora' in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    meta_mlp_params = []
    for m in net_sim.unet.modules():
        if isinstance(m, ResnetBlock2DWithMeta):
            meta_mlp_params.extend(m.mlp.parameters())
    meta_mlp_params = [p for p in meta_mlp_params if p.requires_grad]
    assert len(meta_mlp_params) > 0, 'No meta-mlp parameters found! Check model conversion.'
    layers_to_opt += meta_mlp_params
    dataset_train = CodecDataset(args.train_data, qp=args.fixed_qp, codec_type=args.fixed_codec_type)
    dataset_val = CodecDataset(args.val_data, qp=args.fixed_qp_val, codec_type=args.fixed_codec_type_val)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, persistent_workers=True if args.dataloader_num_workers > 0 else False)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    disc_lr = args.disc_learning_rate if args.disc_learning_rate is not None else args.learning_rate * args.disc_lr_scale
    print(f'disc_lr: {disc_lr}')
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=disc_lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    net_sim, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(net_sim, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc)
    net_de, net_lpips = accelerator.prepare(net_de, net_lpips)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    net_sim.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    for name, module in net_disc.named_modules():
        if 'attn' in name:
            module.fused_attn = False
    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc='Steps', disable=not accelerator.is_local_main_process)
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, (x_src, x_tgt, pos_tag_prompt, name, bpp, codec, qp) in enumerate(dl_train):
            l_acc = [net_sim]
            l_acc.append(net_disc)
            with accelerator.accumulate(*l_acc):
                x_src, x_tgt = (x_src.to(accelerator.device), x_tgt.to(accelerator.device))
                x_ori_size_src = (x_src + 1.0) / 2.0
                B, C, H, W = x_src.shape
                with torch.no_grad():
                    deg_score = net_de(x_ori_size_src.detach()).detach()
                if accelerator.is_local_main_process and global_step == 0:
                    print('[Train] Sample prompt:', pos_tag_prompt[0])
                mixed_tag_prompt = pos_tag_prompt
                mixed_tgt = x_tgt
                temb = build_temb(codec, bpp, device=accelerator.device)
                x_tgt_pred, _ = net_sim(x_src.detach(), deg_score, mixed_tag_prompt, meta_vec=temb)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction='mean') * args.lambda_l2
                loss = loss_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() * args.lambda_lpips
                loss += loss_lpips
                fake_out_G = net_disc(x_tgt_pred)
                lossG = accelerator.unwrap_model(net_disc).disc_loss(fake_out_G, for_G=True).mean() * args.lambda_gan
                loss += lossG
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                disc_out = net_disc(torch.cat([x_tgt.detach(), x_tgt_pred.detach()], dim=0), detach=True)[0]
                batch_out = [torch.chunk(disc_out_i, dim=0, chunks=2) for disc_out_i in disc_out]
                batch_out_T = [list(x) for x in zip(*batch_out)]
                real_out, fake_out = batch_out_T
                lossD_real = accelerator.unwrap_model(net_disc).disc_loss([real_out], for_real=True).mean() * args.lambda_gan
                lossD_fake = accelerator.unwrap_model(net_disc).disc_loss([fake_out], for_real=False).mean() * args.lambda_gan
                lossD = lossD_real + lossD_fake
                accelerator.backward(lossD.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logs = {'loss_l2': loss_l2.item()}
                    logs['loss_lpips'] = loss_lpips.item()
                    logs['lossD'] = lossD.item()
                    progress_bar.set_postfix(**logs)
                    if args.wandb:
                        wandb.log(logs, step=global_step)
                    if (global_step - 1) % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, 'checkpoints', f'lora_{global_step}.pkl')
                        accelerator.unwrap_model(net_sim).save_model(outf)
                    if args.eval_freq > 0 and (global_step - 1) % args.eval_freq == 0:
                        accelerator.unwrap_model(net_sim).set_eval()
                        l_l2, l_lpips = ([], [])
                        for step, (x_src, x_tgt, pos_tag_prompt, name, bpp, codec, qp) in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src, x_tgt = (x_src.to(accelerator.device), x_tgt.to(accelerator.device))
                            x_ori_size_src = (x_src + 1.0) / 2.0
                            B, C, H, W = x_src.shape
                            assert B == 1, 'Use batch size 1 for eval.'
                            with torch.no_grad():
                                with torch.no_grad():
                                    deg_score = net_de(x_ori_size_src.detach())
                                temb = build_temb(codec, bpp, device=accelerator.device)
                                x_tgt_pred, intermediates = accelerator.unwrap_model(net_sim)(x_src.detach(), deg_score, pos_tag_prompt, meta_vec=temb)
                            rec_img = x_tgt.cpu().detach() * 0.5 + 0.5
                            rec_pil = transforms.ToPILImage()(rec_img[0])
                            rec_subdir = os.path.join(args.output_dir, f'{global_step}_val', 'real')
                            os.makedirs(rec_subdir, exist_ok=True)
                            rec_path = os.path.join(rec_subdir, f'{name[0]}.png')
                            rec_pil.save(rec_path)
                            x_tgt_pred = x_tgt_pred.cpu().detach()
                            output_subdir = os.path.join(args.output_dir, f'{global_step}_val', 'sim_val')
                            os.makedirs(output_subdir, exist_ok=True)
                            out_path = os.path.join(output_subdir, f'{name[0]}.png')
                            torchvision.utils.save_image(x_tgt_pred.cpu().detach(), out_path, normalize=True, value_range=(-1, 1))
                        val_results = evaluate(output_subdir, rec_subdir, None, accelerator.device)
                        sim_t = os.path.join(output_subdir, 'results.txt')
                        with open(sim_t, 'w', encoding='utf-8') as f:
                            for item in val_results:
                                f.write(f'{item}\n')
                        if args.wandb:
                            wandb.log(logs, step=global_step)
                        accelerator.unwrap_model(net_sim).set_train()
                accelerator.wait_for_everyone()
                if args.stop_steps is not None and global_step >= args.stop_steps:
                    break
        if args.stop_steps is not None and global_step >= args.stop_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(net_sim).save_model(os.path.join(args.output_dir, 'checkpoints', 'simulator_final.pkl'))
    if args.wandb and accelerator.is_main_process:
        wandb.finish()
if __name__ == '__main__':
    main(args)
