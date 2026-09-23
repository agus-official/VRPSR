import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import codecsimulator
CODEC_CLASSES = {'libx264': (codecsimulator.PYRGBx264Codec, 'h264'), 'libx265': (codecsimulator.PYRGBx265Codec, 'h265'), 'libx266': (codecsimulator.PYRGBx266Codec, 'h266')}
_CODEC2IDX = {'libx264': 0, 'libx265': 1, 'libx266': 2}

def proc_h26x(img, qp, codec_class, codec_name, width=512, height=512):
    codec = codec_class(width, height)
    img_np = np.asarray(img)
    out_np = np.zeros_like(img_np)
    packet_size = codec.Encode(img_np, out_np, qp)
    rec = Image.fromarray(out_np, 'RGB')
    packet_size = packet_size / 1000
    return (img, rec, packet_size)

def proc(img, codec_type, qp):
    codec_data = CODEC_CLASSES.get(codec_type)
    if codec_data is None:
        raise ValueError(f'Unsupported codec type: {codec_type}')
    codec_class, codec_name = codec_data
    return proc_h26x(img, qp, codec_class, codec_name)

def get_prompt(blur_img, codec_type, qp):
    prompts = []
    blur_264_imgs = []
    bpps = []
    for i in range(blur_img.size(0)):
        img_pil = transforms.ToPILImage()(blur_img[i].cpu())
        _, blur_264, bpp = proc(img_pil, codec_type[i], qp[i].cpu().item())
        prompts.append('A {0} {1:.4f} bpp compressed image.'.format(codec_type[i], bpp))
        blur_264_imgs.append(blur_264)
        bpps.append(bpp)
    to_tensor = transforms.ToTensor()
    blur_264_imgs = [to_tensor(img) for img in blur_264_imgs]
    blur_264_imgs = torch.stack(blur_264_imgs)
    bpps = torch.tensor(bpps, dtype=torch.float32)
    return (prompts, blur_264_imgs, bpps)

def real_codec_proc_with_prompt(tgt_codec, tgt_qp, input_img, device):
    x_codec_preds = []
    bpps = []
    prompts = []
    for i in range(input_img.size(0)):
        img = input_img[i] * 0.5 + 0.5
        img = torch.clamp(img, 0, 1)
        img = transforms.ToPILImage()(img.cpu())
        _, x_codec_pred, bpp = proc(img, codec_type=tgt_codec[i], qp=tgt_qp[i].cpu().item())
        if isinstance(x_codec_pred, Image.Image):
            x_codec_pred = transforms.ToTensor()(x_codec_pred)
        bpps.append(bpp)
        x_codec_preds.append(x_codec_pred)
        prompts.append('A {0} {1:.4f} bpp compressed image.'.format(tgt_codec[i], bpp))
    bpps = torch.tensor(bpps, dtype=torch.float32)
    x_codec_preds = torch.stack(x_codec_preds)
    x_codec_preds = x_codec_preds * 2.0 - 1.0
    x_codec_preds = torch.clamp(x_codec_preds, -1, 1)
    return (bpps, x_codec_preds.to(device), prompts)

def find_best_qp(tgt_bpp, tgt_codec, input_img, device, qp_min=25, qp_max=51):
    best_qps = []
    best_bpps = []
    best_x_codec_preds = []
    for i in range(input_img.size(0)):
        img = input_img[i]
        img = torch.clamp(img, -1, 1)
        img = img * 0.5 + 0.5
        left, right = (qp_min, qp_max)
        best_qp = qp_min
        best_bpp = float('inf')
        best_x_codec_pred = None
        while left <= right:
            qp_mid = (left + right) // 2
            img_pil = transforms.ToPILImage()(img.cpu())
            _, x_codec_pred, bpp = proc(img_pil, codec_type=tgt_codec[i], qp=qp_mid)
            if isinstance(x_codec_pred, Image.Image):
                x_codec_pred = transforms.ToTensor()(x_codec_pred)
            target = tgt_bpp[i].item()
            if abs(bpp - target) < abs(best_bpp - target):
                best_bpp = bpp
                best_qp = qp_mid
                best_x_codec_pred = x_codec_pred
            if bpp > target:
                left = qp_mid + 1
            else:
                right = qp_mid - 1
        best_qps.append(best_qp)
        best_bpps.append(best_bpp)
        best_x_codec_preds.append(best_x_codec_pred)
    best_qps = torch.tensor(best_qps, dtype=torch.int)
    best_bpps = torch.tensor(best_bpps, dtype=torch.float32)
    best_x_codec_preds = torch.stack(best_x_codec_preds)
    best_x_codec_preds = best_x_codec_preds * 2.0 - 1.0
    best_x_codec_preds = torch.clamp(best_x_codec_preds, -1, 1)
    return (best_qps, best_bpps, best_x_codec_preds.to(device))

def build_temb(codec_type_batch, tgt_bpp, device=None):
    """
    将 codec 类型 + bpp 拼成 (B, 4) 的 temb 张量.

    Args:
        codec_type_batch (Iterable[str]): 长度为 B 的字符串列表，如 ["x264", "x265", ...]
        tgt_bpp (Tensor | Iterable[float]): 形状 (B,) 的 bpp 数值.
        device (torch.device | str | None): 输出张量所在设备; 默认沿用 tgt_bpp.device.

    Returns:
        Tensor: (B, 4) float32, 值为 one‑hot(codec) ‑‑ concat ‑‑ bpp
    """
    codec_idx = torch.tensor([_CODEC2IDX[c] for c in codec_type_batch], dtype=torch.long, device=device)
    codec_onehot = F.one_hot(codec_idx, num_classes=3).float()
    if not torch.is_tensor(tgt_bpp):
        tgt_bpp = torch.tensor(tgt_bpp, dtype=torch.float32)
    if device is not None:
        tgt_bpp = tgt_bpp.to(device)
    bpp_col = tgt_bpp.view(-1, 1).float()
    temb = torch.cat([codec_onehot, bpp_col], dim=1)
    return temb
