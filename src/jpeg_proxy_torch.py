import torch
import torch.nn.functional as F
import numpy as np
import itertools


def pad_to_multiple(x, bsize):
    h, w = x.shape[-3:-1]  # [B, H, W, C]
    pad_h = (bsize - h % bsize) % bsize
    pad_w = (bsize - w % bsize) % bsize
    return F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode='replicate')


def get_dct_matrix(dct_size):
    dct_mat = np.zeros((dct_size, dct_size), dtype=np.float32)
    for i, j in itertools.product(range(dct_size), repeat=2):
        dct_mat[i, j] = np.cos((2 * i + 1) * j * np.pi / (2 * dct_size))
    dct_mat *= np.sqrt(2 / dct_size)
    dct_mat[:, 0] *= 1 / np.sqrt(2)
    return torch.tensor(dct_mat, dtype=torch.float32)

def ste_round(x):
    # forward: round(x)；backward: d/dx ≈ 1
    return (x - x.detach()) + torch.round(x).detach()


class JpegProxy(torch.nn.Module):
    def __init__(
        self,
        downsample_chroma,
        luma_quant_table,
        chroma_quant_table,
        convert_to_yuv=True,
        clip_to_image_max=True,
        dct_size=8,
        upsample_method='bilinear'
    ):
        super().__init__()
        self.downsample_chroma = downsample_chroma
        self.convert_to_yuv = convert_to_yuv
        self.clip_to_image_max = clip_to_image_max
        self.dct_size = dct_size
        self.upsample_method = upsample_method

        # DCT Matrix
        self.dct_mat = get_dct_matrix(dct_size)
        self.idct_mat = self.dct_mat.t()
        self.dct2d_mat = torch.kron(self.dct_mat, self.dct_mat)

        # Quant tables
        self.luma_quant_table = torch.tensor(luma_quant_table, dtype=torch.float32).view(-1)
        self.chroma_quant_table = torch.tensor(chroma_quant_table, dtype=torch.float32).view(-1)

        # RGB <-> YUV matrix
        self.rgb_from_yuv = torch.tensor([
            [1.0, 1.0, 1.0],
            [0.0, -0.344136, 1.772],
            [1.402, -0.714136, 0.0],
        ], dtype=torch.float32)

        self.yuv_from_rgb = torch.tensor([
            [0.299, -0.168736, 0.5],
            [0.587, -0.331264, -0.418688],
            [0.114, 0.5, -0.081312],
        ], dtype=torch.float32)

    def _rgb_to_yuv(self, x):
        return torch.tensordot(x, self.yuv_from_rgb.to(x.device), dims=([3], [0])) + \
               torch.tensor([0, 128, 128], dtype=x.dtype, device=x.device)

    def _yuv_to_rgb(self, x):
        return torch.tensordot(x - torch.tensor([0, 128, 128], dtype=x.dtype, device=x.device),
                               self.rgb_from_yuv.to(x.device), dims=([3], [0]))

    def _forward_dct(self, x):
        device = x.device
        b, h, w = x.shape[:3]

        x = x.unfold(1, self.dct_size, self.dct_size) \
            .unfold(2, self.dct_size, self.dct_size)
        x = x.contiguous().view(b, -1, self.dct_size * self.dct_size)

        x = x - torch.tensor(128.0, dtype=x.dtype, device=device)

        return torch.matmul(x, self.dct2d_mat.to(device))


    def _inverse_dct(self, coeffs, height, width):
        device = coeffs.device
        b, blocks, _ = coeffs.shape

        # IDCT
        x = torch.matmul(coeffs, self.dct2d_mat.t().to(device)) + \
            torch.tensor(128.0, dtype=coeffs.dtype, device=device)

        # block → full image
        x = x.view(b,
                height // self.dct_size,
                width  // self.dct_size,
                self.dct_size,
                self.dct_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(b, height, width, 1)
        return x

    def forward(self, image, rounding_fn=ste_round):
        b, c, h, w = image.shape
        assert c == 3, "Input must have 3 channels"

        image = image * 255.0

        image = image.permute(0, 2, 3, 1)           # BCHW → BHWC
        orig_h, orig_w = h, w

        pad_multiple = self.dct_size * (2 if self.downsample_chroma else 1)
        image = pad_to_multiple(image, pad_multiple)
        padded_h, padded_w = image.shape[1:3]

        if self.convert_to_yuv:
            image = self._rgb_to_yuv(image)

        quantized_dct = {}
        output = []

        for ch, key in zip(range(3), ['y', 'u', 'v']):
            channel = image[..., ch:ch+1]

            if ch > 0 and self.downsample_chroma:
                channel = F.interpolate(
                    channel.permute(0, 3, 1, 2),
                    scale_factor=0.5,
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)

            coeffs = self._forward_dct(channel)
            quant_table = self.luma_quant_table if ch == 0 else self.chroma_quant_table
            quant_table = quant_table.to(image.device)
            q_coeffs = rounding_fn(coeffs / quant_table)
            quantized_dct[key] = q_coeffs

            dequant = q_coeffs * quant_table
            recon = self._inverse_dct(dequant, channel.shape[1], channel.shape[2])

            if ch > 0 and self.downsample_chroma:
                recon = F.interpolate(
                    recon.permute(0, 3, 1, 2),
                    size=(padded_h, padded_w),
                    mode=self.upsample_method,
                    align_corners=False
                ).permute(0, 2, 3, 1)

            output.append(recon)

        recon_image = torch.cat(output, dim=-1)

        if self.convert_to_yuv:
            recon_image = self._yuv_to_rgb(recon_image)

        recon_image = recon_image[:, :orig_h, :orig_w, :]
        recon_image = torch.clamp(recon_image, 0.0, 255.0) / 255.0

        # BHWC → BCHW
        return recon_image.permute(0, 3, 1, 2), quantized_dct


class JpegWrapper(torch.nn.Module):
    def __init__(self, convert_to_yuv=True, downsample_chroma=True, clip_to_image_max=True, jpeg_quantizer_fn=ste_round):
        super().__init__()
        quality_factor = 2
        quant_table = np.ones((8, 8), dtype=np.float32) * (100 / quality_factor)
        self.jpeg = JpegProxy(
            downsample_chroma=downsample_chroma,
            luma_quant_table=quant_table,
            chroma_quant_table=quant_table,
            convert_to_yuv=convert_to_yuv,
            clip_to_image_max=clip_to_image_max,
        )
        self.rounding_fn = jpeg_quantizer_fn

    def forward(self, inputs):
        if inputs.shape[1] < 3:
            pad = torch.zeros(inputs.shape[0], 3 - inputs.shape[1], inputs.shape[2], inputs.shape[3], dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat([inputs, pad], dim=1)

        jpeg_output, _ = self.jpeg(inputs, rounding_fn=self.rounding_fn)
        return jpeg_output
