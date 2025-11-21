import torch
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
import math

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_embedding(timesteps, embedding_dim):
    """
    timesteps: (B, 4)  [c1, c2, c3, bpp]
    embedding_dim: total dim (should be divisible by 4)
    """
    embedding_outs = []
    assert timesteps.shape[1] == 4
    for i in range(4):
        embedding_out = get_timestep_embedding(timesteps[:, i], embedding_dim // 4)
        embedding_outs.append(embedding_out)
    return torch.cat(embedding_outs, dim=1)  # (B, embedding_dim)


class ResnetBlock2DWithMeta(ResnetBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=None,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="silu",
        skip_time_act=False,
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        meta_dim: int = 256,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels,
            groups=groups,
            groups_out=groups_out,
            pre_norm=pre_norm,
            eps=eps,
            non_linearity=non_linearity,
            skip_time_act=skip_time_act,
            time_embedding_norm=time_embedding_norm,
            kernel=kernel,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
        )

        self.out_channels = out_channels or in_channels
        self.meta_dim = meta_dim

        if meta_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(meta_dim, self.out_channels * 4),
                nn.SiLU(),
                nn.Linear(self.out_channels * 4, self.out_channels * 2)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        self.meta_emb = None

        if hasattr(self, "time_emb_proj") and not hasattr(self, "time_embedding_proj"):
            self.time_embedding_proj = self.time_emb_proj

    def set_meta_emb(self, meta_emb):
        self.meta_emb = meta_emb

    def forward(self, input_tensor, temb=None, **kwargs):
        scale = kwargs.get("scale", 1.0)
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_embedding_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(input_tensor)
        else:
            shortcut = input_tensor

        # meta_vec
        if self.meta_emb is not None and hasattr(self, 'mlp'):
            modulation = self.mlp(self.meta_emb)  # [B, C*2]
            shift, scale = modulation.chunk(2, dim=1)
            hidden_states = hidden_states * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        output_tensor = self.output_scale_factor * (shortcut + hidden_states)
        return output_tensor

class UNet2DConditionModelWithMeta(UNet2DConditionModel):
    """
    Extended UNet that accepts external meta_vec for conditional control.
    Usage: forward(..., meta_vec=meta_vec)  # meta_vec: (B, 4) [c1,c2,c3,bpp]
    """

    def __init__(self, meta_dim: int = 1024, **unet_kwargs):
        super().__init__(**unet_kwargs)
        self.meta_dim = meta_dim
        for name, module in list(self.named_modules()):
            if isinstance(module, ResnetBlock2D):
                parent, attr = name.rsplit('.', 1) if '.' in name else ('', name)
                container = self.get_submodule(parent) if parent else self
                setattr(container, attr, ResnetBlock2DWithMeta(module, meta_dim))
        self.meta_dim = meta_dim

    @classmethod
    def from_pretrained(
        cls, model_id: str, *args,
        bpp_embed_dim: int = 128,
        meta_dim: int = 256,
        **kwargs
    ):
        model = UNet2DConditionModel.from_pretrained(model_id, *args, **kwargs)
        model = cls.convert_to_with_meta(model, meta_dim=meta_dim, bpp_embed_dim=bpp_embed_dim, unet_cls=cls)
        model.__class__ = cls
        return model

    @staticmethod
    def convert_to_with_meta(model, meta_dim=256, bpp_embed_dim=128, unet_cls=None):
        from unet_with_meta import ResnetBlock2DWithMeta

        blocks_to_replace = []
        ref_block_type = type(model.down_blocks[0].resnets[0])
        for name, module in model.named_modules():
            if isinstance(module, ref_block_type):
                parent_name = ".".join(name.split(".")[:-1])
                child_name  = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                blocks_to_replace.append((parent, child_name, module))

        for parent, child_name, old_block in blocks_to_replace:
            kwargs = dict(
                in_channels          = old_block.in_channels,
                out_channels         = old_block.out_channels,
                conv_shortcut        = getattr(old_block, "use_conv_shortcut", False),
                dropout              = old_block.dropout.p if hasattr(old_block, "dropout") else 0.0,
                temb_channels        = (
                    getattr(old_block, "time_emb_proj", None)  or
                    getattr(old_block, "time_embedding_proj", None)
                )
            )
            if kwargs["temb_channels"] is not None:
                kwargs["temb_channels"] = kwargs["temb_channels"].in_features
            kwargs.update(
                groups               = old_block.norm1.num_groups,
                groups_out           = old_block.norm2.num_groups,
                pre_norm             = old_block.pre_norm,
                eps                  = old_block.norm1.eps,
                non_linearity        = 'silu',
                skip_time_act        = old_block.skip_time_act,
                time_embedding_norm  = old_block.time_embedding_norm,
                output_scale_factor  = old_block.output_scale_factor,
                use_in_shortcut      = old_block.use_in_shortcut,
                up                   = old_block.up,
                down                 = old_block.down,
                meta_dim             = meta_dim,
            )
            if old_block.conv2.out_channels != old_block.out_channels:
                kwargs["conv_2d_out_channels"] = old_block.conv2.out_channels

            new_block = ResnetBlock2DWithMeta(**kwargs)
            new_block.load_state_dict(old_block.state_dict(), strict=False)

            setattr(parent, child_name, new_block)

        model.meta_dim = meta_dim
        if hasattr(unet_cls, "_init_bpp_head"):
            model._init_bpp_head(bpp_embed_dim)

        return model

    def forward(self, sample, timestep, encoder_hidden_states, *,
                meta_vec=None, return_dict=True, **kwargs):
        # meta_emb = get_embedding(meta_vec, self.meta_dim) if meta_vec is not None else None

        meta_emb = None
        if meta_vec is not None:
            meta_emb = get_embedding(meta_vec, self.meta_dim)  # (B, meta_dim)

        touched = []
        if meta_emb is not None:
            for m in self.modules():
                if isinstance(m, ResnetBlock2DWithMeta):
                    m.set_meta_emb(meta_emb)
                    touched.append(m)
        try:
            return super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        finally:
            for m in touched:
                m.set_meta_emb(None)