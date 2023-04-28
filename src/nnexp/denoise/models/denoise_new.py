from typing import List, Literal, Tuple
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import einops

from nnexp import base_model
from nnexp.images import conv_types
from .model_shared import SinPositionEmbedding, SelfAttention, CrossAttention, CrossAttentionConv
from diffusers import Transformer2DModel

# TODO: fix the unprocessed borders around output - probably need to run @ higher res?

EmbedPos = Literal['first', 'last',          # first and last of the entire down stack
                   'res_first', 'res_last',  # first and last of a given resolution/channel
                   'up_first', 'up_last',
                   'up_res_first', 'up_res_last',
                   ]

# channels = [128, 256]
# - DownResBlock(
#     Down(128, 128, s=1),
#     Down(128, 256, s=2)
#   )
# - DownResBlock(
#     Down(256, 256, s=1),
#     Down(256, 256, s=1)
#   )
#
# - UpResBlock(
#     Up(256, 256, s=1),
#     Up(256, 256, s=1),
#   )
# - UpResBlock(
#     Up(256, 128, s=2),
#     Up(128, 128, s=1),
#   )

@dataclass(kw_only=True)
class Config:
    # time_pos: List[EmbedPos]

    # sa_nheads: int
    # sa_pos: List[EmbedPos]
    groupnorm_ngroups: int
    nonlinearity: conv_types.ConvNonlinearity

    down_channels: List[int]
    ngroups: int
    nstride1: int

    input_chan: int  # chan, size for whole network
    input_size: int

    @property
    def time_emblen(self) -> int:
        return max(self.down_channels)

    def in_channels(self, dir: conv_types.Direction) -> List[int]:
        if dir == 'up':
            return list(reversed(self.out_channels('down')))
        return [self.input_chan] + self.down_channels
    
    def out_channels(self, dir: conv_types.Direction) -> List[int]:
        if dir == 'up':
            return list(reversed(self.in_channels('down')))
        return self.down_channels + [self.down_channels[-1]]

class ApplyTimeEmbedding(nn.Module):
    def __init__(self, *, chan: int, time_emblen: int):
        super().__init__()
        self.time_mean_std = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emblen, chan * 2),
        )
    
    def forward(self, inputs: Tensor, time_embed: Tensor) -> Tensor:
        mean_std = self.time_mean_std(time_embed)
        mean_std = einops.rearrange(mean_std, "b c -> b c 1 1")
        mean, std = mean_std.chunk(2, dim=1)

        out = inputs * (std + 1) + mean
        return out

class ResnetBlock2D(nn.Sequential):
    def __init__(self, in_chan: int, out_chan: int, cfg: Config):
        super().__init__()

        self.norm1 = nn.GroupNorm(cfg.groupnorm_ngroups, in_chan)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = ApplyTimeEmbedding(chan=out_chan, time_emblen=cfg.time_emblen)
        self.norm2 = nn.GroupNorm(cfg.groupnorm_ngroups, out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = cfg.nonlinearity.create()
    
    def forward(self, input: Tensor, time_embed: Tensor) -> Tensor:
        out = input
        for mod in self:
            # print()
            # print(f"  res2d:  in = {out.shape} {type(mod).__name__}")
            if isinstance(mod, ApplyTimeEmbedding):
                out = mod.forward(out, time_embed=time_embed)
            else:
                out = mod.forward(out)
            # print(f"  res2d: out = {out.shape}")
        return out

class ResGroup(nn.Sequential):
    def forward(self, input: Tensor, time_embed: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tensor:
        out = input
        for mod in self:
            # print(f"resgroup:  in = {out.shape} {type(mod).__name__}")
            if isinstance(mod, ResnetBlock2D):
                out = mod.forward(out, time_embed)
            else:
                out = mod.forward(out)
            # print(f"resgroup: out = {out.shape}")
        return out

class DownResGroup(ResGroup):
    def __init__(self, in_chan: int, out_chan: int, cfg: Config):
        super().__init__()
        # self.attentions = nn.ModuleList()

        for _ in range(cfg.nstride1):
            self.append(ResnetBlock2D(in_chan=in_chan, out_chan=in_chan, cfg=cfg))

        if in_chan != out_chan:
            self.downsample = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1)

class DownHalf(nn.Sequential):
    def __init__(self, cfg: Config):
        super().__init__()

        in_size = cfg.input_size
        in_chans = cfg.in_channels('down')
        out_chans = cfg.out_channels('down')

        self.in_conv = nn.Conv2d(in_chans[0], out_chans[0], kernel_size=1, stride=1)

        for in_chan, out_chan in zip(in_chans[1:], out_chans[1:]):
            for i in range(cfg.ngroups):
                self.append(DownResGroup(in_chan=in_chan, out_chan=in_chan, cfg=cfg))
            self.append(DownResGroup(in_chan=in_chan, out_chan=out_chan, cfg=cfg))

            if in_chan != out_chan:
                in_size //= 2
        
        self.out_chan = out_chan
        self.out_size = in_size
    
    def forward(self, inputs: Tensor, time_embed: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tuple[Tensor, List[Tensor]]:
        out = inputs

        out_list: List[Tensor] = list()
        for mod in self:
            if isinstance(mod, DownResGroup):
                out = mod.forward(out, time_embed, clip_embed, clip_scale)
                out_list.append(out)
            else:
                out = mod.forward(out)

        return out, out_list

class UpResGroup(ResGroup):
    def __init__(self, in_chan: int, out_chan: int, cfg: Config):
        super().__init__()

        # self.attentions = nn.Sequential()

        self.append(nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1))
        for idx in range(cfg.nstride1):
            self.append(ResnetBlock2D(in_chan=in_chan, out_chan=in_chan, cfg=cfg))

        if in_chan != out_chan:
            self.upsample = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1)

class UpHalf(nn.Sequential):
    def __init__(self, cfg: Config):
        super().__init__()

        in_size = cfg.input_size
        in_chans = cfg.in_channels('up')
        out_chans = cfg.out_channels('up')

        for in_chan, out_chan in zip(in_chans[:-1], out_chans[:-1]):
            for i in range(cfg.ngroups):
                # self.append(UpResGroup(in_chan=in_chan * 2, out_chan=in_chan * 2, cfg=cfg))
                self.append(UpResGroup(in_chan=in_chan, out_chan=in_chan, cfg=cfg))
            # self.append(UpResGroup(in_chan=in_chan * 2, out_chan=out_chan, cfg=cfg))
            self.append(UpResGroup(in_chan=in_chan, out_chan=out_chan, cfg=cfg))

            if in_chan != out_chan:
                in_size *= 2
        
        self.out_conv = nn.Conv2d(in_chans[-1], out_chans[-1], kernel_size=1, stride=1)
    
    def forward(self, inputs: Tensor, out_list: List[Tensor],
                time_embed: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tensor:
        out = inputs
        out_list = list(out_list)

        for mod in self:
            if isinstance(mod, UpResGroup):
                down_out = out_list.pop()
                out = torch.cat([out, down_out], dim=1)
                out = mod.forward(out, time_embed, clip_embed, clip_scale)
            else:
                out = mod.forward(out)

        return out

class DenoiseModelNew(base_model.BaseModel):
    _model_fields = ("in_chan in_size channels ngroups nstride1 "
                     "clip_scale_default nonlinearity_type").split()
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    # clip_emblen: int
    clip_scale_default: float

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 channels: List[int], 
                 ngroups: int, nstride1: int, 
                 clip_scale_default: float = 1.0,
                 nonlinearity_type: conv_types.NlType = 'silu'):
        super().__init__()

        lat_size = in_size // (2 ** (len(channels) - 1))
        self.in_dim = [in_chan, in_size, in_size]
        self.latent_dim = [channels[-1], lat_size, lat_size]

        nonlinearity = conv_types.ConvNonlinearity(nonlinearity_type)
        cfg = Config(
            nonlinearity=nonlinearity,
            down_channels=channels, 
            ngroups=ngroups, nstride1=nstride1,
            groupnorm_ngroups=8,
            input_chan=in_chan, input_size=in_size
        )

        time_emblen = cfg.time_emblen
        self.gen_time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down = DownHalf(cfg=cfg)
        # down_chan = self.down.out_chan
        # self.cross_attn = Transformer2DModel(in_channels=down_chan, out_channels=down_chan, norm_num_groups=8)
        self.up = UpHalf(cfg=cfg)

        self.channels = channels
        self.ngroups = ngroups
        self.nstride1 = nstride1
        self.in_chan = in_chan
        self.in_size = in_size
        # self.time_pos = time_pos
        # self.sa_nheads = sa_nheads
        # self.sa_pos = sa_pos
        self.clip_scale_default = clip_scale_default
        self.nonlinearity_type = nonlinearity_type
    
    def _get_time_emb(self, inputs: Tensor, time: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb is not None:
            return time_emb

        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        return self.gen_time_emb(time)

    def forward(self, inputs: Tensor, time: Tensor = None, clip_embed: Tensor = None, clip_scale: Tensor = None) -> Tensor:
        batch = inputs.shape[0]
        if clip_scale is None:
            clip_scale = torch.ones((batch, 1, 1, 1), device=inputs.device, dtype=inputs.dtype) * self.clip_scale_default
        elif len(clip_scale.shape) == 1:
            clip_scale = clip_scale.view(batch, 1, 1, 1)

        if clip_embed is None:
            clip_embed = torch.zeros((batch, 7, 768), device=inputs.device, dtype=inputs.dtype)

        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)

        # down
        out, down_list = self.down.forward(inputs=inputs, time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)

        # mid
        # xformer_out = self.cross_attn.forward(hidden_states=out,
        #                                     encoder_hidden_states=clip_embed,
        #                                     timestep=time)
        # out = xformer_out.sample

        # up
        out = self.up.forward(out, down_list,
                              time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)

        return out

