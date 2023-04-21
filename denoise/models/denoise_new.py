from typing import List, Literal, Tuple
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import einops
import torch.nn.functional as F

import base_model
import conv_types
from .model_shared import SinPositionEmbedding, SelfAttention, CrossAttention


# TODO: fix the unprocessed borders around output - probably need to run @ higher res?

EmbedPos = Literal['first', 'last',          # first and last of the entire down stack
                   'res_first', 'res_last',  # first and last of a given resolution/channel
                   ]

#
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
    time_emblen: int
    time_pos: EmbedPos = 'last'

    sa_nheads: int
    sa_pos: EmbedPos = 'begin'
    ca_nheads: int
    ca_pos: EmbedPos = 'end'
    clip_emblen: int
    nonlinearity: conv_types.ConvNonlinearity

    down_channels: List[int]
    nstride1: int

    input_chan: int  # chan, size for whole network
    input_size: int

    def in_channels(self, dir: conv_types.Direction) -> List[int]:
        if dir == 'up':
            return list(reversed(self.out_channels('down')))
        return [self.input_chan] + self.down_channels
    
    def out_channels(self, dir: conv_types.Direction) -> List[int]:
        if dir == 'up':
            return list(reversed(self.in_channels('down')))
        return self.down_channels + [self.down_channels[-1]]

class ApplyTimeEmbedding(nn.Module):
    def __init__(self, *, in_chan: int, time_emblen: int):
        super().__init__()
        self.time_mean_std = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emblen, in_chan * 2),
        )
    
    def forward(self, inputs: Tensor, time_emb: Tensor) -> Tensor:
        time_mean_std = self.time_mean_std(time_emb)
        time_mean_std = einops.rearrange(time_mean_std, "b c -> b c 1 1")
        mean, std = time_mean_std.chunk(2, dim=1)

        out = inputs * (std + 1) + mean
        return out

class DownConv(nn.Sequential):
    def __init__(self, *,
                 in_chan: int, out_chan: int, stride: int,
                 cfg: Config):
        super().__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)
        self.nonlinearity = cfg.nonlinearity.create()

class UpConv(nn.Sequential):
    def __init__(self, *,
                 in_chan: int, out_chan: int, stride: int,
                 cfg: Config):
        super().__init__()

        if stride > 1:
            output_padding = 1
        else:
            output_padding = 0
        self.conv = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, output_padding=output_padding)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)
        self.nonlinearity = cfg.nonlinearity.create()

def add_embed(seq: nn.Sequential, pos: EmbedPos, chan: int, size: int,
              cfg: Config):
    if cfg.sa_pos == pos:
        seq.append(SelfAttention(in_chan=chan, out_chan=chan, nheads=cfg.sa_nheads))
    if cfg.ca_pos == pos:
        seq.append(CrossAttention(in_chan=chan, out_chan=chan, in_size=size, nheads=cfg.ca_nheads, clip_emblen=cfg.clip_emblen))
    if cfg.time_pos == pos:
        seq.append(ApplyTimeEmbedding(in_chan=chan, time_emblen=cfg.time_emblen))

class DownResBlock(nn.Sequential):
    def __init__(self, *, 
                 in_chan: int, in_size: int,
                 out_chan: int,
                 cfg: Config):
        super().__init__()

        for one_index in range(cfg.nstride1):
            if one_index == 0:
                add_embed(self, 'res_first', chan=in_chan, size=in_size, cfg=cfg)
            elif one_index == cfg.nstride1 - 1:
                add_embed(self, 'res_last', chan=in_chan, size=in_size, cfg=cfg)

            self.append(DownConv(in_chan=in_chan, out_chan=in_chan, stride=1, cfg=cfg))

        if out_chan != in_chan:
            self.append(DownConv(in_chan=in_chan, out_chan=out_chan, stride=2, cfg=cfg))

    def forward(self, 
                inputs: Tensor, 
                time_embed: Tensor, 
                clip_embed: Tensor, clip_scale: float) -> Tuple[Tensor, List[Tensor]]:
        out = inputs
        for down_mod in self:
            if isinstance(down_mod, SelfAttention):
                out = down_mod.forward(out)
            elif isinstance(down_mod, CrossAttention):
                out = down_mod.forward(out, clip_embed, clip_scale=clip_scale)
            elif isinstance(down_mod, ApplyTimeEmbedding):
                out = down_mod.forward(out, time_embed)
            else:
                out = down_mod.forward(out)

        return out

class DownHalf(nn.Sequential):
    def __init__(self, cfg: Config):
        super().__init__()

        size = cfg.input_size
        in_chans = cfg.in_channels('down')
        out_chans = cfg.out_channels('down')
        self.in_conv = nn.Conv2d(in_chans[0], out_chans[0], kernel_size=1, stride=1)

        self.downs = nn.Sequential()
        for chan_idx, (in_chan, out_chan) in enumerate(zip(in_chans[1:], out_chans[1:])):
            if chan_idx == 0:
                add_embed(self.downs, 'first', chan=in_chan, size=size, cfg=cfg)

            self.downs.append(
                DownResBlock(in_chan=in_chan, in_size=size, out_chan=out_chan, cfg=cfg)
            )

            if in_chan != out_chan:
                size //= 2
            else:
                add_embed(self.downs, 'last', in_chan, size, cfg=cfg)
    
    def forward(self, inputs: Tensor, time_embed: Tensor, clip_embed: Tensor, clip_scale: float) -> Tuple[Tensor, List[Tensor]]:
        out = self.in_conv(inputs)

        down_outputs: List[Tensor] = list()
        for down_mod in self.downs:
            if isinstance(down_mod, SelfAttention):
                out = down_mod.forward(out)
            elif isinstance(down_mod, CrossAttention):
                out = down_mod.forward(out, clip_embed, clip_scale)
            elif isinstance(down_mod, ApplyTimeEmbedding):
                out = down_mod.forward(out, time_embed)
            else:
                out = down_mod.forward(out, time_embed, clip_embed, clip_scale)
                down_outputs.append(out)

        return out, down_outputs

class UpResBlock(nn.Sequential):
    def __init__(self, *, 
                 in_chan: int, 
                 out_chan: int,
                 cfg: Config):
        super().__init__()

        doubled = False
        if in_chan != out_chan:
            self.append(UpConv(in_chan=in_chan * 2, out_chan=out_chan, stride=2, cfg=cfg))
            in_chan = out_chan
            doubled = True

        for one_index in range(cfg.nstride1):
            if one_index == 0 and not doubled:
                one_in_chan = in_chan * 2
            else:
                one_in_chan = in_chan

            self.append(UpConv(in_chan=one_in_chan, out_chan=out_chan, stride=1, cfg=cfg))
        
        
class UpHalf(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        in_chans = cfg.in_channels('up')
        out_chans = cfg.out_channels('up')
        self.cfg = cfg

        self.ups = nn.Sequential()
        for in_chan, out_chan in zip(in_chans[:-1], out_chans[:-1]):
            self.ups.append(
                UpResBlock(in_chan=in_chan, out_chan=out_chan, cfg=cfg)
            )

        self.out_conv = nn.Conv2d(in_chans[-1], out_chans[-1], kernel_size=1, stride=1)
    
    def forward(self, inputs: Tensor, down_outputs: List[Tensor]) -> Tensor:
        out = inputs
        down_outputs = list(down_outputs)

        for mod in self.ups:
            down_out = down_outputs.pop()
            out = torch.cat([out, down_out], dim=1)
            out = mod.forward(out)
        
        out = self.out_conv(out)
        return out

class DenoiseModelNew(base_model.BaseModel):
    _model_fields = ('in_chan in_size channels nstride1 '
                     'time_pos sa_nheads sa_pos ca_nheads ca_pos clip_emblen clip_scale_default '
                     'nonlinearity_type').split(' ')
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    clip_emblen: int
    clip_scale_default: float

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 channels: List[int], nstride1: int, 
                 time_pos: EmbedPos = 'last',
                 sa_nheads: int, sa_pos: EmbedPos = 'first',
                 ca_nheads: int, ca_pos: EmbedPos = 'last',
                 clip_emblen: int = None,
                 clip_scale_default: float = 1.0,
                 nonlinearity_type: conv_types.NlType = 'silu'):
        super().__init__()

        nonlinearity = conv_types.ConvNonlinearity(nonlinearity_type)
        time_emblen = max(channels)
        cfg = Config(time_emblen=time_emblen, time_pos=time_pos,
                     sa_nheads=sa_nheads, sa_pos=sa_pos,
                     ca_nheads=ca_nheads, ca_pos=ca_pos,
                     clip_emblen=clip_emblen, nonlinearity=nonlinearity,
                     down_channels=channels, nstride1=nstride1,
                     input_chan=in_chan, input_size=in_size)

        self.gen_time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down = DownHalf(cfg=cfg)
        self.up = UpHalf(cfg=cfg)

        self.channels = channels
        self.nstride1 = nstride1
        self.in_chan = in_chan
        self.in_size = in_size
        self.time_pos = time_pos
        self.sa_nheads = sa_nheads
        self.sa_pos = sa_pos
        self.ca_nheads = ca_nheads
        self.ca_pos = ca_pos
        self.clip_emblen = clip_emblen
        self.clip_scale_default = clip_scale_default
        self.nonlinearity_type = nonlinearity_type

        self.in_dim = [in_chan, in_size, in_size]
        lat_size = in_size // (2 ** (len(channels) - 1))
        self.latent_dim = [channels[-1], lat_size, lat_size]
    
    def _get_time_emb(self, inputs: Tensor, time: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb is not None:
            return time_emb

        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        return self.gen_time_emb(time)

    def forward(self, inputs: Tensor, time: Tensor = None, clip_embed: Tensor = None, clip_scale: Tensor = None) -> Tensor:
        if clip_scale is None:
            batch = inputs.shape[0]
            clip_scale = torch.ones((batch, 1, 1, 1), device=inputs.device, dtype=inputs.dtype) * self.clip_scale_default

        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)

        out, down_outputs = self.down.forward(inputs=inputs, time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)
        out = self.up.forward(out, down_outputs)

        return out

