from typing import List, Literal, Tuple
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import einops

from nnexp import base_model
from nnexp.images import conv_types
from .model_shared import SinPositionEmbedding, SelfAttention, CrossAttention, CrossAttentionConv


# TODO: fix the unprocessed borders around output - probably need to run @ higher res?

EmbedPos = Literal['first', 'last',          # first and last of the entire down stack
                   'res_first', 'res_last',  # first and last of a given resolution/channel
                   'up_first', 'up_last',
                   'up_res_first', 'up_res_last',
                   ]

@dataclass(kw_only=True)
class Config:
    time_pos: List[EmbedPos]

    sa_nheads: int
    sa_pos: List[EmbedPos]
    ca_nheads: int
    ca_pos: List[EmbedPos]
    ca_pos_conv: List[EmbedPos]
    ca_pos_lin: List[EmbedPos]
    nonlinearity: conv_types.ConvNonlinearity

    down_channels: List[int]
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

class ResBlock(nn.Sequential):
    def forward(self, 
                inputs: Tensor, 
                time_embed: Tensor = None, 
                clip_embed: Tensor = None, clip_scale: Tensor = None) -> Tuple[Tensor, List[Tensor]]:
        out = inputs
        for mod in self:
            if isinstance(mod, SelfAttention):
                out = mod.forward(out)
            elif type(mod) in [CrossAttention, CrossAttentionConv]:
                out = mod.forward(out, clip_embed, clip_scale=clip_scale)
            elif isinstance(mod, ApplyTimeEmbedding):
                out = mod.forward(out, time_embed)
            # elif isinstance(mod, ApplyClipEmbedding):
            #     out = mod.forward(out, clip_embed, clip_scale)
            else:
                out = mod.forward(out)

        return out

def add_embed(seq: nn.Sequential, pos: EmbedPos, chan: int, size: int,
              cfg: Config):
    if cfg.sa_pos and pos in cfg.sa_pos:
        seq.append(SelfAttention(chan=chan, nheads=cfg.sa_nheads))
    # if cfg.ca_pos and pos in cfg.ca_pos:
    #     seq.append(CrossAttention(clip_emblen=cfg.clip_emblen, chan=chan, size=size, nheads=cfg.ca_nheads))
    # if cfg.ca_pos_conv and pos in cfg.ca_pos_conv:
    #     seq.append(CrossAttentionConv(clip_emblen=cfg.clip_emblen, chan=chan, size=size, nheads=cfg.ca_nheads))
    # if cfg.ca_pos_lin and pos in cfg.ca_pos_lin:
    #     seq.append(ApplyClipEmbedding(chan=chan, clip_emblen=cfg.clip_emblen))
    if cfg.time_pos and pos in cfg.time_pos:
        seq.append(ApplyTimeEmbedding(time_emblen=cfg.time_emblen, chan=chan))

class DownResBlock(ResBlock):
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


class DownHalf(nn.Sequential):
    def __init__(self, cfg: Config):
        super().__init__()

        in_size = cfg.input_size
        in_chans = cfg.in_channels('down')
        out_chans = cfg.out_channels('down')

        self.in_conv = nn.Conv2d(in_chans[0], out_chans[0], kernel_size=1, stride=1)
        self.downs = nn.Sequential()

        add_embed(self.downs, 'first', chan=in_chans[0], size=in_size, cfg=cfg)

        for chan_idx, (in_chan, out_chan) in enumerate(zip(in_chans[1:], out_chans[1:])):
            self.downs.append(
                DownResBlock(in_chan=in_chan, in_size=in_size, out_chan=out_chan, cfg=cfg)
            )

            if in_chan != out_chan:
                in_size //= 2

        add_embed(self.downs, 'last', in_chans[-1], in_size, cfg=cfg)
    
    def forward(self, inputs: Tensor, time_embed: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tuple[Tensor, List[Tensor]]:
        out = self.in_conv(inputs)

        down_outputs: List[Tensor] = list()
        for down_mod in self.downs:
            if isinstance(down_mod, SelfAttention):
                out = down_mod.forward(out)
            elif type(down_mod) in [CrossAttention, CrossAttentionConv]:
                out = down_mod.forward(out, clip_embed, clip_scale)
            elif isinstance(down_mod, ApplyTimeEmbedding):
                out = down_mod.forward(out, time_embed)
            # elif isinstance(down_mod, ApplyClipEmbedding):
            #     out = down_mod.forward(out, clip_embed, clip_scale)
            else:
                out = down_mod.forward(out, time_embed, clip_embed, clip_scale)
                down_outputs.append(out)

        return out, down_outputs

class UpResBlock(ResBlock):
    def __init__(self, *, 
                 in_chan: int, in_size: int,
                 out_chan: int,
                 cfg: Config,
                 do_residual: bool = True):
        super().__init__()

        res_doubled = False
        if in_chan != out_chan:
            if do_residual:
                self.append(UpConv(in_chan=in_chan * 2, out_chan=out_chan, stride=2, cfg=cfg))
            else:
                self.append(UpConv(in_chan=in_chan, out_chan=out_chan, stride=2, cfg=cfg))
            in_chan = out_chan
            res_doubled = True

        for one_index in range(cfg.nstride1):
            if do_residual and one_index == 0 and not res_doubled:
                one_in_chan = in_chan * 2
            else:
                one_in_chan = in_chan

            self.append(UpConv(in_chan=one_in_chan, out_chan=out_chan, stride=1, cfg=cfg))

            if one_index == 0:
                add_embed(self, 'up_res_first', chan=out_chan, size=in_size, cfg=cfg)
            elif one_index == cfg.nstride1 - 1:
                add_embed(self, 'up_res_last', chan=out_chan, size=in_size, cfg=cfg)

class UpHalf(nn.Module):
    def __init__(self, cfg: Config, in_size: int, do_residual: bool = True):
        super().__init__()

        self.do_residual = do_residual

        in_chans = cfg.in_channels('up')
        out_chans = cfg.out_channels('up')
        self.ups = nn.Sequential()

        add_embed(self.ups, 'up_first', chan=in_chans[0], size=in_size, cfg=cfg)

        for chan_idx, (in_chan, out_chan) in enumerate(zip(in_chans[:-1], out_chans[:-1])):
            self.ups.append(
                UpResBlock(in_chan=in_chan, in_size=in_size, out_chan=out_chan, cfg=cfg,
                           do_residual=do_residual)
            )

            if in_chan != out_chan:
                in_size *= 2

        add_embed(self.ups, 'up_last', chan=in_chans[-1], size=in_size, cfg=cfg)

        self.out_conv = nn.Conv2d(in_chans[-1], out_chans[-1], kernel_size=1, stride=1)
    
    def forward(self, inputs: Tensor, down_outputs: List[Tensor],
                time_embed: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tensor:
        out = inputs
        down_outputs = list(down_outputs)

        for mod in self.ups:
            if isinstance(mod, SelfAttention):
                out = mod.forward(out)
            elif type(mod) in [CrossAttention, CrossAttentionConv]:
                out = mod.forward(out, clip_embed, clip_scale)
            elif isinstance(mod, ApplyTimeEmbedding):
                out = mod.forward(out, time_embed)
            # elif isinstance(mod, ApplyClipEmbedding):
            #     out = mod.forward(out, clip_embed, clip_scale)
            elif self.do_residual:
                down_out = down_outputs.pop()
                out = torch.cat([out, down_out], dim=1)
                out = mod.forward(out, time_embed, clip_embed, clip_scale)
            else:
                out = mod.forward(out, time_embed, clip_embed, clip_scale)
        
        out = self.out_conv(out)
        return out

class DenoiseModelNew(base_model.BaseModel):
    _model_fields = ("in_chan in_size channels nstride1 "
                     "time_pos sa_nheads sa_pos "
                     "ca_nheads ca_pos ca_pos_conv ca_pos_lin "
                     "clip_scale_default nonlinearity_type").split()
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    clip_scale_default: float

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 channels: List[int], nstride1: int, 
                 time_pos: List[EmbedPos] = ['last'],
                 sa_nheads: int, sa_pos: List[EmbedPos] = ['first'],
                 ca_nheads: int, 
                 ca_pos: List[EmbedPos] = ['last'], 
                 ca_pos_conv: List[EmbedPos] = [],
                 ca_pos_lin: List[EmbedPos] = [],
                 clip_scale_default: float = 1.0,
                 nonlinearity_type: conv_types.NlType = 'silu'):
        super().__init__()

        lat_size = in_size // (2 ** (len(channels) - 1))
        self.in_dim = [in_chan, in_size, in_size]
        self.latent_dim = [channels[-1], lat_size, lat_size]

        nonlinearity = conv_types.ConvNonlinearity(nonlinearity_type)
        cfg = Config(time_pos=time_pos,
                     sa_nheads=sa_nheads, sa_pos=sa_pos,
                     ca_nheads=ca_nheads, 
                     ca_pos=ca_pos, ca_pos_conv=ca_pos_conv, ca_pos_lin=ca_pos_lin,
                     nonlinearity=nonlinearity,
                     down_channels=channels, nstride1=nstride1,
                     input_chan=in_chan, input_size=in_size)

        time_emblen = cfg.time_emblen
        self.gen_time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down = DownHalf(cfg=cfg)
        self.up = UpHalf(cfg=cfg, in_size=lat_size)

        self.channels = channels
        self.nstride1 = nstride1
        self.in_chan = in_chan
        self.in_size = in_size
        self.time_pos = time_pos
        self.sa_nheads = sa_nheads
        self.sa_pos = sa_pos
        self.ca_nheads = ca_nheads
        self.ca_pos = ca_pos
        self.ca_pos_conv = ca_pos_conv
        self.ca_pos_lin = ca_pos_lin
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

        out, down_outputs = self.down.forward(inputs=inputs, time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)
        out = self.up.forward(out, down_outputs,
                              time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)

        return out

