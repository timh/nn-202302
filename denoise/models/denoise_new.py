from typing import List, Literal, Tuple

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
class ApplyTimeEmbedding(nn.Module):
    def __init__(self, *,
                 in_chan: int,
                 time_emblen: int):
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
        
class Conv(nn.Sequential):
    def __init__(self, *,
                 dir: conv_types.Direction,
                 in_chan: int, out_chan: int, stride: int,
                 nonlinearity: conv_types.ConvNonlinearity):
        super().__init__()

        self.dir = dir

        if dir == 'down':
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)
        self.nonlinearity = nonlinearity.create()
    
    def __extra_repr__(self) -> str:
        return f"dir: {self.dir}"

class DownHalf(nn.Sequential):
    def __init__(self, *, 
                 in_chan: int, in_size: int,
                 channels: List[int], nstride1: int,
                 time_emblen: int,
                 time_pos: EmbedPos = 'last',
                 sa_nheads: int, sa_pos: EmbedPos = 'begin',
                 ca_nheads: int, ca_pos: EmbedPos = 'end',
                 clip_emblen: int = None,
                 nonlinearity: conv_types.ConvNonlinearity):
        super().__init__()

        def add_embed(pos: EmbedPos, chan: int, size: int):
            if sa_pos == pos:
                self.downs.append(SelfAttention(in_chan=chan, out_chan=chan, nheads=sa_nheads))
            if ca_pos == pos:
                self.downs.append(CrossAttention(in_chan=chan, out_chan=chan, in_size=size, nheads=ca_nheads, clip_emblen=clip_emblen))
            if time_pos == pos:
                self.downs.append(ApplyTimeEmbedding(in_chan=chan, time_emblen=time_emblen))

        self.in_conv = nn.Conv2d(in_chan, channels[0], kernel_size=1, stride=1)

        size = in_size
        in_chan = channels[0]
        self.downs = nn.Sequential()
        for chan_idx, out_chan in enumerate(channels):
            if chan_idx == 0:
                add_embed('first', in_chan, size)

            for one_index in range(nstride1):
                self.downs.append(Conv(dir='down', in_chan=in_chan, out_chan=in_chan, stride=1, nonlinearity=nonlinearity))
                if one_index == 0:
                    add_embed('res_first', in_chan, size)
                if one_index == nstride1 - 1:
                    add_embed('res_last', in_chan, size)
            
            self.downs.append(Conv(dir='down', in_chan=in_chan, out_chan=out_chan, stride=2, nonlinearity=nonlinearity))
            size //= 2

            add_embed('last', out_chan, size)
            in_chan = out_chan
    
    def forward(self, inputs: Tensor, time_embed: Tensor, clip_embed: Tensor, clip_scale: float) -> Tuple[Tensor, List[Tensor]]:
        down_outputs: List[Tensor] = list()
        out = self.in_conv(inputs)
        for down_mod in self.downs:
            if isinstance(down_mod, SelfAttention):
                out = down_mod.forward(out)
            elif isinstance(down_mod, CrossAttention):
                out = down_mod.forward(out, clip_embed, clip_scale=clip_scale)
            elif isinstance(down_mod, ApplyTimeEmbedding):
                out = down_mod.forward(out, time_embed)
            else:
                out = down_mod.forward(out)
                down_outputs.append(out)
        return out, down_outputs

class UpResBlock(nn.Sequential):
    def __init__(self, *, 
                 in_chan: int, out_chan: int, nstride1: int,
                 nonlinearity: conv_types.ConvNonlinearity):
        super().__init__()
        for one_index in range(nstride1):
            if one_index == 0:
                one_in_chan = in_chan * 2
            else:
                one_in_chan = in_chan
            self.append(Conv(dir='up', in_chan=one_in_chan, out_chan=in_chan, stride=1, nonlinearity=nonlinearity))
        
        self.append(Conv(dir='up', in_chan=in_chan, out_chan=out_chan, stride=2, nonlinearity=nonlinearity))

class UpHalf(nn.Module):
    def __init__(self, *, 
                 result_out_chan: int, 
                 channels: List[int], nstride1: int,
                 nonlinearity: conv_types.ConvNonlinearity):
        """call with reversed(channels)!!!"""
        super().__init__()
        self.ups = nn.Sequential()
        in_chan = channels[0]
        for out_chan in channels:
            self.ups.append(UpResBlock(in_chan=in_chan, out_chan=out_chan, nstride1=nstride1, nonlinearity=nonlinearity))
            in_chan = out_chan

        self.out_conv = nn.Conv2d(out_chan, result_out_chan, kernel_size=1, stride=1)
    
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
    _model_fields = ('channels nstride1 in_size clip_emblen clip_scale_default '
                     'time_pos sa_nheads sa_pos ca_nheads ca_pos').split(' ')
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

        self.gen_time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down = DownHalf(in_chan=in_chan, in_size=in_size, channels=channels, nstride1=nstride1,
                             time_emblen=time_emblen, time_pos=time_pos,
                             sa_nheads=sa_nheads, sa_pos=sa_pos,
                             ca_nheads=ca_nheads, ca_pos=ca_pos,
                             clip_emblen=clip_emblen, nonlinearity=nonlinearity)

        self.up = UpHalf(result_out_chan=in_chan, channels=list(reversed(channels)), nstride1=nstride1,
                         nonlinearity=nonlinearity)

        self.channels = channels
        self.nstride1 = nstride1
        self.in_size = in_size
        self.time_pos = time_pos
        self.sa_nheads = sa_nheads
        self.sa_pos = sa_pos
        self.ca_nheads = ca_nheads
        self.ca_pos = ca_pos
        self.clip_emblen = clip_emblen
        self.clip_scale_default = clip_scale_default

        self.in_dim = [channels[0], in_size, in_size]
        lat_size = in_size // (2 ** len(channels))
        self.latent_dim = [channels[-1], lat_size, lat_size]
    
    def _get_time_emb(self, inputs: Tensor, time: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb is not None:
            return time_emb

        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        return self.gen_time_emb(time)

    def forward(self, inputs: Tensor, time: Tensor = None, clip_embed: Tensor = None, clip_scale: float = None) -> Tensor:
        clip_scale = clip_scale or self.clip_scale_default

        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)

        out, down_outputs = self.down.forward(inputs=inputs, time_embed=time_emb, clip_embed=clip_embed, clip_scale=clip_scale)
        out = self.up.forward(out, down_outputs)

        _, out_chan, out_size, _ = out.shape
        in_chan, in_size, _ = self.in_dim
        if out_size < in_size:
            diff_2 = (in_size - out_size) // 2
            diff_mod = (in_size - out_size) % 2
            out = F.pad(input=out, pad=(diff_2, diff_2 + diff_mod, diff_2, diff_2 + diff_mod), value=1.0)
        
        return out

