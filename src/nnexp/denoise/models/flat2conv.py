from typing import List, Literal

import torch
from torch import nn, Tensor

from nnexp import base_model
from nnexp.images import conv_types
from .denoise_new import Config

NLType = Literal['relu', 'silu', 'gelu']

class Clip2Conv(nn.Sequential):
    def __init__(self, *,
                 emb_len: int,
                 chan: int, size: int, 
                 nlinear: int, hidlen: int = None,
                 cfg: Config):
        super().__init__()

        flat_len = chan * size * size
        if hidlen is None:
            hidlen = flat_len

        self.linear = nn.Sequential()
        lin_in = emb_len
        for i in range(nlinear):
            lin_out = flat_len if i == nlinear - 1 else hidlen
            self.linear.append(nn.Linear(lin_in, lin_out))
            self.linear.append(cfg.nonlinearity.create())
            self.linear.append(nn.LayerNorm(normalized_shape=(lin_out,)))
            lin_in = lin_out
        
        # (flat_len,) -> (first_chan, first_size, first_size)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, size, size))

class ConvLayer(nn.Module):
    def __init__(self, *,
                 emb_len: int,
                 in_chan: int, in_size: int, out_chan: int,
                 nlinear: int, hidlen: int = None,
                 cfg: Config, do_combine: bool):
        super().__init__()
        self.clip2conv = Clip2Conv(emb_len=emb_len, chan=in_chan, size=in_size, nlinear=nlinear, hidlen=hidlen, cfg=cfg)

        self.do_combine = do_combine
        if do_combine:
            self.combine = nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1)
        self.upres = UpResBlock(in_chan=in_chan, in_size=in_size, out_chan=out_chan, cfg=cfg, do_residual=False)
    
    def forward(self, prev_layer: Tensor, clip_embed: Tensor, *args) -> Tensor:
        out = self.clip2conv.forward(clip_embed)
        if self.do_combine:
            combined_in = torch.cat([prev_layer, out], dim=1)
            out = self.combine(combined_in)
        out = self.upres(out)
        return out


FIELDS = ("in_len first_dim out_dim channels nstride1 nlinear hidlen nonlinearity_type "
          "sa_nheads sa_pos").split()
class EmbedToLatent(base_model.BaseModel):
    _metadata_fields = FIELDS
    _model_fields = _metadata_fields

    def __init__(self, 
                 *,
                 in_len: int, first_dim: List[int], out_dim: List[int], 
                 channels: List[int], nstride1: int,
                 nlinear: int, hidlen: int = None,
                 sa_nheads: int,
                 nonlinearity_type: conv_types.NlType = 'silu'):
        super().__init__()
        self.in_len = in_len
        self.first_dim = first_dim
        self.out_dim = out_dim
        self.channels = channels
        self.nstride1 = nstride1
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.sa_nheads = sa_nheads
        self.sa_pos = sa_pos
        self.nonlinearity_type = nonlinearity_type

        nonlinearity = conv_types.ConvNonlinearity(nonlinearity_type)

        # in_len -> hidlen -> flat_len

        down_channels = list(reversed(channels))

        first_chan, first_size = first_dim[:2]
        cfg = Config(time_pos=list(),
                     sa_nheads=sa_nheads, sa_pos=sa_pos,
                     ca_nheads=0, ca_pos=list(), ca_pos_conv=list(),
                     clip_emblen=0, 
                     nonlinearity=nonlinearity,
                     down_channels=down_channels, nstride1=nstride1,
                     input_chan=first_chan, input_size=first_size)

        #    (first_chan, first_size, first_size)
        # ...
        # -> (out_chan, out_size, out_size)

        in_chan, in_size = first_chan, first_size
        self.layers = nn.Sequential()
        for i, out_chan in enumerate(channels):
            do_combine = i > 0

            layer = ConvLayer(emb_len=in_len,
                              in_chan=in_chan, in_size=in_size, out_chan=out_chan,
                              nlinear=nlinear, hidlen=hidlen, cfg=cfg, do_combine=do_combine)
            self.layers.append(layer)

            if in_chan != out_chan:
                in_size = in_size * 2
            in_chan = out_chan

            if i == len(channels) - 1 and out_chan != out_dim[0]:
                # final layer to convert down to out_dim
                layer = ConvLayer(emb_len=in_len, in_chan=out_chan, in_size=in_size, out_chan=out_dim[0],
                                  nlinear=nlinear, hidlen=hidlen, cfg=cfg, do_combine=True)


    def forward(self, clip_embed: Tensor) -> Tensor:
        out = None

        for mod in self.layers:
            out = mod.forward(out, clip_embed)

        return out

