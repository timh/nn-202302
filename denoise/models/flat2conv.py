from typing import List, Dict, Literal
import functools, operator

from torch import nn, Tensor

import sys
sys.path.append("..")
import base_model
from .denoise_new import Config, EmbedPos, UpResBlock, DownResBlock
from .model_shared import SinPositionEmbedding
import conv_types

NLType = Literal['relu', 'silu', 'gelu']

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
                 sa_nheads: int, sa_pos: EmbedPos = 'first',
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

        flat_len = functools.reduce(operator.mul, first_dim, 1)
        if hidlen is None:
            hidlen = flat_len
        
        # in_len -> hidlen -> flat_len
        self.linear = nn.Sequential()
        lin_in = in_len
        for i in range(nlinear):
            lin_out = flat_len if i == nlinear - 1 else hidlen
            self.linear.append(nn.Linear(lin_in, lin_out))
            self.linear.append(nonlinearity.create())
            lin_in = lin_out
        
        # (flat_len,) -> (first_chan, first_size, first_size)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=first_dim)

        first_chan, first_size = first_dim[:2]
        out_chan, out_size = out_dim[:2]
        cfg = Config(time_pos=None,
                     sa_nheads=sa_nheads, sa_pos=sa_pos,
                     ca_nheads=0, ca_pos=None, clip_emblen=0, 
                     nonlinearity=nonlinearity,
                     down_channels=channels, nstride1=nstride1,
                     input_chan=first_chan, input_size=first_size)

        #    (first_chan, first_size, first_size)
        # ...
        # -> (out_chan, out_size, out_size)
        if first_size < out_size:
            cfg.down_channels = list(reversed(cfg.down_channels))

            up_chan, up_size = first_chan, first_size
            self.convs = nn.Sequential()
            for chan in channels:
                self.convs.append(UpResBlock(in_chan=up_chan, in_size=up_size, out_chan=chan, cfg=cfg, do_residual=False))
                up_chan = chan
                up_size = up_size * 2
        else:
            down_chan, down_size = first_chan, first_size
            self.convs = nn.Sequential()
            for chan in channels:
                self.convs.append(DownResBlock(in_chan=down_chan, in_size=down_size, out_chan=chan, cfg=cfg))
                down_chan = chan
                down_size = down_size // 2
    
    def forward(self, inputs: Tensor, *args) -> Tensor:
        # in_len -> ... -> flat_len
        out = self.linear(inputs)

        #    (flat_len, )
        # -> (first_chan, first_size, first_size)
        out = self.unflatten(out)

        #    (first_chan, first_size, first_size)
        # -> (out_chan, out_size, out_size)
        out = self.convs.forward(out)
        if isinstance(out, tuple):
            out = out[0]

        return out

