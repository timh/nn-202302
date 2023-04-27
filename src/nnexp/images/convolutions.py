from typing import List
from torch import nn, Tensor

from nnexp.images import conv_types

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, layers[-1].out_chan, out_size, out_size)
"""
class DownStack(nn.Sequential):
    """side dimension of image after convolutions are processed"""
    out_dim: List[int]
    layer_ins: List[Tensor]

    def __init__(self, *, in_size: int, in_chan: int, cfg: conv_types.ConvConfig):
        super().__init__()

        all_layers = cfg.create_down_all()
        for layer in all_layers:
            seq = nn.Sequential(*layer)
            self.append(seq)

        out_chan = cfg.layers[-1].out_chan('down')
        out_size = cfg.layers[-1].out_size('down')

        self.out_dim = [out_chan, out_size, out_size]
        self.out_size = out_size
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.layer_ins = []
        out = inputs
        for layer in self:
            self.layer_ins.append(out)
            out = layer(out)
        return out

"""
inputs: (batch, layers[-1].out_chan, out_size, out_size)
return: (batch, nchannels, image_size, image_size)
"""
class UpStack(nn.Sequential):
    def __init__(self, *, in_size: int, in_chan: int, 
                 cfg: conv_types.ConvConfig):
        super().__init__()

        all_layers = cfg.create_up_all()
        for layer in all_layers:
            seq = nn.Sequential(*layer)
            self.append(seq)

        out_chan = cfg.layers[-1].out_chan('up')
        out_size = cfg.layers[-1].out_size('up')

        self.out_dim = [out_chan, out_size, out_size]
        self.out_size = out_size
