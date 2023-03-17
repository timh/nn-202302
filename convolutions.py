from typing import List
from torch import nn, Tensor

# from . import conv_types
import conv_types

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, layers[-1].out_chan, out_size, out_size)
"""
class DownStack(nn.Sequential):
    """side dimension of image after convolutions are processed"""
    out_dim: List[int]

    def __init__(self, *, image_size: int, nchannels: int, cfg: conv_types.ConvConfig):
        super().__init__()

        channels = cfg.get_channels_down(nchannels)
        sizes = cfg.get_sizes_down_actual(image_size)

        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i:i + 2]
            in_size, out_size = sizes[i:i + 2]

            conv = nn.Conv2d(in_chan, out_chan, 
                             kernel_size=layer.kernel_size, stride=layer.stride, 
                             padding=layer.down_padding)
            print(f"down: add {in_chan=} {out_chan=} {in_size=} {out_size=}")

            layer = nn.Sequential()
            layer.append(conv)
            layer.append(cfg.create_norm(out_chan=out_chan, out_size=out_size))
            layer.append(cfg.create_inner_nl())
            self.append(nn.Sequential(layer))

        self.out_dim = [channels[-1], sizes[-1], sizes[-1]]
        self.out_size = sizes[-1]

"""
inputs: (batch, layers[-1].out_chan, out_size, out_size)
return: (batch, nchannels, image_size, image_size)
"""
class UpStack(nn.Sequential):
    def __init__(self, *, image_size: int, nchannels: int, 
                 cfg: conv_types.ConvConfig):
        super().__init__()

        channels = cfg.get_channels_up(nchannels)
        sizes = cfg.get_sizes_up_actual(image_size)

        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i:i + 2]
            in_size, out_size = sizes[i:i + 2]

            print(f"up: {in_chan=} {out_chan=} {in_size=} {out_size=}")

            conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                      kernel_size=layer.kernel_size, stride=layer.stride, 
                                      padding=layer.up_padding, output_padding=layer.up_output_padding)
            
            layer = nn.Sequential()
            layer.append(conv)
            layer.append(cfg.create_norm(out_chan=out_chan, out_size=out_size))
            if i < len(cfg.layers) - 1:
                layer.append(cfg.create_inner_nl())
            else:
                layer.append(cfg.create_final_nl())

            self.append(layer)

        self.out_dim = [channels[-1], sizes[-1], sizes[-1]]
        self.out_size = sizes[-1]
