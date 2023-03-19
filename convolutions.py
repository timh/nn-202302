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

            seq = nn.Sequential()
            seq.append(conv)
            if layer.max_pool_kern:
                seq.append(nn.MaxPool2d(kernel_size=layer.max_pool_kern))
            seq.append(cfg.create_inner_norm(out_shape=(out_chan, out_size, out_size)))
            seq.append(cfg.create_inner_nl())
            self.append(nn.Sequential(seq))

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
        encoder_out_size = cfg.get_sizes_down_actual(image_size)[-1]
        sizes = cfg.get_sizes_up_actual(encoder_out_size)

        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i:i + 2]
            in_size, out_size = sizes[i:i + 2]

            conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                      kernel_size=layer.kernel_size, stride=layer.stride, 
                                      padding=layer.up_padding, output_padding=layer.up_output_padding)
            
            seq = nn.Sequential()
            seq.append(conv)
            if layer.max_pool_kern:
                seq.append(nn.Upsample(scale_factor=layer.max_pool_kern))
            if i < len(cfg.layers) - 1:
                seq.append(cfg.create_inner_norm(out_shape=(out_chan, out_size, out_size)))
                seq.append(cfg.create_inner_nl())
            else:
                seq.append(cfg.create_final_norm(out_shape=(out_chan, out_size, out_size)))
                seq.append(cfg.create_final_nl())

            self.append(seq)

        self.out_dim = [channels[-1], sizes[-1], sizes[-1]]
        self.out_size = sizes[-1]
