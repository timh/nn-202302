# %%
import sys
import math
from typing import List, Dict

sys.path.append("..")
import base_model
import model
from model import ConvDesc
import torch
from torch import nn, Tensor

class VAEModel(base_model.BaseModel):
    _metadata_fields = {'image_size', 'nchannels', 'channels', 'nonlinearity', 'do_flat_conv2d', 'kernel_size'}
    def __init__(self, image_size: int, 
                 channels = [32, 16, 4],
                 nonlinearity = "sigmoid",
                 do_flat_conv2d: bool = False,
                 kernel_size: int = 3, nchannels = 3) -> None:
        super().__init__()

        self.image_size = image_size
        self.nchannels = nchannels
        self.channels = channels
        self.nonlinearity = nonlinearity
        self.do_flat_conv2d = do_flat_conv2d
        self.kernel_size = kernel_size
        self.nchannels = nchannels

        def nonlinear_fn():
            if nonlinearity == "sigmoid":
                return nn.Sigmoid()
            elif nonlinearity == "relu":
                return nn.ReLU(True)
            elif nonlinearity == "leaky-relu":
                return nn.LeakyReLU(inplace=True)
            elif nonlinearity == "tanh":
                return nn.Tanh()
            elif nonlinearity == "gelu":
                return nn.GELU()
            else:
                raise Exception(f"unknown {nonlinearity=}")

        all_channels = [nchannels] + channels

        padding = 2 ** (len(channels) - 1)
        in_size = image_size + padding * 2

        self.encoder = nn.Sequential()
        self.encoder.append(nn.ConstantPad2d(padding, value=0))
        for i in range(len(all_channels) - 1):
            in_chan, out_chan = all_channels[i:i + 2]
            self.encoder.append(nn.Conv2d(in_chan, out_chan,
                                          kernel_size=kernel_size, stride=2,
                                          padding=0))

            out_size = (in_size - kernel_size) // 2 + 1
            self.encoder.append(nn.LayerNorm((out_chan, out_size, out_size)))
            self.encoder.append(nonlinear_fn())

            if do_flat_conv2d:
                self.encoder.append(nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1))
                self.encoder.append(nn.LayerNorm((out_chan, out_size, out_size)))
                self.encoder.append(nonlinear_fn())

            in_size = out_size
        
        self.embdim = (all_channels[-1], in_size, in_size)

        self.decoder = nn.Sequential()
        for i in reversed(range(len(all_channels) - 1)):
            in_chan, out_chan = reversed(all_channels[i:i + 2])
            padding = 1 if i > len(channels) - 2 else 0
            output_padding = 1 if i == 0 else 0
            
            out_size = (in_size - 1) * 2 - 2 * padding + kernel_size + output_padding
            self.decoder.append(nn.ConvTranspose2d(in_chan, out_chan, kernel_size=kernel_size,
                                                   stride=2, padding=padding, output_padding=output_padding))
            self.decoder.append(nn.LayerNorm((out_chan, out_size, out_size)))
            self.decoder.append(nonlinear_fn())

            if do_flat_conv2d:
                self.decoder.append(nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1))
                self.decoder.append(nn.LayerNorm((out_chan, out_size, out_size)))
                self.decoder.append(nonlinear_fn())

            in_size = out_size
            
    """
       (batch, nchan, size, size)
    -> (batch, nchan, size, size)
    """
    def forward(self, inputs: Tensor) -> Tensor:
        #    (batch, nchan, size, size)
        # -> (batch, encoder_out_size, descs[-1].channels, descs[-1].channels)
        out = self.encoder(inputs)

        #    (batch, encoder_out_size, descs[-1].channels, descs[-1].channels)
        # -> (batch, nchan, size, ize)
        out = self.decoder(out)

        return out
    
if __name__ == "__main__":
    device = "cuda"
    image_size = 512
    nchannels = 3
    # channels = [64,64,64]
    # channels = [64,64,64,32]
    channels = [64,64,64,64,64]
    
    inputs = torch.rand((1, nchannels, image_size, image_size)).to(device)
    print(f"{inputs.shape=}")

    net = VAEModel(image_size, channels=channels).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"{net.embdim=}")
    print(f"{nparams/1e6:.3f}")

    out = net.encoder(inputs)
    print(f"encoder out: {out.shape=}")

    out = net.decoder(out)
    print(f"decoder out: {out.shape=}")
