# %%
import sys
from typing import List, Union, Tuple, Callable
from dataclasses import dataclass

import torch
from torch import nn, Tensor

sys.path.append("..")
from experiment import Experiment
import trainer

@dataclass
class ConvDesc:
    channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 1
    output_padding: int = 0

    def get_out_size_encode(self, size: int) -> int:
        out = (size + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out

    def get_out_size_decode(self, size: int) -> int:
        out = (size - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        return out

class Encoder(nn.Module):
    conv_seq: nn.Sequential

    """side dimension of image after convolutions are processed"""
    out_size: int

    def __init__(self, image_size: int, emblen: int, hidlen: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        self.image_size = image_size
        self.descs = descs
        self.nchannels = nchannels

        channels = [nchannels] + [d.channels for d in descs]

        out_size = image_size
        for d in descs:
            out_size = d.get_out_size_encode(out_size)
        self.out_size = out_size

        self.conv_seq = nn.Sequential()
        for i, desc in enumerate(descs):
            inchan, outchan = channels[i:i + 2]
            conv = nn.Conv2d(in_channels=inchan, out_channels=outchan, 
                             kernel_size=desc.kernel_size, stride=desc.stride, padding=desc.padding,
                             device=device)
            self.conv_seq.append(conv)
            if i > 0 and i < len(descs) - 1:
                self.conv_seq.append(nn.BatchNorm2d(outchan, device=device))
            self.conv_seq.append(nn.ReLU(True))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Sequential(
            nn.Linear(descs[-1].channels * out_size * out_size, hidlen),
            nn.ReLU(True),
            nn.Linear(hidlen, emblen),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv_seq(inputs)
        out = self.flatten(out)
        out = self.linear(out)

        return out

class Decoder(nn.Module):
    conv_seq: nn.Sequential

    """
    """
    def __init__(self, image_size: int, encoder_out_size: int, emblen: int, hidlen: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        self.image_size = image_size
        self.descs = descs
        self.emblen = emblen

        firstchan = descs[0].channels
        self.linear = nn.Sequential(
            nn.Linear(emblen, hidlen),
            nn.ReLU(True),
            nn.Linear(hidlen, firstchan * encoder_out_size * encoder_out_size),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(firstchan, encoder_out_size, encoder_out_size))

        channels = [d.channels for d in descs] + [nchannels]
        self.conv_seq = nn.Sequential()
        out_size = encoder_out_size
        for i, desc in enumerate(descs):
            inchan, outchan = channels[i:i+2]
            out_size = desc.get_out_size_decode(out_size)

            conv = nn.ConvTranspose2d(in_channels=inchan, out_channels=outchan, kernel_size=desc.kernel_size, 
                                      stride=desc.stride, padding=desc.padding, output_padding=desc.output_padding,
                                      device=device)
            self.conv_seq.append(conv)
            self.conv_seq.append(nn.BatchNorm2d(outchan, device=device))
            self.conv_seq.append(nn.ReLU(True))

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.linear(inputs)
        out = self.unflatten(out)
        out = self.conv_seq(out)
        # out = torch.sigmoid(out)

        return out

class ConvEncDec(nn.Module):
    encoder: Encoder
    linear_layers: nn.Sequential
    decoder: Decoder

    def __init__(self, image_size: int, emblen: int, nlinear: int, hidlen: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        out_image_size = image_size
        for desc in descs:
            out_image_size = desc.get_out_size_encode(out_image_size)

        self.encoder = Encoder(image_size=image_size, 
                               emblen=emblen, hidlen=hidlen, descs=descs, 
                               nchannels=nchannels, device=device)
        self.linear_layers = nn.Sequential()
        for i in range(nlinear):
            in_features = emblen if i == 0 else hidlen
            out_features = hidlen if i < nlinear - 1 else emblen
            self.linear_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            self.linear_layers.append(nn.ReLU(True))

        out_size = self.encoder.out_size
        descs_rev = list(reversed(descs))
        self.decoder = Decoder(image_size=image_size, encoder_out_size=out_size,
                               emblen=emblen, hidlen=hidlen, descs=descs_rev, 
                               nchannels=nchannels, device=device)
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.encoder(inputs)
        out = self.linear_layers(out)
        out = self.decoder(out)
        return out

"""
generate an image from pure noise.
"""
def generate(exp: Experiment, num_steps: int, size: int, input: Tensor = None, device = "cpu") -> Tensor:
    if input is None:
        input = torch.rand((1, 3, size, size), device=device)
    orig_input = input
    exp.net.eval()
    if num_steps <= 1:
        return input
    with torch.no_grad():
        for step in range(num_steps - 1):
            out_noise = exp.net.forward(input)
            keep_noise_amount = (step + 1) / num_steps
            out = input - keep_noise_amount * out_noise
            input = out
    return out

def gen_noise(size) -> Tensor:
    return torch.normal(mean=0, std=0.5, size=size)

"""generate a list of ConvDescs from a string like the following:

input: "k5-p2-c16,c32,c64,c32,c16"

where
    k = kernel_size
    p = padding
    c = channels

would return a list of 5 ConvDesc's:
    [ConvDesc(kernel_size=5, padding=2, channels=16),
     ConvDesc(kernel_size=5, padding=2, channels=32), 
     ... for same kernel_size/padding and channels=64, 32, 16.

This returns a ConvDesc for each comma-separated substring.

Each ConvDesc *must* have a (c)hannel set, but the (k)ernel_size and (p)adding
will carry on from block to block.
"""
def gen_descs(s: str) -> List[ConvDesc]:
    kernel_size = 0
    padding = 0
    stride = 1

    descs: List[ConvDesc] = list()
    for onedesc_str in s.split(","):
        channels = 0
        for part in onedesc_str.split("-"):
            if part.startswith("c"):
                channels = int(part[1:])
            elif part.startswith("k"):
                kernel_size = int(part[1:])
            elif part.startswith("p"):
                padding = int(part[1:])
            elif part.startswith("s"):
                stride = int(part[1:])
        
        if channels == 0:
            raise ValueError("channels not defined. it must be repeated each comma-separated description.")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")

        # onedesc = ConvDesc(channels=channels, kernel_size=kernel_size, padding=padding, stride=stride)
        onedesc = ConvDesc(channels=channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=padding)
        descs.append(onedesc)
    return descs

if __name__ == "__main__":
    import importlib
    # descs = gen_descs("k3-s2-p1-c8,c16,c32")
    descs = gen_descs("p1-k3-s2-c8,c16,c32")
    image_size = 128
    emblen = 64
    nlinear = 2
    hidlen = 64

    net = ConvEncDec(image_size=image_size, emblen=emblen, nlinear=nlinear, hidlen=hidlen, descs=descs, nchannels=3).to("cuda")
    inputs = torch.rand((1, 3, image_size, image_size), device="cuda")
    print(net.encoder)
    print(net.decoder)
    print(f"{net.encoder.out_size=}")
    print(f"{inputs.shape=}")
    out = net(inputs)
    print(f"{out.shape=}")

