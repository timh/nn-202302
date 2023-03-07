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
    do_bnorm = True
    do_relu = True

    def get_out_size(self, image_size: int) -> int:
        out = (image_size + self.padding * 2 + self.kernel_size) // (self.stride + 1)
        return out

def gen_layers(image_size: int, descs: List[ConvDesc], layer_fn: Callable, in_channels: int, device = "cpu"):
    res: List[nn.Module] = list()
    for i, desc in enumerate(descs):
        if i == 0:
            inchan = in_channels
        else:
            inchan = descs[i - 1].channels

        module = layer_fn(in_channels=inchan, out_channels=desc.channels, kernel_size=desc.kernel_size, stride=desc.stride, padding=desc.padding, device=device)
        res.append(module)

        if desc.do_bnorm:
            res.append(nn.BatchNorm2d(desc.channels, device=device))
        if desc.do_relu:
            res.append(nn.ReLU())

    return res

class Encoder(nn.Module):
    conv_seq: nn.Sequential

    def __init__(self, image_size: int, emblen: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        self.image_size = image_size
        self.descs = descs
        self.nchannels = nchannels

        layers = gen_layers(image_size=image_size, in_channels=nchannels, descs=descs, 
                            layer_fn=nn.Conv2d, device=device)
        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # print(f"{descs[-1].channels=}")
        # print(f"{image_size=}")
        self.linear = nn.Linear(descs[-1].channels * image_size * image_size, emblen)

    def forward(self, inputs: Tensor) -> Tensor:
        # print(f"  Encoder {inputs.shape=}")

        # print(f"    enc conv_seq =\n{self.conv_seq}")
        out = self.conv_seq(inputs)
        # print(f"    enc conv_seq {out.shape=}")

        # print(f"    enc flatten = {self.flatten}")
        out = self.flatten(out)
        # print(f"    enc flatten {out.shape=}")

        # print(f"    enc linear = {self.linear}")
        out = self.linear(out)
        # print(f"    enc linear {out.shape=}")

        return out

class Decoder(nn.Module):
    conv_seq: nn.Sequential

    def __init__(self, emblen: int, image_size: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        self.image_size = image_size
        self.descs = descs
        self.emblen = emblen

        self.linear = nn.Linear(emblen, nchannels * image_size * image_size)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(nchannels, image_size, image_size))

        layers = gen_layers(image_size=image_size, in_channels=nchannels, descs=descs, 
                            layer_fn=nn.ConvTranspose2d, device=device)
        layers.append(nn.Conv2d(descs[-1].channels, nchannels, kernel_size=3, padding=1, stride=1))
        self.conv_seq = nn.Sequential(*layers)
    
    def forward(self, inputs: Tensor) -> Tensor:
        # print(f"  Decoder {inputs.shape=}")

        # print(f"    dec linear = {self.linear}")
        out = self.linear(inputs)
        # print(f"    dec linear {out.shape=}")

        # print(f"    dec unflatten = {self.unflatten}")
        out = self.unflatten(out)
        # print(f"    dec unflatten {out.shape=}")

        # print(f"    dec conv_seq =\n{self.conv_seq}")
        out = self.conv_seq(out)
        # print(f"    dec conv_seq {out.shape=}")

        return out

class ConvEncDec(nn.Module):
    encoder: Encoder
    linear_layers: nn.Sequential
    decoder: Decoder

    def __init__(self, image_size: int, emblen: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        out_image_size = image_size
        for desc in descs:
            out_image_size = desc.get_out_size(out_image_size)
        print(f"{out_image_size=}")

        self.encoder = Encoder(image_size=image_size, emblen=emblen, descs=descs, nchannels=nchannels, device=device)
        self.linear_layers = nn.Identity()
        descs_rev = list(reversed(descs))
        self.decoder = Decoder(image_size=image_size, emblen=emblen, descs=descs_rev, nchannels=nchannels, device=device)
    
    def forward(self, inputs: Tensor) -> Tensor:
        # print(f"ConvEncDec {inputs.shape=}")

        # print(f"encoder:\n{self.encoder}")
        out = self.encoder(inputs)
        # print(f"encoder {out.shape=}")

        # print(f"linear_layers:\n{self.linear_layers}")
        out = self.linear_layers(out)
        # print(f"linear_layers {out.shape=}")

        # print(f"decoder:\n{self.decoder}")
        out = self.decoder(out)
        # print(f"decoder {out.shape=}")

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
        
        if channels == 0:
            raise ValueError("channels not defined. it must be repeated each comma-separated description.")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")

        onedesc = ConvDesc(channels=channels, kernel_size=kernel_size, padding=padding)
        descs.append(onedesc)
    return descs

if __name__ == "__main__":
    import importlib
    descs = gen_descs("k3-p1-c8,c16,c32")
    image_size = 100
    emblen = 64

    net = ConvEncDec(image_size=image_size, emblen=emblen, descs=descs, nchannels=3).to("cuda")
    inputs = torch.rand((1, 3, image_size, image_size), device="cuda")
    print(f"{inputs.shape=}")
    out = net(inputs)
    print(f"{out.shape=}")

