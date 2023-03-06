import sys
from typing import List, Union, Tuple
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

class ConvDenoiser(nn.Module):
    seq: nn.Sequential

    def __init__(self, image_size: int, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()

        self.seq = nn.Sequential()
        for i, desc in enumerate(descs):
            if i == 0:
                inchan = nchannels
            else:
                inchan = descs[i - 1].channels
            module = nn.Conv2d(in_channels=inchan, out_channels=desc.channels, kernel_size=desc.kernel_size, stride=desc.stride, padding=desc.padding, device=device)
            self.seq.append(module)

            if desc.do_bnorm:
                self.seq.append(nn.BatchNorm2d(desc.channels, device=device))
            if desc.do_relu:
                self.seq.append(nn.ReLU())

        self.seq.append(nn.Conv2d(descs[-1].channels, nchannels, kernel_size=3, device=device, padding=1))
        self.image_size = image_size
        self.descs = descs
        self.nchannels = nchannels
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.seq(inputs)

    
def get_optim_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    lr = 1e-3
    optim = torch.optim.AdamW(exp.net.parameters(), lr=1e-3)
    if not hasattr(exp, "sched_type") or exp.sched_type is None or exp.sched_type == "nanogpt":
        scheduler = trainer.NanoGPTCosineScheduler(optim, lr, lr / 10, warmup_epochs=0, lr_decay_epochs=exp.epochs)
    elif exp.sched_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1.0, total_iters=0)
    else:
        raise ValueError(f"unknown {exp.sched_type=}")

    return optim, scheduler

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
            out = (keep_noise_amount * out_noise) + (1 - keep_noise_amount) * input
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
