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
    padding: int = 2
    do_bnorm = True
    do_relu = True

class Denoiser(nn.Sequential):
    def __init__(self, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()

        mods: List[nn.Module] = list()
        for i, desc in enumerate(descs):
            if i == 0:
                inchan = nchannels
            else:
                inchan = descs[i - 1].channels
            module = nn.Conv2d(in_channels=inchan, out_channels=desc.channels, kernel_size=desc.kernel_size, stride=desc.stride, padding=desc.padding, device=device)
            mods.append(module)

            if desc.do_bnorm:
                mods.append(nn.BatchNorm2d(desc.channels, device=device))
            if desc.do_relu:
                mods.append(nn.ReLU())

        mods.append(nn.Conv2d(descs[-1].channels, nchannels, kernel_size=3, device=device, padding=1))
        for mod in mods:
            self.append(mod)
        self.descs = descs
        self.nchannels = nchannels
    
def get_optim_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    lr = 1e-3
    optim = torch.optim.AdamW(exp.net.parameters(), lr=1e-3)
    scheduler = trainer.NanoGPTCosineScheduler(optim, lr, lr / 10, warmup_epochs=0, lr_decay_epochs=exp.epochs)
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
