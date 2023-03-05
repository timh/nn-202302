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

class Denoiser(nn.Module):
    def __init__(self, descs: List[ConvDesc], nchannels = 3, device = "cpu"):
        super().__init__()
        convs: List[nn.Conv2d] = list()
        for i, desc in enumerate(descs):
            if i == 0:
                inchan = nchannels
            else:
                inchan = descs[i - 1].channels
            module = nn.Conv2d(in_channels=inchan, out_channels=desc.channels, kernel_size=desc.kernel_size, stride=desc.stride, padding=desc.padding, device=device)
            convs.append(module)

            if desc.do_bnorm:
                convs.append(nn.BatchNorm2d(desc.channels, device=device))

        convs.append(nn.Conv2d(descs[-1].channels, nchannels, kernel_size=3, device=device, padding=1))
        self.convs = nn.Sequential(*convs)
        self.descs = descs
        self.nchannels = nchannels
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.convs(inputs)

def get_optim_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    lr = 1e-3
    optim = torch.optim.AdamW(exp.net.parameters(), lr=1e-3)
    scheduler = trainer.NanoGPTCosineScheduler(optim, lr, lr / 10, warmup_epochs=0, lr_decay_epochs=exp.epochs)
    return optim, scheduler
