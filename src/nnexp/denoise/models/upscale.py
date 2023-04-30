# %%
from typing import List
from dataclasses import dataclass
import re

import torch
from torch import nn, FloatTensor

from nnexp import base_model
from .model_shared import SelfAttention

GN_GROUPS = 4

# 4   4k5   4s2   4k5s2
RE_CHAN = re.compile(r"^(\d+)(?:k(\d+))?(?:s(\d+))?$")
@dataclass
class Desc:
    in_chan: int = None
    out_chan: int = None
    kernel_size: int = None
    stride: int = None
    sa_nheads: int = None

    @staticmethod
    def parse_channels(in_chan: int | None, s: str) -> List['Desc']:
        res: List[Desc] = list()
        for part in s.split("-"):
            if part.startswith("sa"):
                sa_nheads = int(part[2:])
                res.append(Desc(sa_nheads=sa_nheads, in_chan=in_chan, out_chan=in_chan))
                continue

            match = RE_CHAN.match(part)
            if not match:
                raise ValueError(f"{part} doesn't match regex")

            chan_str, kern_str, stride_str = match.groups()
            chan = int(chan_str)
            kern = int(kern_str or 3)
            stride = int(stride_str or 1)

            if in_chan is None:
                in_chan = chan

            chandesc = Desc(in_chan=in_chan, out_chan=chan, kernel_size=kern, stride=stride)
            res.append(chandesc)

            in_chan = chan
        return res

class ResnetBlock(nn.Sequential):
    # (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
    # (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
    # (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
    # (dropout): Dropout(p=0.0, inplace=False)
    # (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (nonlinearity): SiLU()
    def __init__(self, in_chan: int, out_chan: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(in_chan, in_chan)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=padding)

        self.norm2 = nn.GroupNorm(out_chan, out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size, padding=padding)

        self.nonlinearity = nn.SiLU()

class DownChannelBlock(nn.Sequential):
    def __init__(self, chandesc: Desc):
        super().__init__()
        self.resnet = ResnetBlock(chandesc.in_chan, chandesc.out_chan, kernel_size=chandesc.kernel_size)

        if chandesc.stride > 1:
            padding = (chandesc.kernel_size - 1) // 2
            self.downsize = \
                nn.Conv2d(chandesc.in_chan, chandesc.out_chan, 
                          kernel_size=chandesc.kernel_size, stride=chandesc.stride,
                          padding=padding)
            
class UpChannelBlock(nn.Sequential):
    def __init__(self, chandesc: Desc):
        super().__init__()

        self.resnet = ResnetBlock(chandesc.in_chan, chandesc.out_chan, kernel_size=chandesc.kernel_size)

        if chandesc.stride > 1:
            padding = (chandesc.kernel_size - 1) // 2
            self.upsize = \
                nn.ConvTranspose2d(chandesc.in_chan, chandesc.out_chan, 
                                   kernel_size=chandesc.kernel_size, stride=chandesc.stride,
                                   padding=padding, output_padding=1)
            
class UpscaleModel(base_model.BaseModel):
    _model_fields = 'down_str up_str'.split()
    _metadata_fields = _model_fields + ['latent_chan']

    def __init__(self, *,
                 down_str: str, 
                 up_str: str):
        super().__init__()

        self.down_str = down_str
        self.up_str = up_str

        self.upsize = nn.Upsample(scale_factor=2)

        down_descs = Desc.parse_channels(None, down_str)
        self.in_conv = nn.Conv2d(3, down_descs[0].in_chan, kernel_size=1, stride=1)
        self.channel_blocks = nn.Sequential()
        for down in down_descs:
            if down.sa_nheads:
                self.channel_blocks.append(SelfAttention(down.in_chan, down.sa_nheads))
            else:
                self.channel_blocks.append(DownChannelBlock(down))

        self.latent_chan = down_descs[-1].out_chan
        up_descs = Desc.parse_channels(self.latent_chan, up_str)
        for up in up_descs:
            if up.sa_nheads:
                self.channel_blocks.append(SelfAttention(up.in_chan, up.sa_nheads))
            else:
                self.channel_blocks.append(UpChannelBlock(up))

        self.out_conv = nn.Conv2d(up_descs[-1].out_chan, 3, kernel_size=1, stride=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input: FloatTensor) -> FloatTensor:
        out = self.upsize(input)

        out = self.in_conv(out)
        for chan_mod in self.channel_blocks:
            out = chan_mod.forward(out)
        out = self.out_conv(out)

        return self.sigmoid(out)
