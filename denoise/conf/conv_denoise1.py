import sys
import argparse
from typing import List
import torch
from torch import Tensor, nn


sys.path.append("..")
import model
from model import ConvDesc, ConvDenoiser

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str

batch_size = 128
minicnt = 10


convdesc_strs = [
    "k5-p2-c16,c32,c64,c32,c16",
    "k3-p1-c16,c32,c64,c32,c16",
    "k7-p3-c128,c64,c32,c16"
]
exp_descs = [
    dict(label="conv_denoise1_" + s, net=ConvDenoiser(cfg.image_size, descs=model.gen_descs(s)))
        for s in convdesc_strs
]
