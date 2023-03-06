import sys
import argparse
from typing import List
import torch
from torch import Tensor, nn

sys.path.append("..")
sys.path.append("../..")
import model
from model import ConvDesc, ConvDenoiser
from experiment import Experiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

batch_size = 128
minicnt = 1

convdesc_strs = [
    "k5-p2-c16,c32,c64,c32,c16",
    "k3-p1-c16,c32,c64,c32,c16",
    "k7-p3-c128,c64,c32,c16",
    "k7-p3-c32,c64,c128,c256,c128,c64,c32",
    "k5-p2-c8,c16,c32,c64,c128,c64,c32,c16,c8"
    # "k5-p2-c16,c32,c64,c128,c256,c128,c64,c32,c16",
    "k5-p2-c16,c32,c64,c32,c16",
    "k5-p2-c16,c32,c64,c128,c256,c512,c128,c64,c32,c16",
]

exps = [
    Experiment(label="conv_denoise1_" + s, 
               lazy_net_fn=lambda _exp: ConvDenoiser(cfg.image_size, descs=model.gen_descs(s)))
        for s in convdesc_strs
]
