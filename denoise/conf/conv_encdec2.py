import sys
import argparse
from typing import List
import torch
from torch import Tensor, nn
from functools import partial

sys.path.append("..")
sys.path.append("../..")
import model
from model import ConvEncDec
from experiment import Experiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

batch_size = 128
minicnt = 4

convdesc_str_values = [
    # "k3-p1-c16,c32,c64",
    # "k3-p1-c6,c12,c24,c48"
    "k3-p1-c3,c4,c5"
]
emblen_values = [256, 512, 1024]
nlinear_values = [2, 4]
# hidlen_values = [128, 256]
lr_values = [
    (1e-3, 1e-4, "nanogpt"),
    (1e-2, 1e-3, "nanogpt"),
    (1e-3, 1e-3, "constant"),
    (1e-4, 1e-4, "constant")
]

for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(convdesc_str)
    for emblen in emblen_values:
        hidlen = emblen
        for nlinear in nlinear_values:
            # for hidlen in hidlen_values:
                for startlr, endlr, sched_type in lr_values:
                    label = f"conv_encdec2_{convdesc_str}"
                    extras = dict(emblen=emblen, nlin=nlinear, hidlen=hidlen)
                    extras_str = ",".join(f"{k}_{v}" for k, v in extras.items())
                    label = f"{label},{extras_str}"
                    lazy_net_fn = \
                        lambda _exp: ConvEncDec(image_size=cfg.image_size, emblen=emblen, 
                                                nlinear=nlinear, hidlen=hidlen, 
                                                descs=descs, nchannels=3, device=device)
                    exp = Experiment(label=label, lazy_net_fn=lazy_net_fn,
                                     startlr=startlr, endlr=endlr, sched_type=sched_type)
                    exps.append(exp)

import random
random.shuffle(exps)