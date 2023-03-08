import sys
import argparse
from typing import List, Dict
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
    "k3-s2-op1-p1-c32,c64,c64"
]
emblen_values = [384]
nlinear_values = [0, 1, 2, 3]
hidlen_values = [128, 384]
do_batchnorm_values = [True]
lr_values = [
    (1e-3, 1e-3, "constant"),
]

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(_exp):
        return ConvEncDec(**kwargs)
    return fn

for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(convdesc_str)
    for emblen in emblen_values:
        # hidlen = emblen
        for nlinear in nlinear_values:
            for hidlen in hidlen_values:
                for startlr, endlr, sched_type in lr_values:
                    for do_batchnorm in do_batchnorm_values:
                        label = convdesc_str
                        extras = dict(emblen=emblen, nlin=nlinear, hidlen=hidlen)
                        extras_str = ",".join(f"{k}_{v}" for k, v in extras.items())
                        label = f"{label},{extras_str}"
                        if do_batchnorm:
                            label += ",bnorm"
                        
                        args = dict(image_size=cfg.image_size, emblen=emblen, 
                                    nlinear=nlinear, hidlen=hidlen, 
                                    do_layernorm=False, do_batchnorm=do_batchnorm,
                                    descs=descs, nchannels=3, device=device)
                        exp = Experiment(label=label, lazy_net_fn=lazy_net_fn(args),
                                         startlr=startlr, endlr=endlr, sched_type=sched_type)
                        exps.append(exp)

import random
random.shuffle(exps)