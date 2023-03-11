import sys
import argparse
from typing import List, Dict
import torch
from torch import Tensor, nn
from functools import partial
import itertools

sys.path.append("..")
sys.path.append("../..")
import model
from model import ConvEncDec
from experiment import Experiment
from denoise_exp import DNExperiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

# TODO - use allow bias=False for ConvNet
convdesc_str_values = [
    "k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024",
]
emblen_values = [0, 32, 256]
nlinear_values = [0, 1, 2]
hidlen_values = [128, 384]
do_batchnorm_values = [False]
do_layernorm_values = [True]
use_bias_values = [True]
# use_bias_values = [False]
# loss_type_values = ["l1", "edge+l1", "l2"]
loss_type_values = ["edge+l1"]

lr_values = [
    # (5e-4, 5e-4, "constant"),
    (5e-3, 5e-4, "nanogpt"),
    (1e-3, 1e-4, "nanogpt"),
    # (5e-4, 5e-5, "nanogpt"),
]

def lazy_net_fn(**kwargs):
    def fn(_exp):
        return ConvEncDec(**kwargs)
    return fn

layout_values = list(itertools.product(emblen_values, nlinear_values, hidlen_values))
model_twiddles = list(itertools.product(do_batchnorm_values, do_layernorm_values, use_bias_values))
for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(cfg.image_size, convdesc_str)
    for emblen, nlinear, hidlen in layout_values:   
        for startlr, endlr, sched_type in lr_values:
            for do_batchnorm, do_layernorm, use_bias in model_twiddles:
                for loss_type in loss_type_values:
                    label = convdesc_str
                    extras = dict(emblen=emblen, nlin=nlinear, hidlen=hidlen)
                    extras_str = ",".join(f"{k}_{v}" for k, v in extras.items())
                    label = f"{label},{extras_str}"
                    
                    net_args = dict(image_size=cfg.image_size, emblen=emblen, 
                                    nlinear=nlinear, hidlen=hidlen, 
                                    do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, 
                                    use_bias=use_bias,
                                    descs=descs, nchannels=3, device=device)
                    exp_args = net_args.copy()
                    exp_args.pop('descs')
                    exp_args['conv_descs'] = convdesc_str
                    exp = DNExperiment(label=label, lazy_net_fn=lazy_net_fn(**net_args),
                                       startlr=startlr, endlr=endlr, sched_type=sched_type,
                                       loss_type=loss_type,
                                       **exp_args)
                    exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)