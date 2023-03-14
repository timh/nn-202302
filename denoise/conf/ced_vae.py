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
from denoise_exp import DNExperiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[DNExperiment]

convdesc_str_values = [
    # "k3-s1-mp2-c128,mp2-c64,mp2-c32",
    # "k3-s1-mp2-c64,mp2-c16,mp2-c4"
    # "k3-s2-c64,c32,c8"
    # "k4-s2-c128,k3-s1-c64,k4-s2-c32,k3-s1-c16,k3-s1-c8,k4-s2-c4"
    "k4-s2-c64,c16,c4",
    "k4-s2-c16,c8,c4",
    "k4-s2-c32,c16,c4",
]
# emblen_values = [32, 64, 128]
# emblen_values = [0, 128]
# emblen_values = [0, 4 * 64 * 64]
emblen_values = [32]
# do_variational_values = [True, False]
do_variational_values = [True]
# emblen_values = [64]
nlinear_values = [0]
hidlen_values = [0]
loss_type_values = ["l1"] #, "edge+l1"] #, "l2"]

lr_values = [
    (1e-3, 1e-4, "nanogpt"),
]
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        net = ConvEncDec(**kwargs)
        exp.label += ",latdim_" + "_".join(map(str, net.latent_dim))
        return net
    return fn

layout_values = list(itertools.product(emblen_values, nlinear_values, hidlen_values))
for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(convdesc_str)
    for emblen, nlinear, hidlen in layout_values:   
        for do_variational in do_variational_values:
            for startlr, endlr, sched_type in lr_values:
                for loss_type in loss_type_values:
                    label = convdesc_str
                    extras = dict(emblen=emblen)
                    extras_str = ",".join(f"{k}_{v}" for k, v in extras.items())
                    label = f"{label},{extras_str}"
                    
                    net_args = dict(
                        image_size=cfg.image_size, nchannels=3,
                        emblen=emblen, nlinear=nlinear, hidlen=hidlen, do_variational=do_variational,
                        do_layernorm=True, 
                        descs=descs, device=device
                    )
                    exp_args = net_args.copy()
                    exp_args.pop('descs')
                    exp_args.pop('do_variational')
                    exp_args['conv_descs'] = convdesc_str
                    exp = DNExperiment(label=label, lazy_net_fn=lazy_net_fn(net_args),
                                       startlr=startlr, endlr=endlr, 
                                       optim_type=optim_type, sched_type=sched_type,
                                       loss_type=loss_type,
                                       **exp_args)
                    exp.do_variational = do_variational
                    exps.append(exp)

print(f"{len(exps)=}")
# import random
# random.shuffle(exps)