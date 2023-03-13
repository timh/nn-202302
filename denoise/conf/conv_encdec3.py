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
    # "k4-s2-c32,c64,c64",
    "k3-s2-c32,c64,c64",
    # "k4-s2-c64,c32,c16",
    # "k4-s2-c64,c32,c32,c16",
    # "k4-s2-c32,c16,c8"
]
emblen_values = [64]   # 64 * 64
# nlinear_values = [0, 1, 2, 3]
nlinear_values = [0, 1, 2]
# hidlen_values = [128, 256, 384]
hidlen_values = [384]
do_batchnorm_values = [False]
do_layernorm_values = [True]
# loss_type_values = ["edge*l1", "edge+l1", "l1"]
loss_type_values = ["l1"]

# NOTE: "mape" seems to deliver different sorting than the others. l1, rpd, 
# distance, l2 all generally ~agree about winners.
lr_values = [
    (1e-3, 1e-3, "constant"),
    # (1e-3, 1e-4, "nanogpt"),
]
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(_exp):
        return ConvEncDec(**kwargs)
    return fn

layout_values = list(itertools.product(emblen_values, nlinear_values, hidlen_values))
model_twiddles = list(itertools.product(do_batchnorm_values, do_layernorm_values))
for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(cfg.image_size, convdesc_str)
    for emblen, nlinear, hidlen in layout_values:   
        for startlr, endlr, sched_type in lr_values:
            for do_batchnorm, do_layernorm in model_twiddles:
                for loss_type in loss_type_values:
                    label = convdesc_str
                    extras = dict(emblen=emblen, nlin=nlinear, hidlen=hidlen)
                    extras_str = ",".join(f"{k}_{v}" for k, v in extras.items())
                    label = f"{label},{extras_str}"
                    
                    net_args = dict(
                        image_size=cfg.image_size, nchannels=3,
                        emblen=emblen, nlinear=nlinear, hidlen=hidlen, 
                        do_layernorm=do_layernorm, do_batchnorm=do_batchnorm,
                        use_bias=True, 
                        descs=descs, device=device
                    )
                    exp_args = net_args.copy()
                    exp_args.pop('descs')
                    exp_args['conv_descs'] = convdesc_str
                    exp = DNExperiment(label=label, lazy_net_fn=lazy_net_fn(net_args),
                                       startlr=startlr, endlr=endlr, 
                                       optim_type=optim_type, sched_type=sched_type,
                                       loss_type=loss_type,
                                       **exp_args)
                    exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)