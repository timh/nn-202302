import sys
import argparse
# from typing import List, Dict
from typing import List, Dict
# import torch
# from torch import Tensor, nn
# from functools import partial
# import itertools

sys.path.append("..")
sys.path.append("../..")
import model
from model import ConvEncDec
from denoise_exp import DNExperiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[DNExperiment]

loss_type = "l1"
startlr = 1e-3
endlr = 1e-4
sched_type = "nanogpt"
optim_type = "adamw"

# ch_128_64_32
image_size = cfg.image_size
channels = [128, 64, 32]
kernel_size = 3
conv_descr = "k3-s2-c128,c64,c32"
conv_descr_maxpool = "k3-s1-mp2-c128,c64,c32"
kernel_size_values = [3]

def lazy_encdec(kwargs: Dict[str, any]):
    def fn(exp: DNExperiment):
        net = model.ConvEncDec(**kwargs)
        return net
    return fn

def make_exp(kwargs: Dict[str, any], label: str, lazy_net_fn) -> DNExperiment:
    exp = DNExperiment(label=label, image_size=kwargs['image_size'],
                       loss_type=loss_type, startlr=startlr, endlr=endlr,
                       sched_type=sched_type, optim_type=optim_type,
                       lazy_net_fn=lazy_net_fn(kwargs))
    for field, val in kwargs.items():
        setattr(exp, field, val)
    return exp

# ConvEncDec, stride=2, no linear.
encdec_args = dict(image_size=image_size, nchannels=3,
                   emblen=0, nlinear=0, hidlen=0, 
                   do_layernorm=True, do_batchnorm=False, use_bias=True,
                   descs=model.gen_descs(image_size, conv_descr))
encdec_label = f"encdec,{conv_descr},image_size_{image_size}"
encdec_exp = make_exp(encdec_args, encdec_label, lazy_encdec)

# ConvEncDec, maxpool intead of stride=2, no linear.
encdec_maxpool_args = encdec_args.copy()
encdec_maxpool_args['descs'] = model.gen_descs(image_size, conv_descr_maxpool)
encdec_maxpool_label = f"encdec-maxpool,{conv_descr_maxpool},image_size_{image_size}"
encdec_maxpool_exp = make_exp(encdec_maxpool_args, encdec_maxpool_label, lazy_encdec)

# ConvEncDec, maxpool intead of stride=2, no linear.
encdec_maxpool_lin_args = encdec_args.copy()
extras = dict(emblen=128, nlinear=2, hidlen=128)
encdec_maxpool_lin_args['descs'] = model.gen_descs(image_size, conv_descr_maxpool)
extras_str = ",".join([f"{field}={val}" for field, val in extras.items()])
encdec_maxpool_lin_label = f"encdec-maxpool-lin,{conv_descr_maxpool},image_size_{image_size}"
encdec_maxpool_lin_label += f",{extras_str}"
encdec_maxpool_lin_exp = make_exp(encdec_maxpool_lin_args, encdec_maxpool_lin_label, lazy_encdec)

exps = [
    encdec_exp,
    encdec_maxpool_exp,
    encdec_maxpool_lin_exp,
]

print(f"{len(exps)=}")
import random
random.shuffle(exps)
