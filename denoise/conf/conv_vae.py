import sys
import argparse
from typing import List, Dict
import torch
from torch import Tensor, nn
from functools import reduce
import itertools

sys.path.append("..")
sys.path.append("../..")
import model
from model import ConvEncDec
from denoise_exp import DNExperiment
import train_util

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[DNExperiment]

convdesc_str_values = [
    # "k3-s1-mp2-c128,mp2-c64,mp2-c32",
    # "k3-s1-mp2-c64,mp2-c16,mp2-c4"
    # "k3-s2-c64,c32,c8"
    # "k3-s2-c32"
    # "k4-s2-c128,k3-s1-c64,k4-s2-c32,k3-s1-c16,k3-s1-c8,k4-s2-c4"
    # "k4-s2-c64,c16,c4",
    # "k4-s2-c16,c8,c4",
    # "k4-s2-c32,c16,c4",
    "k3-s2-c32,c64,c128,c256,c512",     # pytorch-vae
    "k3-s2-c8,c16,c32,c64,c128",
    "k3-s2-c16,c32,c64,c128,c256",
]
# emblen_values = [0, 4 * 64 * 64]
emblen_values = [0]
do_variational_values = [True]
# do_variational_values = [False]
# loss_type_values = ["l1", "l2"]
loss_type_values = ["l1"]
# kl_weight_values = [2.5e-3, 2.5e-4, 2.5e-5]
kl_weight_values = [2.5e-4]

lr_values = [
    (1e-3, 1e-4, "nanogpt"),
    # (5e-3, 5e-4, "nanogpt"),
    # (5e-3, 5e-4, "nanogpt"),
]
sched_warmup_epochs = 10
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        net = ConvEncDec(**kwargs)
        exp.label += ",latdim_" + "_".join(map(str, net.latent_dim))

        latent_flat = reduce(lambda res, i: res * i, net.latent_dim, 1)
        img_flat = cfg.image_size * cfg.image_size * 3
        ratio = latent_flat / img_flat
        exp.label += f",ratio_{ratio:.3f}"

        return net
    return fn

for convdesc_str in convdesc_str_values:
    descs = model.gen_descs(convdesc_str)
    for emblen in emblen_values:
        for do_variational in do_variational_values:
            for kl_weight in kl_weight_values:
                for startlr, endlr, sched_type in lr_values:
                    for loss_type in loss_type_values:
                        label_parts = [convdesc_str]
                        # label_parts.append(f"emblen_{emblen}")
                        label_parts.append(f"image_size_{cfg.image_size}")
                        if do_variational:
                            label_parts.append(f"kl_weight_{kl_weight:.1E}")
                        if sched_warmup_epochs:
                            label_parts.append(f"warmup_{sched_warmup_epochs}")
                        label = ",".join(label_parts)

                        net_args = dict(
                            image_size=cfg.image_size, nchannels=3, 
                            do_variational=do_variational,
                            emblen=emblen, nlinear=0, hidlen=0, 
                            do_layernorm=False, do_batchnorm=True,
                            descs=descs, device=device
                        )
                        exp_args = net_args.copy()
                        exp_args.pop('device')
                        exp_args.pop('descs')
                        
                        exp = DNExperiment(label=label, 
                                           lazy_net_fn=lazy_net_fn(net_args),
                                           startlr=startlr, endlr=endlr, 
                                               sched_warmup_epochs=sched_warmup_epochs,
                                           optim_type=optim_type, sched_type=sched_type,
                                           conv_descs=convdesc_str,
                                           **exp_args)

                        loss_fn = train_util.get_loss_fn(loss_type)
                        if do_variational:
                            exp.loss_type = f"{loss_type}+kl"
                            exp.label += f",loss_{loss_type}+kl"
                            loss_fn = model.get_kl_loss_fn(exp, kl_weight=kl_weight, backing_loss_fn=loss_fn)
                        else:
                            exp.loss_type = loss_type
                            exp.label += f",loss_{loss_type}"
                        exp.loss_fn = loss_fn

                        exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)