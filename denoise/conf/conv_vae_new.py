import sys
import argparse
from typing import List, Dict
import torch
from torch import Tensor, nn
from functools import reduce
import itertools

sys.path.append("..")
sys.path.append("../..")
import conv_types
import model_new
from model_new import VarEncDec
# from denoise_exp import DNExperiment
from experiment import Experiment
import train_util

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]
dirname: str

conv_layers_str_values = [
    # "k3-s2-32-16-8",
    "k3-s2-32-64-128-256-512",
]
# emblen_values = [2048, 4096, 8192]
# emblen_values = [1024, 2048, 4096]
emblen_values = [2048, 4096]
loss_type_values = ["l1"]
# kld_weight_values = [0.05]
kld_weight_values = [2e-5]
# kld_weight_values = [cfg.image_size / 2526] # image size / num samples
inner_nl_values = ['relu', 'silu']
linear_nl_values = ['relu', 'silu']
# final_nl_values = ['sigmoid']
final_nl_values = ['silu']
# inner_norm_type_values = ['layer', 'batch', 'group']
inner_norm_type_values = ['layer', 'group']

lr_values = [
    (1e-3, 1e-4, "nanogpt"),
    # (2e-3, 2e-4, "nanogpt"),
    # (5e-3, 5e-4, "nanogpt"),
]
# if cfg.max_epochs > 20:
#     sched_warmup_epochs = 20
#     kld_warmup_epochs = max(cfg.max_epochs // 10, 20)
# else:
#     sched_warmup_epochs = 10
#     kld_warmup_epochs = max(cfg.max_epochs // 10, 10)
kld_warmup_epochs = 10
sched_warmup_epochs = 5

# warmup_epochs = 0
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        net = VarEncDec(**kwargs)
        latent_dim = net.encoder.out_dim
        latent_flat = reduce(lambda res, i: res * i, latent_dim, 1)
        img_flat = cfg.image_size * cfg.image_size * 3
        ratio = latent_flat / img_flat
        exp.label += f",ratio_{ratio:.3f}"

        return net
    return fn

twiddles = list(itertools.product(inner_nl_values, linear_nl_values, final_nl_values, inner_norm_type_values))
for conv_layers_str in conv_layers_str_values:
    for emblen in emblen_values:
        for kld_weight in kld_weight_values:
            for inner_nl, linear_nl, final_nl, inner_norm_type in twiddles:
                conv_cfg = conv_types.make_config(conv_layers_str, 
                                                  inner_nonlinearity_type=inner_nl,
                                                  linear_nonlinearity_type=linear_nl,
                                                  final_nonlinearity_type=final_nl,
                                                  inner_norm_type=inner_norm_type)
                for startlr, endlr, sched_type in lr_values:
                    for loss_type in loss_type_values:
                        label_parts = [conv_layers_str]
                        label_parts.append(f"emblen_{emblen}")
                        label_parts.append(f"image_size_{cfg.image_size}")
                        label = ",".join(label_parts)

                        net_args = dict(
                            image_size=cfg.image_size, nchannels=3, 
                            emblen=emblen, nlinear=0, hidlen=0, 
                            cfg=conv_cfg
                        )
                        exp = Experiment(label=label, 
                                         lazy_net_fn=lazy_net_fn(net_args),
                                         startlr=startlr, endlr=endlr, 
                                             sched_warmup_epochs=sched_warmup_epochs,
                                         optim_type=optim_type, sched_type=sched_type)

                        loss_fn = train_util.get_loss_fn(loss_type)

                        exp.net_layers_str = conv_layers_str
                        exp.loss_type = f"{loss_type}+kl"
                        exp.label += f",loss_{loss_type}+kl"
                        exp.loss_fn = model_new.get_kld_loss_fn(exp, dirname=dirname, kld_weight=kld_weight, backing_loss_fn=loss_fn, kld_warmup_epochs=kld_warmup_epochs)

                        exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)