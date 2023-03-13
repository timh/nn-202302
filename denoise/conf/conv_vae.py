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
import model_vae
from denoise_exp import DNExperiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[DNExperiment]

loss_type_values = ["l1"]

lr_values = [
    # (1e-3, 1e-3, "constant"),
    # (1e-3, 1e-4, "nanogpt"),
    (1e-3, 1e-5, "nanogpt"),
]
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp: DNExperiment):
        net = model_vae.VAEModel(**kwargs).to(device)
        exp.label += ",embdim_" + "_".join(map(str, net.embdim))
        exp.embdim = net.embdim
        return net
    return fn

channels_values = [
    [32, 16, 4],
    # [16, 8, 4],
    [64, 16, 4],
    [128, 32, 4],
    [64, 32, 16],
    [128, 64, 32]
]
kernel_size_values = [3]
# nonlinearity_values = ["relu", "sigmoid", "tanh", "leaky-relu", "gelu"]
# nonlinearity_values = ["sigmoid", "leaky-relu", "gelu"]
nonlinearity_values = ["leaky-relu", "gelu"]
# nonlinearity_values = ["sigmoid"]
do_flat_conv2d_values = [True, False]
# do_flat_conv2d_values = [False]
# loss_type_values = ["l1"]
loss_type_values = ["edge+l1", "l1"]

for channels in channels_values:
    for kernel_size in kernel_size_values:
        for startlr, endlr, sched_type in lr_values:
            for loss_type in loss_type_values:
                for nonlinearity in nonlinearity_values:
                    for do_flat_conv2d in do_flat_conv2d_values:
                        label = "ch_" + "_".join(map(str, channels))
                        label += f",image_size_{cfg.image_size}"
                        if do_flat_conv2d:
                            label += ",do_flat_conv2d"
                        label += f",nonlinearity_{nonlinearity}"
                        label += f",kernel_size_{kernel_size}"
                        
                        net_args = dict(
                            image_size=cfg.image_size, 
                            nchannels=3,
                            channels=channels,
                            nonlinearity=nonlinearity,
                            do_flat_conv2d=do_flat_conv2d,
                            kernel_size=kernel_size
                        )
                        exp = DNExperiment(label=label, lazy_net_fn=lazy_net_fn(net_args),
                                        startlr=startlr, endlr=endlr, 
                                        optim_type=optim_type, sched_type=sched_type,
                                        loss_type=loss_type)
                        for field, val in net_args.items():
                            setattr(exp, field, val)
                        
                        exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)