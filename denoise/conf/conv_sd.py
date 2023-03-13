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

ch_values = [32, 64]
out_ch_values = [3]
num_res_blocks_values = [1, 2, 4]
loss_type_values = ["l1"]

# NOTE: "mape" seems to deliver different sorting than the others. l1, rpd, 
# distance, l2 all generally ~agree about winners.
lr_values = [
    # (1e-4, 1e-5, "constant"),
    (1e-4, 1e-5, "nanogpt"),
]
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    import model_sd
    def fn(_exp):
        return model_sd.Model(use_timestep=cfg.use_timestep, **kwargs)
    return fn

for ch in ch_values:
    for out_ch in out_ch_values:
        for num_res_blocks in num_res_blocks_values:
            for loss_type in loss_type_values:
                for startlr, endlr, sched_type in lr_values:
                    args = dict(
                        ch=ch, out_ch=out_ch, num_res_blocks=num_res_blocks,
                        in_channels=3, resolution=cfg.image_size
                    )
                    label = ",".join([f"{k}_{v}" for k, v in args.items()])
                    exp = Experiment(label=label, lazy_net_fn=lazy_net_fn(args),
                                     startlr=startlr, endlr=endlr, 
                                     optim_type=optim_type, sched_type=sched_type,
                                     loss_type=loss_type)
                    for k, v in args.items():
                        setattr(exp, k, v)
                    exps.append(exp)

print(f"{len(exps)=}")
import random
random.shuffle(exps)