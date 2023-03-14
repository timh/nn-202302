import sys
import argparse
from typing import List, Dict, Callable
import torch
from torch import Tensor, nn

sys.path.append("..")
sys.path.append("../..")
import model_vanvae
from experiment import Experiment
import train_util

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

hidden_dims_values = [
    [32, 64, 128, 256, 512],  # from PyTorch-VAE
    # [16, 32, 64, 128, 256],
    # [32, 64, 128],
]
latent_dim_values = [128]      # from PyTorch-VAE
kld_weight_values = [2.5e-4]   # from PyTorch-VAE
#kld_weight_values += [2.5e-3, 2.5e-5]

lr_values = [
    # (1e-3, 1e-4, "nanogpt"),
    # (0.005, ) from 
    (5e-3, 5e-4, "nanogpt") # 5e-3 = start for PyTorch-VAE
]
loss_type_values = ["l1", "l2"]
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        net = model_vanvae.VanillaVAE(**kwargs)
        return net
    return fn

def loss_fn(exp: Experiment, kld_weight: float, recons_loss_type: str) -> Callable[[Tensor, Tensor], Tensor]:
    recons_loss_fn = train_util.get_loss_fn(recons_loss_type)
    def fn(output: List[any], truth: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        net: model_vanvae.VanillaVAE = exp.net
        kld_loss = net.loss_function()
        recons_loss = recons_loss_fn(output, truth)
        loss = recons_loss + kld_weight * kld_loss
        # print(f"loss: kld={kld_loss:.5f} {recons_loss_type}={recons_loss:.5f}: result={loss:.5f}")
        return loss
    return fn

for hidden_dims in hidden_dims_values:
    for latent_dim in latent_dim_values:
        for kld_weight in kld_weight_values:
            for loss_type in loss_type_values:
                for startlr, endlr, sched_type in lr_values:
                    hdims_str = "_".join(map(str, [h for h in hidden_dims]))
                    label = f"lat_dim_{latent_dim}"
                    label += f",hid_dims_{hdims_str}"
                    label += f",kld_weight_{kld_weight:.1E}"
                    label += f",loss_{loss_type}+kld"
                    
                    net_args = dict(
                        in_channels=3, hidden_dims=hidden_dims, latent_dim=latent_dim, image_size=cfg.image_size,
                    )
                    exp_args = net_args.copy()
                    exp = Experiment(label=label, lazy_net_fn=lazy_net_fn(net_args),
                                        startlr=startlr, endlr=endlr, 
                                        optim_type=optim_type, sched_type=sched_type)
                    exp.loss_fn = loss_fn(exp, kld_weight, loss_type)
                    exps.append(exp)

print(f"{len(exps)=}")
# import random
# random.shuffle(exps)