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
from experiment import Experiment
import train_util

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]
dirname: str

conv_layers_str_values = [
    # "k3-s2-32-64-128-256-512",  # pytorch-vae - does ok but doesn't get any better > 100 epochs
    # "k4-s2-32-64-128-256-512",
    # "k4-s2-16-32-64-128-256",    # works pretty well
    # "k4-s2-16-32-32-64-128-256",   # blurry

    # "k7-s2-16-32-64-128-256",   # failed with tensor size mismatch
    # "k6-s2-16-32-64-128-256",   # failed with tensor size mismatch
    # "k5-s2-16-32-64-128-256",   # failed with tensor size mismatch



    # "k5-p2-s2-8-16-32-64-128-256",
    # "k4-s2-8-16-32-64-128-256",
    # "k3-s2-8-16-32-64-128-256",
    # "k3-s1-mp2-8-16-32-64-128-256",  # blurry

    # "k3-s2-32-64-128-256-512",    # pytorch-vae - does well
    # "k3-s1-mp2-32-64-128-256-512",

    # AutoEncoderKL from test_diffusers_aekl.ipynb
    ("k3-s1-128-" +  # conv_in
       (             # down_blocks[0]
        "s1-"
        "128-128-"   # down_blocks[0].resnets[0]
        "128-128-"   # down_blocks[0].resnets[1]
        "s2-128-"    # down_blocks[0].downsamplers[0]
       ) +
       (             # down_blocks[1]
        "s1-"
        "256-256-"   # down_blocks[1].resnets[0]
        "256-256-"   # down_blocks[1].resnets[1]
        "s2-256-"    # down_blocks[1].downsamplers[0]
       ) +
       (             # down_blocks[2]
        "s1-"
        "512-512-"   # down_blocks[2].resnets[0]
        "512-512-"   # down_blocks[2].resnets[1]
        "s2-512-"    # down_blocks[2].downsamplers[0]
       ) +
       (             # down_blocks[3]
        "s1-"
        "512-512-"   # down_blocks[3].resnets[0]
        "512-512-"   # down_blocks[3].resnets[1]
       ) +
       "8"           # conv_out
    )
]
# k5-p2-s2-8-16-32-64-128-256, enc_kern_1, = blurry


# emblen_values = [1024, 2048, 4096, 8192]
# encoder_kernel_size_values = [3, 5, 7]
# encoder_kernel_size_values = [1, 3]
encoder_kernel_size_values = [3]
emblen_values = [0]
if cfg.image_size == 128:
    emblen_values = [ev//2 for ev in emblen_values]

# l1 = blurrier than l2_sqrt
# loss_type_values = ["l1", "l2_sqrt"]
loss_type_values = ["l2_sqrt"]
# kld_weight_values = [2e-5]
kld_weight_values = [2e-6]
# kld_weight_values = [cfg.image_size / 2526] # image size / num samples
inner_nl_values = ['silu']
linear_nl_values = ['silu']
final_nl_values = ['sigmoid']
inner_norm_type_values = ['group']

lr_values = [
    (5e-4, 5e-5, "nanogpt"),
    # (1e-3, 1e-4, "nanogpt"),  # often fails with NaN
]
kld_warmup_epochs = 10
sched_warmup_epochs = 5

# warmup_epochs = 0
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        net = VarEncDec(**kwargs)
        if "ratio" not in exp.label:
            latent_dim = net.encoder.out_dim
            print(f"{latent_dim=}")
            latent_flat = reduce(lambda res, i: res * i, latent_dim, 1)
            img_flat = cfg.image_size * cfg.image_size * 3
            ratio = latent_flat / img_flat
            exp.label += f",ratio_{ratio:.3f}"

        return net
    return fn

twiddles = list(itertools.product(inner_nl_values, linear_nl_values, final_nl_values, inner_norm_type_values))
for conv_layers_str in conv_layers_str_values:
    for emblen in emblen_values:
        for enc_kern_size in encoder_kernel_size_values:
            if emblen != 0 and enc_kern_size != 0:
                continue
            for kld_weight in kld_weight_values:
                # if enc_kern_size:
                #     kld_weight *= 10
                for inner_nl, linear_nl, final_nl, inner_norm_type in twiddles:
                    conv_cfg = conv_types.make_config(conv_layers_str, 
                                                    inner_nl_type=inner_nl,
                                                    linear_nl_type=linear_nl,
                                                    final_nl_type=final_nl,
                                                    inner_norm_type=inner_norm_type)
                    for startlr, endlr, sched_type in lr_values:
                        for loss_type in loss_type_values:
                            label_parts = [conv_layers_str]
                            if emblen:
                                label_parts.append(f"emblen_{emblen}")
                            if enc_kern_size:
                                label_parts.append(f"enc_kern_{enc_kern_size}")
                            label_parts.append(f"image_size_{cfg.image_size}")
                            label_parts.append(f"inl_{inner_nl}")
                            label_parts.append(f"fnl_{final_nl}")
                            label = ",".join(label_parts)

                            net_args = dict(
                                image_size=cfg.image_size, nchannels=3, 
                                emblen=emblen, nlinear=0, hidlen=0, 
                                cfg=conv_cfg, encoder_kernel_size=enc_kern_size,
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
                            exp.kld_weight = kld_weight

                            exps.append(exp)

# exps = exps[:1]
import random
random.shuffle(exps)
print(f"{len(exps)=}")
