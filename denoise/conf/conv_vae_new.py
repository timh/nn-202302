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
    # "k4-s2-32-64-128-256-512",
    # "k4-s2-16-32-64-128-256",        # works pretty well
    # "k4-s2-16-32-32-64-128-256",     # blurry

    # "k7-s2-16-32-64-128-256",        # failed with tensor size mismatch
    # "k6-s2-16-32-64-128-256",        # failed with tensor size mismatch
    # "k5-s2-16-32-64-128-256",        # failed with tensor size mismatch

    # "k3-s1-mp2-8-16-32-64-128-256",  # blurry
    # "k3-s1-mp2-32-64-128-256-512",   # blurry

    # "k5-p2-s2-8-16-32-64-128-256",
    # "k4-s2-8-16-32-64-128-256",
    # "k3-s2-8-16-32-64-128-256",

    # "k3-s2-32-64-128-256-512",       # pytorch-vae - does well

    # "k3-s1-128x3-s2-128-s1-256x2-s2-256-s1-512x2-s2-512-s1-512x2-8",
]

# in the style of diffusers.AutoencoderKL (but without residual connections)
def make_aekl_layers(num_stride1 = 4, last_chan = 8) -> str:
    downblocks = list()
    downblocks.append(f"k3-s1-{cfg.image_size // 4}")                    # conv_in

    size4 = cfg.image_size // 4
    size2 = cfg.image_size // 2
    size1 = cfg.image_size

    downblocks.append(f"s1-{size4}x{num_stride1}")   # down_blocks[0].resnets[0-num_stride1]
    downblocks.append(f"s2-{size4}")                 # down_blocks[0].downsample

    downblocks.append(f"s1-{size2}x{num_stride1}")   # down_blocks[1].resnets[0-num_stride1]
    downblocks.append(f"s2-{size2}")                 # down_blocks[1].downsample

    downblocks.append(f"s1-{size1}x{num_stride1}")   # down_blocks[2].resnets[0-num_stride1]
    downblocks.append(f"s2-{size1}")                 # down_blocks[2].downsample

    downblocks.append(f"s1-{size1}x{num_stride1}")   # down_blocks[3].resnets[0-num_stride1]
    downblocks.append(str(last_chan))                # conv_out

    return "-".join(downblocks)

# num1x1 = 2, last_chan = 8 seems to work well and fast.
conv_layers_str_values.clear()
for num1x1 in [2]:
    # for last_chan in [4, 8]:
    for last_chan in [8]:
        conv_layers_str_values.append(make_aekl_layers(num_stride1=num1x1, last_chan=last_chan))

            
encoder_kernel_size_values = [3]
emblen_values = [0]

# l1 = blurrier than l2_sqrt
# loss_type_values = ["l1", "l2_sqrt"]
# loss_type_values = ["l2_sqrt", "edge+l2_sqrt"]
loss_type_values = ["edge+l2_sqrt"]
kld_weight_values = [2e-4, 2e-5, 0.1]
# kld_weight_values = [2e-6]
# kld_weight_values = [cfg.image_size / 2526] # image size / num samples
inner_nl_values = ['silu']
linear_nl_values = ['silu']
final_nl_values = ['sigmoid']
inner_norm_type_values = ['group']

lr_values = [
    # (5e-4, 5e-5, "nanogpt"),
    (1e-3, 1e-4, "nanogpt"),  # often fails with NaN
]
kld_warmup_epochs = 10
sched_warmup_epochs = 5

# warmup_epochs = 0
optim_type = "adamw"

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        return VarEncDec(**kwargs)
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
                    sizes = conv_cfg.get_sizes_down_actual(in_size=cfg.image_size)
                    channels = conv_cfg.get_channels_down(nchannels=3)
                    latent_dim = [channels[-1], sizes[-1], sizes[-1]]
                    print(f"{latent_dim=}")
                    latent_flat = reduce(lambda res, i: res * i, latent_dim, 1)
                    img_flat = cfg.image_size * cfg.image_size * 3
                    ratio = latent_flat / img_flat
                    latent_dim_str = "_".join(map(str, latent_dim))

                    for startlr, endlr, sched_type in lr_values:
                        for loss_type in loss_type_values:
                            label_parts = [conv_layers_str]
                            if emblen:
                                label_parts.append(f"emblen_{emblen}")
                            if enc_kern_size:
                                label_parts.append(f"enc_kern_{enc_kern_size}")

                            label_parts.append(f"latdim_{latent_dim_str}")
                            label_parts.append(f"ratio_{ratio:.3f}")
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
