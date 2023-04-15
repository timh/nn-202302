import sys
import argparse
from typing import List, Dict
from functools import reduce
import itertools

sys.path.append("..")
sys.path.append("../..")
import conv_types
from models import vae
from models.vae import VarEncDec
from experiment import Experiment
import train_util

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

def layer(size: int, perlayer: int) -> str:
    return f"{size}x{perlayer}-{size}s2"

def layers(nlayers: int, perlayer: int, start_chan: int, end_chan: int = 8) -> str:
    layer_strs: List[str] = list()
    in_chan = start_chan
    for _ in range(nlayers):
        layer_strs.append(layer(in_chan, perlayer))
        in_chan //= 2

    return "k3-" + "-".join(layer_strs) + f"-{end_chan}"

conv_layers_str_values = [
    layers(nlayers=3, perlayer=2, start_chan=128, end_chan=4),  # 0.02
    layers(nlayers=3, perlayer=2, start_chan=256, end_chan=4),  # 0.02

    layers(nlayers=2, perlayer=2, start_chan=64, end_chan=1),   # 0.02
    "k3-256x2-mp2-128x2-mp2-64x2-mp2-4",

    # layers(nlayers=2, perlayer=2, end_chan=8),    # the best @ size 256. ratio 0.04
    # layers(nlayers=3, perlayer=2, end_chan=8),      # ratio 0.02
    # layers(nlayers=3, perlayer=2, end_chan=4),    # ratio 0.005

    # layers(nlayers=3, perlayer=2, end_chan=8),           # gdyfmh
    # "k3-s1-256x2-s2-256-s1-128x2-s2-128-s1-64x2-s2-64-8" # gdyfmh
]

# python train_ae.py -b 16 --amp -n 500 --startlr 2e-3 --endlr 2e-4 --resume -c conf/ae_vae.py -I 256 -d images.alex-1024

# very good:
# vae 'kepvzt': 
# - k3-s1-256x2-s2-256-s1-128x2-s2-128-8
# - kld_weight 2e-5  
encoder_kernel_size_values = [3]

# l1 = blurrier than l2_sqrt
loss_type_values = ["edge+l2_sqrt"]
# kld_weight_values = [2e-4, 2e-5, 2e-6]
kld_weight_values = [2e-4]
inner_nl_values = ['silu']
linear_nl_values = ['silu']
final_nl_values = ['sigmoid']
inner_norm_type_values = ['group']
do_residual_values = [False]

kld_warmup_epochs = 10

def lazy_net_fn(kwargs: Dict[str, any]):
    def fn(exp):
        return VarEncDec(**kwargs)
    return fn

twiddles = list(itertools.product(inner_nl_values, linear_nl_values, final_nl_values, inner_norm_type_values, do_residual_values))
for conv_layers_str in conv_layers_str_values:
    for enc_kern_size in encoder_kernel_size_values:
        for kld_weight in kld_weight_values:
            for inner_nl, linear_nl, final_nl, inner_norm_type, do_residual in twiddles:
                conv_cfg = conv_types.make_config(layers_str=conv_layers_str, 
                                                  in_chan=3, in_size=cfg.image_size,
                                                  inner_nl_type=inner_nl,
                                                  linear_nl_type=linear_nl,
                                                  final_nl_type=final_nl,
                                                  inner_norm_type=inner_norm_type)
                latent_dim = conv_cfg.get_out_dim('down')
                latent_flat = reduce(lambda res, i: res * i, latent_dim, 1)
                img_flat = cfg.image_size * cfg.image_size * 3
                ratio = latent_flat / img_flat
                latent_dim_str = "_".join(map(str, latent_dim))

                for loss_type in loss_type_values:
                    label_parts = [conv_cfg.layers_str()]
                    if enc_kern_size:
                        label_parts.append(f"enc_kern_{enc_kern_size}")
                    if do_residual:
                        label_parts.append("residual")
                    label_parts.append(f"klw_{kld_weight:.1E}")

                    label_parts.append(f"latdim_{latent_dim_str}")
                    label_parts.append(f"ratio_{ratio:.3f}")
                    label = ",".join(label_parts)

                    net_args = dict(
                        image_size=cfg.image_size, nchannels=3, 
                        emblen=0, nlinear=0, hidlen=0, 
                        cfg=conv_cfg, encoder_kernel_size=enc_kern_size,
                        do_residual=do_residual
                    )
                    exp = Experiment(label=label, 
                                    lazy_net_fn=lazy_net_fn(net_args))

                    loss_fn = train_util.get_loss_fn(loss_type)

                    exp.loss_type = f"{loss_type}+kl"
                    exp.label += f",loss_{loss_type}+kl"
                    exp.loss_fn = vae.get_kld_loss_fn(exp, dirname="", kld_weight=kld_weight, backing_loss_fn=loss_fn, kld_warmup_epochs=kld_warmup_epochs)
                    exp.kld_weight = kld_weight

                    exps.append(exp)

# exps = exps[:1]
# import random
# random.shuffle(exps)
