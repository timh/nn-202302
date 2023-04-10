from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, denoise

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net_ae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = denoise.DenoiseModel(**kwargs)
        # print(net)
        return net
    return fn

layers_str_list = [
    # "k3-s1-64x2-s2-256-s1-256x2-s2-1024",
    # "k3-" + "-".join([f"s1-{c}x2-s2-{c}" for c in [64, 128, 256, 512, 1024]]),
    # "k3-" + "-".join([f"s1-{c}x2-s2-{c}" for c in [64, 256, 1024]]),
    # "k3-" + "-".join([f"s1-{c}x2-s2-{c}" for c in [64, 128, 256]]),
    # "k3-" + "-".join([f"s1-{c}x2-s2-{c}" for c in [64, 128, 256]]),

    "k3-s1-32x2",
    "k3-s1-32x2-64x2",
    "k3-s1-32x2-64x2-128x2",
    "k3-s1-64x2",
    "k3-s1-64x2-128x2",
    "k3-s1-64x2-128x2-256x2",
    "k3-s1-128x2",
    "k3-s1-128x2-256x2",
    "k3-s1-128x2-256x2-512x2",
    "k3-s1-256x2",
    "k3-s1-256x2-512x2",
    "k3-s1-256x2-512x2-1024x2",


    # "k3-s1-128x2-s2-256-s1-512x2-s2-1024",
    # "k3-s1-128x1-s2-256-s1-512x1-s2-1024",
    # "k3-s1-128x3-s2-256-s1-512x3-s2-1024",
    # "k3-s1-256x2-s2-256-s1-128x2-s2-128-s1-64x2-s2-64",
]
    
twiddles = itertools.product(
    layers_str_list,          # layers_str
    ["l1_smooth"],            # loss_type
)

for layers_str, loss_type in twiddles:
    exp = Experiment()
    latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = latent_dim

    conv_cfg = conv_types.make_config(in_chan=lat_chan, in_size=lat_size, layers_str=layers_str,
                                      inner_nl_type='silu')
    args = dict(in_size=lat_size, in_chan=lat_chan, cfg=conv_cfg)
    print(f"ARGS: {args}")

    exp.label = ",".join([
        "denoise_ae",
        f"layers_{layers_str}"
    ])

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_ae(args)

    exps.append(exp)
