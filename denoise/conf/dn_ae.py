from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, ae_simple

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net_ae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = ae_simple.AEDenoise(**kwargs)
        # print(net)
        return net
    return fn

layers_str_list = [
    "k3-s1-mp2-16-mp2-32-mp2-64",
    "k3-s1-" "16x2-mp2-16-mp0-" "32x2-mp2-32-mp0-" "64x2-mp2-64",
    "k3-s1-mp2-16-mp2-32",
    "k3-s1-16x4",
    "k3-s1-32x4",
    "k3-s1-64x4",
    "k3-s1-128x4",
]
    
twiddles = itertools.product(
    layers_str_list,
    # ["l2", "l1", "l1_smooth"]        # loss_type
    # ["l1_smooth"]                      # loss_type
    ["l2"]                      # loss_type
)

for layers_str, loss_type in twiddles:
    exp = Experiment()
    latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = latent_dim

    conv_cfg = conv_types.make_config(layers_str=layers_str)
    args = dict(image_size=lat_size, nchannels=lat_chan, cfg=conv_cfg)
    print(f"ARGS: {args}")

    exp.label = ",".join([
        "denoise_ae",
        f"layers_{layers_str}"
    ])

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_ae(args)

    exps.append(exp)
