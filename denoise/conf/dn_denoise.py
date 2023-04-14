from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, denoise2

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net_ae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = denoise2.DenoiseModel2(**kwargs)
        # print(net)
        return net
    return fn

layers_str_list = [
                            # at 19 epochs:
    "k3-sa8-128x2",         # #1, #6, boring brown, boring blue
                            # #8 kinda cool with l2
    "k3-128-sa8-128",       # #4, #7, interesting, high contrast
    "k3-sa8-128-sa8-128",   # #3, #5, boring all light blue
    "k3-sa8-128-sa4-128",   # #2, some cool shapes
    "k3-sa4-128-sa4-128",   # 
    "k3-sa4-128-sa8-128",
    # "k3-128-128-sa8",
    # "k3-256-sa8-256",
    # "k3-128-sa8-128-sa8-256-sa8-256",
]
    
twiddles = itertools.product(
    layers_str_list,           # layers_str
    ["l1_smooth", "l1", "l2"], # loss_type
    [True],                    # do_residual
)

for layers_str, loss_type, do_residual in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    conv_cfg = conv_types.make_config(in_chan=lat_chan, in_size=lat_size, layers_str=layers_str,
                                      inner_nl_type='silu', inner_norm_type='group')
    args = dict(in_size=lat_size, in_chan=lat_chan, cfg=conv_cfg, do_residual=do_residual)
    print(f"ARGS: {args}")

    exp.label = ",".join([
        "denoise_dn",
        f"layers_{conv_cfg.layers_str()}"
    ])
    if do_residual:
        exp.label += ",residual"

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_ae(args)

    exps.append(exp)
