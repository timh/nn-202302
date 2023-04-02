from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import ae_simple

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: ae_simple.AEDenoise
cfg: argparse.Namespace

def lazy_net_ae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = ae_simple.AEDenoise(**kwargs)
        print(net)
        return net
    return fn
    
twiddles = itertools.product(
    ["k3-s1-mp2-16-mp2-32-mp2-64"],    # layers_str
    # ["l2", "l1", "l1_smooth"]        # loss_type
    ["l1_smooth"]                      # loss_type
)

for layers_str, loss_type in twiddles:
    exp = Experiment()
    latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = latent_dim

    conv_cfg = conv_types.make_config(layers_str=layers_str)
    args = dict(image_size=lat_size, nchannels=lat_chan, cfg=conv_cfg)
    print(f"ARGS: {args}")

    exp.label = "denoise_ae"
    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_ae(args)

    exps.append(exp)
