from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, linear

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net_linear(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        # net = linear.DenoiseLinear(in_chan, in_size, nlayers)
        net = linear.DenoiseLinear(**kwargs)
        # print(net)
        return net
    return fn
    
twiddles = itertools.product(
    [1, 2, 4],                   # nlayers
    ["l1_smooth"]                # loss_type
    # ["l2", "l1", "l1_smooth"]
)

for nlayers, loss_type in twiddles:
    exp = Experiment()
    latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = latent_dim

    args = dict(in_chan=lat_chan, in_size=lat_size, nlayers=nlayers)

    exp.label = ",".join(["denoise_linear", f"nlayers_{nlayers}"])
    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_linear(args)

    exps.append(exp)
