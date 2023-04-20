from typing import List, Dict, Callable
import itertools
import argparse
from dataclasses import dataclass

from torch import nn

from experiment import Experiment
from models import vae, denoise_new

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = denoise_new.DenoiseModelNew(**kwargs)
        # print(net)
        return net
    return fn

@dataclass
class Config:
    channels: List[int]
    nstride1: int

    time_pos: denoise_new.EmbedPos
    sa_pos: denoise_new.EmbedPos
    ca_pos: denoise_new.EmbedPos

    sa_nheads: int
    ca_nheads: int

# "k3-sa8-256-t+256-ca8",          # mgduhr - ~0.095
# "k3-sa8-128-t+128-ca8",          # delnzi - .103, nan
# "k3-sa8-128-t+256-ca8",          # onhdjw - .097, nan
configs = [
    # ggwqoz
    Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # evnalj
    Config(channels=[256], nstride1=3, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),
    Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='res_first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    Config(channels=[256, 512], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),
    Config(channels=[512], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),
    Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=16, ca_nheads=16),

]

# 

twiddles = itertools.product(
    configs,              # config
    ["l2"],               # loss_type
    # [1.0, 10.0, 100.0],               # clip_scale_default
    [10.0],
)

for config, loss_type, clip_scale_default in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    args = dict(**config.__dict__)

    exp.label = ",".join([
        "denoise_dnnew",
        "chan_" + "_".join(map(str, config.channels)),
        f"ns1_{config.nstride1}",
        f"time_{config.time_pos}",
        f"sa_{config.sa_pos}_{config.sa_nheads}",
        f"ca_{config.ca_pos}_{config.ca_nheads}",
    ])

    print(f"args here: {args}")

    args['in_chan'] = lat_chan
    args['in_size'] = lat_size
    args['clip_emblen'] = cfg.clip_emblen
    args['clip_scale_default'] = clip_scale_default
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)
