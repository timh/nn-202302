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



# NOTE for loss = l2 NOTE
# "k3-sa8-256-t+256-ca8",          # mgduhr - ~0.095
# "k3-sa8-128-t+128-ca8",          # delnzi - .103, nan
# "k3-sa8-128-t+256-ca8",          # onhdjw - .097, nan
configs = [
    # ggwqoz @ scale 10: 0.00688 at 199 epochs - NOTE looks bad
    # xvunyd @ scale  1: 0.00846 at 19 epochs
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # evnalj @ scale 10: 0.01427 at 183 epochs
    # Config(channels=[256], nstride1=3, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # azwczh @ scale 10: 0.00693 at 199 epochs - NOTE looks great @ --clip_scale 5 or 6
    # uzvhrg @ scale  1: 0.00867 at 19 epochs
    # Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # jdbveb @ scale 1: 0.00854 @ 19
    # Config(channels=[128], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # xqusnc @ scale 1: 0.01047 @ 7
    # Config(channels=[128, 256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # qoqbpw @ scale 1: 0.05480 @ 19
    # Config(channels=[16], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # hrmcrb @ scale 1: 
    Config(channels=[32], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # bpyvft @ scale 10: 0.00873 at 19 epochs
    # Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='res_first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # # tyflrq @ scale 10: 0.00898
    # Config(channels=[256, 512], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # # pruazu @ scale 10: 
    # Config(channels=[512], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # # cszdjj @ scale 10: 
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=16, ca_nheads=16),
]

# NOTE for loss l1_smooth NOTE
configs = [
    # ddhbna - 0.00456 @ 19, l1_smooth
    Config(channels=[64], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # apdhpj
    Config(channels=[128], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # chcyiu
    Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),
]

twiddles = itertools.product(
    configs,              # config
    # ["l2"],               # loss_type
    ["l1_smooth"],
    [1.0],
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

    args['in_chan'] = lat_chan
    args['in_size'] = lat_size
    args['clip_emblen'] = cfg.clip_emblen
    args['clip_scale_default'] = clip_scale_default
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)
