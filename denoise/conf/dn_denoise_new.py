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

@dataclass(kw_only=True)
class Config:
    channels: List[int]
    nstride1: int

    time_pos: denoise_new.EmbedPos
    sa_pos: denoise_new.EmbedPos

    # NOTE ca_pos = 'last' is the only choice that remotely works.
    # .. tested on channels=[128] and channels=[128, 256]
    ca_pos: denoise_new.EmbedPos = 'last'

    sa_nheads: int
    ca_nheads: int

# NOTE for various channel configs
#   [128, 256], [128]    nepochs=5    time_pos=res_last    sa_pos=res_first    - best settings

# [256, 1024] / sa_pos=res_last is best for fastcompare-res @ 4
# NOTE fastcompare-res
# positions = ['first', 'last', 'res_first', 'res_last']
# configs = [
#     # python train_denoise.py -d images.1star-2008-1024 -c conf/dn_denoise_new.py -vsc idsdex -n 5 -b 32 \
#     #   --startlr 1e-2 --endlr 1e-3 -e fastcompare
#     #   --startlr 1e-3 --endlr 1e-4 -e fastcompare_128_256
#     Config(channels=channels, nstride1=2, time_pos='res_last', sa_pos=sa_pos, sa_nheads=8, ca_nheads=8)
#     for channels in [
#         [64], [64, 128], [64, 128, 256], [64, 256],
#         [128], [128, 256], [128, 256, 512], [128, 512],
#         [256], [256, 512], [256, 1024], [256, 512, 1024],
#     ]
#     for sa_pos in ['res_first', 'res_last']
# ]

configs = [
    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=16),
    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=4),

    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=4, ca_nheads=8),
    Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=16, ca_nheads=8),
    Config(channels=[64], nstride1=3, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[64], nstride1=4, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),

    Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=4, ca_nheads=8),
    Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=16, ca_nheads=8),
    Config(channels=[256], nstride1=3, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[256], nstride1=4, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),

    Config(channels=[64, 256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[64, 256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=4, ca_nheads=8),
    Config(channels=[64, 256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=16, ca_nheads=8),
    Config(channels=[64, 256], nstride1=3, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
    Config(channels=[64, 256], nstride1=4, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),
]

twiddles = itertools.product(
    configs,              # config
    # ["l2", "l1", "l1_smooth"],             # loss_type
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

