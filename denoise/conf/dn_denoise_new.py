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

configs = [
    # time = res_last is best

    # tjschs - vloss 0.00476, tloss 0.00425 @ 19
    # Config(channels=[256], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # fwlaum - vloss 0.00487, tloss 0.00438 @ 19
    # Config(channels=[128], nstride1=2, time_pos='res_first', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # ctrycp - vloss 0.00479, tloss 0.00428 @ 19
    # Config(channels=[128], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),


    # picvvx - vloss 0.00479, tloss 0.00425 @ 19
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # hmknub - vloss 0.00492, tloss 0.00439 @ 9
    #          vloss 0.00466, tloss 0.00391 @ 49
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_first', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # psdcxy - vloss 0.00467, tloss 0.00412 @ 49
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # ccndsa - vloss 0.00461, tloss 0.00388 @ 49
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='last', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # nrbgis - vloss 0.00490, tloss 0.00423 @ 20
    Config(channels=[256], nstride1=2, time_pos='last', sa_pos='last', ca_pos='last', sa_nheads=8, ca_nheads=8),

    # Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_first', ca_pos='last', sa_nheads=8, ca_nheads=8),

]

positions = ['first', 'last', 'res_first', 'res_last']
configs = [
    # python train_denoise.py -d images.1star-2008-1024 -c conf/dn_denoise_new.py -vsc idsdex -n 5 -b 32 --startlr 1e-2 --endlr 1e-3 -e fastcompare

    Config(channels=[128], nstride1=2, time_pos=time_pos, sa_pos=sa_pos, ca_pos=ca_pos, sa_nheads=8, ca_nheads=8)
    for time_pos in positions
    for sa_pos in positions
    for ca_pos in positions
]


twiddles = itertools.product(
    configs,              # config
    # ["l2"],             # loss_type
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

