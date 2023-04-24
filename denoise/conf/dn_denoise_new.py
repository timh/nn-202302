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

"""
2022/04/22 11:01
main* $ lscd -a 'extra_tag = cfg_fixed_really' 'loss_type = l1_smooth' 'nepochs > 10' -nf net_latent_dim                                                  [10:57:59]
code    saved (rel)  epoch  tloss    vloss    vepoch  nparams   chan       nstride1  sa_hd  ca_hd  elapsed_str
utanys  4h 40m 44s   19     0.00488  0.00561  18      14840968  [64, 256]  2         8      8      22m 34s
cwzzfb  3h 45m 25s   19     0.00520  0.00555  19      17351560  [64, 256]  4         8      8      31m 48s
rwvbty  4h 16m 4s    19     0.00487  0.00551  18      14840968  [64, 256]  2         16     8      26m 11s
jglall  4h 29m 21s   19     0.00493  0.00550  18      14840968  [64, 256]  2         4      8      21m 53s
symdvr  4h 1m 56s    19     0.00482  0.00547  19      16096264  [64, 256]  3         8      8      27m 52s
csfwjl  6h 47m 14s   19     0.00443  0.00515  11      33947080  [64]       2         16     8      14m 15s
hojxgk  6h 43m 24s   19     0.00442  0.00512  19      34021192  [64]       3         8      8      17m 7s
ivshii  6h 29m 53s   19     0.00444  0.00506  19      34095304  [64]       4         8      8      29m 29s
szctfq  5h 55m 13s   19     0.00436  0.00489  15      39299464  [256]      2         4      8      33m 5s
goujzf  4h 52m 27s   19     0.00433  0.00489  14      41661832  [256]      4         8      8      40m 6s
vnlwbs  5h 16m 8s    19     0.00432  0.00484  19      40480648  [256]      3         8      8      38m 58s
ivaozh  8h 26m 1s    99     0.00385  0.00482  47      67537736  [64]       2         8      16     54m 53s
itadue  6h 13m 41s   19     0.00424  0.00479  19      39299464  [256]      2         8      8      32m 14s
kxdlba  6h 52m 20s   99     0.00394  0.00477  57      33947080  [64]       2         4      8      56m 16s
tzfdtd  6m 11s       19     0.00425  0.00476  13      22505000  [256]      2         16     4      26m 58s
bvbmrz  1m 19s       19     0.00425  0.00476  18      22505000  [256]      2         8      4      28m 24s
iaymvm  7h 26m 36s   99     0.00394  0.00474  37      33947080  [64]       2         8      8      49m 3s
zxrhdm  5h 36m 13s   19     0.00424  0.00472  19      39299464  [256]      2         16     8      37m 47s
mmwfxb  7h 56m 36s   99     0.00406  0.00462  83      17152616  [64]       2         8      4      52m 15s
"""
configs = [
    # Config(channels=[64], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=4),
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=4),
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=8, ca_nheads=8),

    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=16, ca_nheads=8),
    # Config(channels=[256], nstride1=2, time_pos='res_last', sa_pos='res_last', sa_nheads=16, ca_nheads=4),
]
    
configs = [
    Config(channels=[128], nstride1=2, 
           time_pos='res_last', sa_pos='res_last', ca_pos=ca_pos,
           sa_nheads=8, ca_nheads=4)
    for ca_pos in ['up_first', 'up_last', 'up_res_first', 'up_res_last',
                   'last']
]

twiddles = itertools.product(
    configs,              # config
    # ["l2", "l1", "l1_smooth"],             # loss_type
    ["l1_smooth"],
    # [1.0, 10.0],
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

