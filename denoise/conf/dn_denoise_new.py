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

"""
main* $ lsc -a 'extra_tag = cfg_fixed_really' 'loss_type = l1_smooth' 'nepochs > 10' --no_net_class -f best_val_loss=vloss best_val_epoch=vepoch  -d -s best_val_loss -f nparams net_channels=chan net_nstride1=nstride1 net_sa_nheads=sa_hd net_ca_nheads=ca_hd elapsed_str
code    saved (rel)  epoch  tloss    vloss    vepoch  nparams   chan       nstride1  sa_hd  ca_hd  elapsed_str  net_latent_dim
utanys  2h 56m 22s   19     0.00488  0.00561  18      14840968  [64, 256]  2         8      8      22m 34s      [256, 32, 32]
cwzzfb  2h 1m 3s     19     0.00520  0.00555  19      17351560  [64, 256]  4         8      8      31m 48s      [256, 32, 32]
rwvbty  2h 31m 42s   19     0.00487  0.00551  18      14840968  [64, 256]  2         16     8      26m 11s      [256, 32, 32]
jglall  2h 44m 59s   19     0.00493  0.00550  18      14840968  [64, 256]  2         4      8      21m 53s      [256, 32, 32]
symdvr  2h 17m 34s   19     0.00482  0.00547  19      16096264  [64, 256]  3         8      8      27m 52s      [256, 32, 32]
csfwjl  5h 2m 52s    19     0.00443  0.00515  11      33947080  [64]       2         16     8      14m 15s      [64, 64, 64]
hojxgk  4h 59m 2s    19     0.00442  0.00512  19      34021192  [64]       3         8      8      17m 7s       [64, 64, 64]
ivshii  4h 45m 31s   19     0.00444  0.00506  19      34095304  [64]       4         8      8      29m 29s      [64, 64, 64]
szctfq  4h 10m 51s   19     0.00436  0.00489  15      39299464  [256]      2         4      8      33m 5s       [256, 64, 64]
goujzf  3h 8m 5s     19     0.00433  0.00489  14      41661832  [256]      4         8      8      40m 6s       [256, 64, 64]
vnlwbs  3h 31m 46s   19     0.00432  0.00484  19      40480648  [256]      3         8      8      38m 58s      [256, 64, 64]
ivaozh  6h 41m 39s   99     0.00385  0.00482  47      67537736  [64]       2         8      16     54m 53s      [64, 64, 64]
itadue  4h 29m 19s   19     0.00424  0.00479  19      39299464  [256]      2         8      8      32m 14s      [256, 64, 64]
kxdlba  5h 7m 58s    99     0.00394  0.00477  57      33947080  [64]       2         4      8      56m 16s      [64, 64, 64]
iaymvm  5h 42m 14s   99     0.00394  0.00474  37      33947080  [64]       2         8      8      49m 3s       [64, 64, 64]
zxrhdm  3h 51m 51s   19     0.00424  0.00472  19      39299464  [256]      2         16     8      37m 47s      [256, 64, 64]
mmwfxb  6h 12m 14s   99     0.00406  0.00462  83      17152616  [64]       2         8      4      52m 15s      [64, 64, 64]
"""
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

