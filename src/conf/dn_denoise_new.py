from typing import List, Dict, Callable
import itertools
import argparse
from dataclasses import dataclass

from torch import nn

from nnexp.experiment import Experiment
from nnexp.denoise.models import vae, denoise_new

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

    time_pos: List[denoise_new.EmbedPos]
    sa_pos: List[denoise_new.EmbedPos]

    # NOTE ca_pos = 'last' is the only choice that remotely works.
    # .. tested on channels=[128] and channels=[128, 256]
    ca_pos: List[denoise_new.EmbedPos] # = ['last']
    ca_pos_conv: List[denoise_new.EmbedPos]
    ca_pos_lin: List[denoise_new.EmbedPos]

    sa_nheads: int
    ca_nheads: int

"""
main* $ lscd -a vae_shortcode=idsdex -nf image_dir unconditional_ratio net_in_dim net_latent_dim -a 'ago < 12h' 'nepochs = 99' -f nparams  elapsed_str                                                      [11:14:57]
code    saved (rel)  epoch  tloss    vloss    vepoch  chan            ns1  sa_hd  ca_hd  nparams   elapsed_str  net_ca_pos_lin
outask  59m 38s      99     0.02624  0.02044  63      [64, 256, 256]  4    2      0      15929736  1h 9m 52s    [res_last, up_res_last]
ifqops  5h 5m 11s    99     0.00519  0.00545  84      [64, 256]       4    2      0      8777992   1h 16m 22s   [res_last]
aakkbm  4h 37m 47s   99     0.00458  0.00441  96      [64, 256]       2    4      0      6923400   56m 28s      [res_last, up_res_last]
eyddci  3h 10m 50s   99     0.00516  0.00437  84      [64, 256, 256]  2    4      0      9875976   1h 15m 14s   [res_last]
btypjb  5h 57m 58s   99     0.00438  0.00435  97      [64, 256]       2    4      0      6267400   59m 50s      [res_last]
khublo  3h 53m 11s   99     0.00449  0.00432  73      [64, 256]       4    4      0      9433992   1h 5m 26s    [res_last, up_res_last]
lftwkc  7h 54m       99     0.00502  0.00431  69      [64, 256, 256]  4    4      0      14748936  1h 35m 11s   [res_last]
fuijgx  6h 30m 5s    99     0.00459  0.00421  92      [64, 64]        4    4      0      1328328   1h 3m 20s    [res_last, up_res_last]
ntbbcj  7h 29m 2s    99     0.00412  0.00416  98      [64, 64]        2    4      0      769480    46m 38s      [res_last]
txkcwc  5h 31m 5s    99     0.00396  0.00414  91      [64, 64]        4    2      0      1328328   1h 41s       [res_last, up_res_last]
wsrijy  2h 3m 39s    99     0.00422  0.00406  70      [64, 64]        2    2      0      1031880   44m 51s      [res_last, up_res_last]
etdhgf  2h 26m 4s    99     0.00520  0.00403  95      [64, 256]       4    4      0      8777992   1h 26m 40s   [res_last]
ogovrd  6h 57m 44s   99     0.00435  0.00401  29      [64, 256]       2    2      0      6923400   36m 23s      [res_last, up_res_last]
jctyfr  8h 50m 40s   99     0.00451  0.00378  62      [64, 64]        4    2      0      1065928   42m 33s      [res_last]
"""
choices: List[denoise_new.EmbedPos] = ['first', 'res_first', 'res_last', 'last',
                                       'up_first', 'up_res_first', 'up_res_last', 'up_last']
configs = [
    Config(channels=channels, nstride1=nstride1, 
           time_pos=time_pos, sa_pos=sa_pos, 
           ca_pos=ca_pos, ca_pos_conv=ca_pos_conv, ca_pos_lin=ca_pos_lin,
           sa_nheads=sa_nheads, ca_nheads=ca_nheads)
    for ca_pos in [ [] ]
    for ca_pos_conv in [ [] ]
    for ca_pos_lin in [
        # ['res_last', 'up_res_last'],
        # ['res_last'],
        []
    ]
    for sa_pos in [ ['up_res_last'] ]
    for time_pos in [ ['res_last', 'up_res_last' ] ]
    for channels in [ 
        # [64, 64], 
        # [32, 64, 128],
        [16, 16, 16, 16],
        # [256, 256],
    ]
    for sa_nheads in [2, 4]
    for ca_nheads in [0]
    for nstride1 in [2, 4]
]
import random
random.shuffle(configs)

twiddles = itertools.product(
    configs,              # config
    # ["l2", "l1", "l1_smooth"],             # loss_type
    # ["l1_smooth"],
    ["l2"],
    # [1.0, 10.0],
    [1.0],
)

for config, loss_type, clip_scale_default in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    args = dict(**config.__dict__)

    sa_pos_str = ".".join(config.sa_pos)
    ca_pos_str = ".".join(config.ca_pos)
    ca_pos_conv_str = ".".join(config.ca_pos_conv)
    ca_pos_lin_str = ".".join(config.ca_pos_lin)
    time_pos_str = ".".join(config.time_pos)
    label_parts = [
        "dn_new",
        "chan-" + "_".join(map(str, config.channels)),
        f"ns1-{config.nstride1}",
        f"time-{time_pos_str}",
        f"sa-{sa_pos_str}",
        f"sa_nh-{config.sa_nheads}"
    ]
    if config.ca_nheads:
        label_parts.append(f"ca_nh-{config.ca_nheads}")

    if ca_pos_str:
        label_parts.append(f"ca_pos-{ca_pos_str}")
    if ca_pos_conv_str:
        label_parts.append(f"ca_pos_conv-{ca_pos_conv_str}")
    if ca_pos_lin_str:
        label_parts.append(f"ca_pos_lin-{ca_pos_lin_str}")

    exp.label = ",".join(label_parts)

    args['in_chan'] = lat_chan
    args['in_size'] = lat_size
    args['clip_scale_default'] = clip_scale_default
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)

