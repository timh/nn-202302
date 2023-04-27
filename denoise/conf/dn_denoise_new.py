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
$ lscd -a vae_shortcode=idsdex -nf image_dir unconditional_ratio net_in_dim net_latent_dim                                                                                       [22:07:13]
code    saved (rel)  epoch  tloss    vloss    vepoch  chan   ns1  sa_hd  ca_hd  net_time_pos                 net_ca_pos                                      net_sa_pos
mcwvrk  2h 31m 1s    19     0.01022  0.00857  2       [256]  2    4      4      ['res_last', 'up_res_last']  ['last']                                        ['last', 'up_first', 'up_last']
jhfgoc  3h 20m 10s   1      0.00646  0.00610  1       [256]  2    4      4      ['res_last', 'up_res_last']  ['last']                                        ['up_first']
tvajkf  6m 9s        2      0.01024  0.00594  1       [256]  2    4      4      ['res_last']                 ['res_last', 'last', 'up_res_last', 'up_last']  ['up_first', 'up_last']
pfyrvc  35m 38s      19     0.11801  0.00564  2       [256]  2    4      4      ['res_last', 'up_res_last']  ['up_res_last', 'up_last']                      ['last', 'up_first', 'up_last']
rspvxt  1h 36m 56s   8      0.00701  0.00557  8       [256]  2    4      4      ['res_last', 'up_res_last']  ['up_res_last', 'up_last']                      ['up_first', 'up_last']
zhtdxq  1h 13s       12     0.14834  0.00526  4       [256]  2    4      4      ['res_last']                 ['up_res_last', 'up_last']                      ['up_first', 'up_last']
geaslg  1h 46m 2s    18     0.17969  0.00525  4       [256]  2    4      4      ['res_last']                 ['last']                                        ['last', 'up_first', 'up_last']
mwqrfw  2h 14m 24s   19     0.00535  0.00478  9       [256]  2    4      4      ['res_last']                 ['last']                                        ['up_first', 'up_last']
rnmteg  49s          21     0.00437  0.00439  18      [256]  2    4      4      ['res_last']                 ['res_last', 'last', 'up_res_last', 'up_last']  ['up_first']
ssrraj  3h 13s       19     0.00482  0.00436  12      [256]  2    4      4      ['res_last', 'up_res_last']  ['res_last', 'last', 'up_res_last', 'up_last']  ['up_first']
"""
choices: List[denoise_new.EmbedPos] = ['first', 'res_first', 'res_last', 'last',
                                       'up_first', 'up_res_first', 'up_res_last', 'up_last']
configs = [
    Config(channels=channels, nstride1=nstride1, 
           time_pos=time_pos, sa_pos=sa_pos, 
           ca_pos=ca_pos, ca_pos_conv=ca_pos_conv, ca_pos_lin=ca_pos_lin,
           sa_nheads=sa_nheads, ca_nheads=ca_nheads)
    # for ca_pos in ['up_first', 'up_last', 'up_res_first', 'up_res_last', 'last']
    # for ca_pos in [ [], ['last'], ['res_last', 'last', 'up_res_last', 'up_last'], ['up_res_last', 'up_last'] ]
    for ca_pos in [ [] ]
    # for ca_pos_conv in [ ['last'], ['res_last', 'last', 'up_res_last', 'up_last'], ['up_res_last', 'up_last'] ]
    for ca_pos_conv in [ 
        # ['first'], ['res_first'], ['res_last'], ['last'], ['up_first'], ['up_res_first'], ['up_res_last'], ['up_last']
        # ['res_first'], ['up_first'], ['up_res_first'], ['up_res_last'], ['up_last']
        # ['up_res_first'], 
        [],
        # ['res_last', 'last', 'up_res_last', 'up_last'], 
        # ['up_res_last', 'up_last']
    ]
    for ca_pos_lin in [
        ['res_last', 'up_res_last']
    ]
    # for sa_pos in ['res_first']
    # for sa_pos in [ ['up_first'], ['up_first', 'up_last'], ['last', 'up_first', 'up_last'] ]
    for sa_pos in [ ['up_first'] ]
    # for time_pos in [ ['res_last'], ['res_last', 'up_res_last'] ]
    for time_pos in [ ['res_last'] ]
    for channels in [ 
        # [64], 
        [64, 64], 
        [64, 64, 64], 
        # [256], 
        [256, 256],
        # [256, 256, 256]
    ]
    for sa_nheads in [2, 4]
    for ca_nheads in [0]
    for nstride1 in [2, 4]
    # for nstride1 in [1, 2, 4]
    # for sa_nheads in [4, 8, 16]
    # for ca_nheads in [4, 8, 16]
    # for channels in [ [64] ]
    # for sa_pos in ['up_first', 'up_last', 'up_res_first', 'up_res_last', 'last']
    # for time_pos in ['up_first', 'up_last', 'up_res_first', 'up_res_last', 'last']
]
import random
random.shuffle(configs)

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

    sa_pos_str = ".".join(config.sa_pos)
    ca_pos_str = ".".join(config.ca_pos)
    ca_pos_conv_str = ".".join(config.ca_pos_conv)
    ca_pos_lin_str = ".".join(config.ca_pos_lin)
    time_pos_str = ".".join(config.time_pos)
    label_parts = [
        "denoise_dnnew",
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
    args['clip_emblen'] = cfg.clip_emblen
    args['clip_scale_default'] = clip_scale_default
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)

