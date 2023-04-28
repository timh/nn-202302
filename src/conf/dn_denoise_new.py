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
        import torchinfo
        import torch
        net = denoise_new.DenoiseModelNew(**kwargs)
        print(net)
        # input_data = (torch.zeros([1, *vae_latent_dim]).to("cuda"), 
        #               torch.zeros((1,)).to("cuda"),
        #               torch.zeros((1, 7, 768)).to("cuda"))
        # torchinfo.summary(model=net, input_data=input_data, depth=10, 
        #                   col_names=['input_size', 'output_size', 'num_params'],
        #                   row_settings=['depth', 'var_names'])
        return net
    return fn

@dataclass(kw_only=True)
class Config:
    channels: List[int]
    ngroups: int
    nstride1: int

    # sa_pos: List[denoise_new.EmbedPos]
    # sa_nheads: int

"""
main* $ lscd -a vae_shortcode=idsdex -nf image_dir unconditional_ratio net_in_dim net_latent_dim -a 'ago < 48h' 'nepochs = 99' -f nparams elapsed_str             [16:46:59]
code    saved (rel)  epoch  tloss    vloss    vepoch  chan            ns1  sa_hd  ca_hd  nparams   elapsed_str  net_ca_pos_lin
outask  6h 31m 44s   99     0.02624  0.02044  63      [64, 256, 256]  4    2      0      15929736  1h 9m 52s    [res_last, up_res_last]
ifqops  10h 37m 17s  99     0.00519  0.00545  84      [64, 256]       4    2      0      8777992   1h 16m 22s   [res_last]
aakkbm  10h 9m 53s   99     0.00458  0.00441  96      [64, 256]       2    4      0      6923400   56m 28s      [res_last, up_res_last]
eyddci  8h 42m 56s   99     0.00516  0.00437  84      [64, 256, 256]  2    4      0      9875976   1h 15m 14s   [res_last]
btypjb  11h 30m 4s   99     0.00438  0.00435  97      [64, 256]       2    4      0      6267400   59m 50s      [res_last]
khublo  9h 25m 17s   99     0.00449  0.00432  73      [64, 256]       4    4      0      9433992   1h 5m 26s    [res_last, up_res_last]
lftwkc  13h 26m 6s   99     0.00502  0.00431  69      [64, 256, 256]  4    4      0      14748936  1h 35m 11s   [res_last]
fuijgx  12h 2m 11s   99     0.00459  0.00421  92      [64, 64]        4    4      0      1328328   1h 3m 20s    [res_last, up_res_last]
ntbbcj  13h 1m 8s    99     0.00412  0.00416  98      [64, 64]        2    4      0      769480    46m 38s      [res_last]
txkcwc  11h 3m 11s   99     0.00396  0.00414  91      [64, 64]        4    2      0      1328328   1h 41s       [res_last, up_res_last]
wsrijy  7h 35m 45s   99     0.00422  0.00406  70      [64, 64]        2    2      0      1031880   44m 51s      [res_last, up_res_last]
etdhgf  7h 58m 10s   99     0.00520  0.00403  95      [64, 256]       4    4      0      8777992   1h 26m 40s   [res_last]
ogovrd  12h 29m 50s  99     0.00435  0.00401  29      [64, 256]       2    2      0      6923400   36m 23s      [res_last, up_res_last]
jctyfr  14h 22m 46s  99     0.00451  0.00378  62      [64, 64]        4    2      0      1065928   42m 33s      [res_last]
"""
choices: List[denoise_new.EmbedPos] = ['first', 'res_first', 'res_last', 'last',
                                       'up_first', 'up_res_first', 'up_res_last', 'up_last']
configs = [
    Config(channels=[64], nstride1=2, ngroups=2),
]
import random
# random.shuffle(configs)

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

    label_parts = [
        "dn_new",
        "chan-" + "_".join(map(str, config.channels)),
        f"ngroups-{config.ngroups}",
        f"nstride1-{config.nstride1}",
    ]

    exp.label = ",".join(label_parts)

    args['in_chan'] = lat_chan
    args['in_size'] = lat_size
    args['clip_scale_default'] = clip_scale_default
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)

