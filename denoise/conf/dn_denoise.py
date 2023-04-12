from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, denoise

# 2023-04-11 13:12pm
#
# $ lsc -nc DenoiseModel -d -s tloss -f elapsed_str net_layers_str -a 'ago < 24h' 'loss_type = TYPE' -f net_sa_nheads=nheads net_norm_num_groups=ngroups
#
# l1_smooth
# code    net_class     saved (rel)  epoch  tloss    elapsed_str  net_layers_str     nheads  ngroups  net_latent_dim
# itfnkx  DenoiseModel  10h 40m 34s  1      0.21227  1m 6s        k3-s1-64x2-128x2   4       64       [128, 64, 64]
# xmfhui  DenoiseModel  10h 39m 56s  1      0.19028  1m 14s       k3-s1-64x2-128x2   8       64       [128, 64, 64]
# gnmzdf  DenoiseModel  10h 39m 21s  1      0.15291  1m 10s       k3-s1-128x4        8       128      [128, 64, 64]
# pbhyur  DenoiseModel  11h 59m 17s  3      0.08849  12m 16s      k3-s1-128x4        4       128      [128, 64, 64]
# vfebqv  DenoiseModel  9h 30m 35s   499    0.06769  2h 1m 33s    k3-s1-64x2         4       64       [64, 64, 64]
# hiiqra  DenoiseModel  20h 8m 25s   49     0.06637  20m 51s      k3-s1-128x2        4       128      [128, 64, 64]
# wvunvd  DenoiseModel  18h 22m 51s  19     0.05880  15m 35s      k3-s1-128x6        2       128      [128, 64, 64]
# amyiet  DenoiseModel  18h 38m 2s   19     0.05677  9m 56s       k3-s1-128x4        2       128      [128, 64, 64]
# rbjrue  DenoiseModel  18h 30m 55s  19     0.05112  12m 44s      k3-s1-128x6        4       128      [128, 64, 64]
# mqpbfp  DenoiseModel  19h 47m 2s   49     0.04801  22m 34s      k3-s1-128x2-64x2   4       64       [64, 64, 64]
# svwajg  DenoiseModel  19h 26m 19s  4      0.03993  2m 57s       k3-s1-256x2        4       256      [256, 64, 64]
# wiuqxb  DenoiseModel  17h 39m      19     0.03864  18m 37s      k3-s1-128x2-256x2  4       128      [256, 64, 64]
# qrltbp  DenoiseModel  18h 4m 37s   19     0.03720  12m 12s      k3-s1-128x6        8       128      [128, 64, 64]
# fqebag  DenoiseModel  19h 7m 41s   25     0.03089  30m 44s      k3-s1-256x4        4       256      [256, 64, 64]
# dwdcij  DenoiseModel  20h 4s       44     0.02650  26m 14s      k3-s1-128x3        4       128      [128, 64, 64]
# npzxed  DenoiseModel  2h 34m 2s    244    0.02454  3h 8m 55s    k3-s1-256x2        4       256      [256, 64, 64]
# legato  DenoiseModel  13h 47m 59s  99     0.01869  45m 21s      k3-s1-128x4        8       128      [128, 64, 64]
# ecpblm  DenoiseModel  10h 41m 51s  4      0.01829  4m 52s       k3-s1-256x2        8       256      [256, 64, 64]
# srzdaf  DenoiseModel  17h 49m 30s  19     0.01360  19m 45s      k3-s1-256x4        8       256      [256, 64, 64]
# bseuxb  DenoiseModel  13h 26m 50s  46     0.00693  53m 55s      k3-s1-128x2-256x2  8       128      [256, 64, 64]
# pjmmcs  DenoiseModel  8h 7m 57s    499    0.00630  2h 49m 4s    k3-s1-64x2         8       64       [64, 64, 64]
# vmttjv  DenoiseModel  16h 45m 19s  248    0.00599  1h 47m 6s    k3-s1-128x4        4       128      [128, 64, 64]
# ifdfxq  DenoiseModel  4h 32m 40s   499    0.00538  3h 48m 26s   k3-s1-128x2        8       128      [128, 64, 64]
# mrqylx  DenoiseModel  6h 27m 30s   499    0.00534  3h 35m 55s   k3-s1-128x2        4       128      [128, 64, 64]
#
# l2:
# code    net_class     saved (rel)  epoch  tloss    elapsed_str  net_layers_str  nheads  ngroups
# mtlyuv  DenoiseModel  1h 36m 14s   49     0.10095  22m 34s      k3-s1-128x2     4       128
# rlxprw  DenoiseModel  34m 16s      49     0.05860  25m 52s      k3-s1-128x4     4       128
# viakwa  DenoiseModel  1h 21m 46s   49     0.01466  28m 49s      k3-s1-128x2     8       128
# lmxtyi  DenoiseModel  17m 58s      49     0.01400  31m 31s      k3-s1-128x4     8       128
#
# l1:
# code    net_class     saved (rel)  epoch  tloss    elapsed_str  net_layers_str  nheads  ngroups
# tnxdns  DenoiseModel  8m 32s       49     0.48663  12m 46s      k3-s1-128x2     0       128
# tcxvnt  DenoiseModel  2h 3m 20s    49     0.19623  21m 22s      k3-s1-128x2     4       128
# cbxdmr  DenoiseModel  1h 5m 50s    49     0.19183  30m 29s      k3-s1-128x4     4       128
# bcmtzn  DenoiseModel  1h 48m 52s   49     0.09924  28m 52s      k3-s1-128x2     8       128
# newivh  DenoiseModel  50m 11s      49     0.09777  30m 20s      k3-s1-128x4     8       128

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net_ae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = denoise.DenoiseModel(**kwargs)
        # print(net)
        return net
    return fn

layers_str_list = [
    "k3-s1-128x2",
    "k3-s1-256x2",
    "k3-s1-512x2",
    # "k3-s1-128x4",
]
    
twiddles = itertools.product(
    layers_str_list,      # layers_str
    # ["l2"],               # loss_type
    ["l1"],               # loss_type
    [True],               # do_residual
    # [0, 4, 8],          # sa_nheads
    [8],                  # sa_nheads
    [(3, 1)]              # sa_kern_pad
)

for layers_str, loss_type, do_residual, sa_nheads, sa_kern_pad in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    sa_kernel_size, sa_padding = sa_kern_pad

    conv_cfg = conv_types.make_config(in_chan=lat_chan, in_size=lat_size, layers_str=layers_str,
                                      inner_nl_type='silu', inner_norm_type='group')
    args = dict(in_size=lat_size, in_chan=lat_chan, cfg=conv_cfg, do_residual=do_residual,
                sa_nheads=sa_nheads, sa_kernel_size=sa_kernel_size, sa_padding=sa_padding)
    print(f"ARGS: {args}")

    exp.label = ",".join([
        "denoise_dn",
        f"layers_{layers_str}"
    ])
    if do_residual:
        exp.label += ",residual"
    if sa_nheads:
        exp.label += f",sa_heads_{sa_nheads}"
    if sa_kernel_size:
        exp.label += f",sa_kern_{sa_kernel_size}"

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_ae(args)

    exps.append(exp)
