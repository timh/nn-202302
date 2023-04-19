from typing import List, Dict, Callable
import itertools
import argparse

from torch import nn

from experiment import Experiment
import conv_types
from models import vae, denoise2

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        net = denoise2.DenoiseModel2(**kwargs)
        # print(net)
        return net
    return fn

# unet..ish.
if False:
    layers_str_list.append(
        "64-"          # init_conv
        "t+64-64-"     # resnet 0
        "t+64-64-"     # resnet 1
        "sa8-"         # residual

        "256s2-"       # sequential 3-4: rearrange, conv2d -> (256, 32, 32)

        "t+256-256-"   # resnet 3-6
        "t+256-256-"   # resnet 3-7

        "512s2-"       # sequential 3-8: rearrange, conv2d -> (512, 16, 16)

        "t+512-512-"   # resnet 3-9
        "t+512-512-"   # resnet 3-10

        "1024"        # sequential 3-11: rearrange, conv2d -> (1024, 16, 16)
    )

# def layer_fn(parts: List[str], between: List[str])
layers_str_list = [
    # train at 1e-2 > 1e-3. if NaN, 1e-3 > 1e-4. 10 epochs.
    # "k3-sa8-256-t+256-ca8-256",    # vzyzqj - ~0.097

    # "k3-sa8-256-t+256-ca8",        # mgduhr - ~0.095
    "k3-sa8-128-t+128-ca8",          # delnzi - .103, nan
    "k3-sa8-128-t+256-ca8",          # onhdjw - .097, nan

    # "k3-sa8-256x2-t+256-256-ca8",  # ampfdi - ~0.099 (got nan)
    # "k3-sa8-256-t+256-256-ca8",    # fwfxvh - ~0.097 (got nan, twice)

    # "k3-sa8-512-t+512-ca8",        # pngfrt - .2362 (got nan)
    # "k3-sa8-512-t+512-ca16",       # hlxloe - .228 (nan)

    # "k3-sa8-64-mp2-t+128-ca8",     # ugrrrb - .100, very fast run, nan @ epoch=8
    # "k3-sa8-64-64s2-t+128-ca8",    # ctpnie - .100, nan @ epoch=8
    # "k3-sa8-64-mp2-t+256-ca8",     # pgyvdq - .245, nan right away
    # "k3-sa8-64-64s2-t+256-ca8",    # uylbkc - .104, nan epoch 4

]

# 

twiddles = itertools.product(
    layers_str_list,           # layers_str
    # ["l1_smooth", "l1", "l2"], # loss_type
    ["l2"], # loss_type
    # ["l1_smooth"],
    [True],                    # do_residual
    # [1.0, 10.0, 100.0],               # clip_scale_default
    [10.0],
)

for layers_str, loss_type, do_residual, clip_scale_default in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    conv_cfg = conv_types.make_config(in_chan=lat_chan, in_size=lat_size, layers_str=layers_str,
                                      inner_nl_type='silu', inner_norm_type='group')
    args = dict(in_size=lat_size, in_chan=lat_chan, cfg=conv_cfg, do_residual=do_residual, clip_scale_default=clip_scale_default)

    print(f"ARGS: {args}")

    exp.label = ",".join([
        "denoise_dn",
        f"layers_{conv_cfg.layers_str()}"
    ])
    if do_residual:
        exp.label += ",residual"

    if any([layer.ca_nheads for layer in conv_cfg.layers]):
        args['clip_emblen'] = cfg.clip_emblen
        exp.label += ",clip"
        exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    exps.append(exp)
