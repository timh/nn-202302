from typing import List, Dict, Callable
import itertools
import argparse
from dataclasses import dataclass

from torch import nn

from nnexp.experiment import Experiment
from nnexp.denoise.models import vae, unet_diffusers

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec
cfg: argparse.Namespace

def lazy_net(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        import torchinfo
        import torch
        net = unet_diffusers.UnetDiffusers(**kwargs)
        print(net)
        input_data = (torch.zeros([1, *vae_latent_dim]).to("cuda"), 
                      torch.zeros((1,)).to("cuda"),
                      torch.zeros((1, 77, 768)).to("cuda"))
        torchinfo.summary(model=net, input_data=input_data, depth=10, 
                          col_names=['input_size', 'output_size', 'num_params'],
                          row_settings=['depth', 'var_names'])
        return net
    return fn

twiddles = itertools.product(
    # ["l2", "l1", "l1_smooth"],             # loss_type
    ["l1_smooth"],
    # [1.0, 10.0],
    [1.0],
)

for loss_type, clip_scale_default in twiddles:
    exp = Experiment()
    vae_latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = vae_latent_dim

    args = dict()

    label_parts = [
        "dn_unet_diffusers",
    ]

    exp.label = ",".join(label_parts)

    # unet_diffusers.UnetDiffusers(sample_size, in_channels, out_channels, layers_per_block, cross_attention_dim=1280, attention_head_dim=8)
    args['sample_size'] = lat_size
    args['in_channels'] = lat_chan
    args['out_channels'] = lat_chan
    exp.use_clip = True

    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net(args)

    print(f"ARGS: {args}")
    exps.append(exp)
