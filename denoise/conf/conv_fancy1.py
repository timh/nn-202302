import sys
import argparse
from typing import List, Dict, Union, Literal
import torch
from torch import Tensor, nn

sys.path.append("..")
import model
from model import ConvDesc, ConvDenoiser

from model_fancy import DenoiseFancy

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str

batch_size = 128
minicnt = 10

nhorizvert_values = [1, 2, 3]
do_layernorm_values = [
    # pre, post
    (False, False),
    (True, False),
    (False, True)
]
do_layernorm_post_top_values = [True, False]
do_relu_values = [True]

exp_descs: List[Dict[str, any]] = list()
for nhorizvert in nhorizvert_values:
    for do_layernorm_pre, do_layernorm_post in do_layernorm_values:
        for do_layernorm_post_top in do_layernorm_post_top_values:
            for do_relu in do_relu_values:
                label_list = [f"hv_{nhorizvert}"]
                if do_layernorm_pre:
                    label_list.append("lnpre")
                if do_layernorm_post:
                    label_list.append("lnpost")
                if do_layernorm_post_top:
                    label_list.append("lnposttop")
                if do_relu:
                    label_list.append("relu")
                label = "conv_fancy1_" + ",".join(label_list)
                net_fn = lambda: DenoiseFancy(image_size=cfg.image_size,
                                              nhoriz=nhorizvert, nvert=nhorizvert,
                                              do_layernorm_pre=do_layernorm_pre,
                                              do_layernorm_post=do_layernorm_post,
                                              do_layernorm_post_top=do_layernorm_post_top,
                                              do_relu=do_relu).to(device)
                exp_descs.append(dict(label=label, net_fn=net_fn))
