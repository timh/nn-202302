# %%
import sys
import argparse
import importlib
import random

from typing import List, Dict
import torch
from torch import Tensor, nn

sys.path.append("..")
sys.path.append("../..")
import model_encdec

importlib.reload(model_encdec)

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str

batch_size = 128 * 2
minicnt = 1

emblen_values = [64, 128, 256, 512]
hidlen_values = [128, 256]
nlevels_values = [2, 3, 4]
startlevel_values = [4, 8, 16]
# sched_type_values = ["nanogpt", "constant"]
sched_type_values = ["constant"]

exp_descs: List[Dict[str, any]] = list()
for emblen in emblen_values:
    for hidlen in hidlen_values:
        for nlevels in nlevels_values:
            for startlevel in startlevel_values:
                for sched_type in sched_type_values:
                    net_fn = lambda: nn.Sequential(
                        model_encdec.Encoder(image_size=cfg.image_size, emblen=emblen, hidlen=hidlen, nlevels=nlevels, startlevel=startlevel),
                        model_encdec.Decoder(image_size=cfg.image_size, emblen=emblen, hidlen=hidlen, nlevels=nlevels, startlevel=startlevel)
                    ).to(device)
                    ed = dict(net_fn=net_fn, sched_type=sched_type,
                              label=f"conv_encdec--emblen_{emblen:03},hidlen_{hidlen:03},nlvl_{nlevels},slvl_{startlevel:02},sched_{sched_type}")
                    exp_descs.append(ed)

random.shuffle(exp_descs)

if False:
    image_size = 128
    encoder = model_encdec.Encoder(image_size=image_size, emblen=emblen, hidlen=128)
    decoder = model_encdec.Decoder(image_size=image_size, emblen=emblen, hidlen=128)

    input = torch.randn((1, 3, 128, 128))
    print(f"{input.shape=}")

    enc_out = encoder(input)
    print(f"{enc_out.shape=}")

    out = decoder(enc_out)
    print(f"{out.shape=}")
# %%


