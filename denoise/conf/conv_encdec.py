# %%
import sys
import argparse
import importlib

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

batch_size = 128
minicnt = 10

emblen_values = [64, 128, 256, 512]

exp_descs: List[Dict[str, any]] = list()
for emblen in emblen_values:
    net_fn = lambda: nn.Sequential(
        model_encdec.Encoder(image_size=cfg.image_size, emblen=emblen),
        model_encdec.Decoder(image_size=cfg.image_size, emblen=emblen)
    ).to(device)
    ed = dict(net_fn=net_fn, label=f"conv_encdec--emblen_{emblen:03}")
    exp_descs.append(ed)


if __name__ == "__main__":
    encoder = model_encdec.Encoder(image_size=cfg.image_size, emblen=emblen)
    decoder = model_encdec.Decoder(image_size=cfg.image_size, emblen=emblen)

    input = torch.randn((1, 3, 128, 128))
    print(f"{input.shape=}")

    enc_out = encoder(input)
    print(f"{enc_out.shape=}")

    out = decoder(enc_out)
    print(f"{out.shape=}")
# %%


