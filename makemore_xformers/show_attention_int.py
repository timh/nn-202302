# %%
import sys
import math
import matplotlib.pyplot as plt
import importlib
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "..")
import model_utils
import model
import text_experiment
import tokens

model_filename = "runs/wt2raw_5000-seqlen 256, wordlen 1, nhead 6, nlayers 4, emblen 96, hidlen 384, optim adamw, sched nanogpt-cosine, startlr 1.00E-03, endlr 1.00E-04, batch 128, minicnt 4, epochs 5000, flash pytorch, compile True, elapsed 580.59s, vloss 0.000.ckpt"
text_filename = "/home/tim/Downloads/wikitext-2-raw/wiki.train+valid.raw"

with open(model_filename, "rb") as file:
    state_dict = torch.load(file)

    # override attention type to get weights returned.
    state_dict["flash"] = "none"
    exp = text_experiment.load_experiment(state_dict, "cuda")

# %%

treader = tokens.WordTextReader(exp.seqlen, exp.wordlen, text_filename, True, "cuda")
dataloader = DataLoader(iter(treader), batch_size=1)
input_toks, truth = next(iter(dataloader))

_outputs, out_weights_all = exp.net(input_toks, return_weights=True)

# for i, out_weights in enumerate(out_weights_all):
#     print(f"{i}. {out_weights=}")

out_weights_all = [w for out_weights in out_weights_all for w in out_weights[0]]

# %%

out_weights_all = [w.detach().cpu() for w in out_weights_all]

out_weights_all = out_weights_all[:5]
labels = [treader.dictionary.token_to_vocab[tok.item()] for tok in input_toks[0]]
print(f"{len(labels)=}")
nrows = len(out_weights_all)
dim = 10
fig = plt.figure(0, figsize=(dim, dim * nrows))
axes = [fig.add_subplot(nrows, 1, i + 1) for i in range(len(out_weights_all))]
for i, w in enumerate(out_weights_all):
    # print(f"{w=}")
    ax = axes[i]
    ax.matshow(w)
    # ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
plt.show(fig)

# %%
plt.imshow(out_weights_all[2])