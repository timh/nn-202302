# %%
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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

def format_fn(tick_val, tick_pos):
    tick_val = int(tick_val)
    if tick_val >= len(labels):
        return ""
    
    return labels[tick_val]

out_weights_all = [w.detach().cpu() for w in out_weights_all]
# out_weights_all = out_weights_all[:5]
labels = [treader.dictionary.token_to_vocab[tok.item()] for tok in input_toks[0]]

max_side = 32
labels = labels[:max_side]
out_weights_plot = [out_weight[:max_side, :max_side] for out_weight in out_weights_all]
locater = MultipleLocator(base=1)

nrows = len(out_weights_plot)
fig_side = 10
fig = plt.figure(0, figsize=(fig_side, fig_side * nrows))
axes = [fig.add_subplot(nrows, 1, i + 1) for i in range(len(out_weights_plot))]
for i, w in enumerate(out_weights_plot):
    ax = axes[i]
    ax.matshow(w)
    ax.xaxis.set_major_formatter(format_fn)
    ax.xaxis.set_major_locator(locater)
    ax.yaxis.set_major_formatter(format_fn)
    ax.yaxis.set_major_locator(locater)
plt.show(fig)


# %%
