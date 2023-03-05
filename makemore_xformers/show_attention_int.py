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

model_filename = "runs/wt2raw_5000-seqlen 128, wordlen 1, nhead 6, nlayers 4, emblen 192, hidlen 768, optim adamw, sched nanogpt-cosine, startlr 1.00E-03, endlr 5.00E-04, batch 128, minicnt 4, epochs 5000, flash pytorch, compile True, elapsed 428.07s, vloss 1.380.ckpt"
text_filename = "/home/tim/Downloads/wikitext-2-raw/wiki.valid.raw"

with open(model_filename, "rb") as file:
    state_dict = torch.load(file)

    # override attention type to get weights returned.
    state_dict["flash"] = "none"
    exp = text_experiment.load_experiment(state_dict, "cuda")

# %%

treader = tokens.WordTextReader(exp.seqlen, exp.wordlen, text_filename, True, "cuda")
dataloader = DataLoader(treader, batch_size=1, shuffle=True)
dataloader_it = iter(dataloader)

# %%

def format_fn(tick_val, tick_pos):
    tick_val = int(tick_val)
    if tick_val >= len(labels):
        return ""
    
    return labels[tick_val]

input_toks, truth = next(dataloader_it)
_outputs, out_weights_all = exp.net(input_toks, return_weights=True)

out_weights_all = [w[0] for w in out_weights_all] # get rid of batch dimension
out_weights_all = [F.softmax(w, dim=-1) for w in out_weights_all]

# mean_weights = torch.zeros_like(out_weights_all[0])
# for w in out_weights_all:
#     mean_weights.add_(w)
# mean_weights = mean_weights / len(out_weights_all)
# out_weights_all.append(mean_weights)

# just show the mean
# out_weights_all = out_weights_all[-1:]

out_weights_all = [w.detach().cpu() for w in out_weights_all]
# out_weights_all = out_weights_all[:5]
labels = [treader.dictionary.token_to_vocab[tok.item()] for tok in input_toks[0]]

max_side = 32
labels = labels[:max_side]
locater = MultipleLocator(base=1)

assert len(out_weights_all) == exp.nlayers
print(f"{out_weights_all[0].shape=}")
assert out_weights_all[0].shape == (exp.nhead, exp.seqlen, exp.seqlen)

nrows = exp.nlayers
ncols = exp.nhead

fig_size = 24
fig_width_inches = fig_size
fig_height_inches = fig_size * nrows / ncols
fig = plt.figure(0, figsize=(fig_width_inches, fig_height_inches))
axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(exp.nlayers * exp.nhead)]

for layer in range(exp.nlayers):
    for head in range(exp.nhead):
        i = layer * exp.nhead + head
        weights = out_weights_all[layer][head][:max_side, :max_side]

        ax = axes[i]
        ax.matshow(weights)
        ax.xaxis.set_major_formatter(format_fn)
        ax.xaxis.set_major_locator(locater)
        ax.yaxis.set_major_formatter(format_fn)
        ax.yaxis.set_major_locator(locater)
plt.show(fig)


# %%
