# %%
import sys
import math
import matplotlib.pyplot as plt
import importlib
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "..")
import model_utils
import model_xformers_tutorial as mxt
import model_xformers_new as mxn

for m in [model_utils, mxn, mxt]:
    importlib.reload(m)

#model_filename = "runs/mm-ss4tut-sgd_20-seq_len 32, wordmaxlen 1, nhead 2, nlayers 2, hidden_len 32, emb_len 64, do_layernorm True, dropout 0.2, batch_size 4096, total_epochs 20, start_lr 5.0E-04, end_lr 5.0E-06, vocab_len 65.torch"
#model_filename = "runs/mm-ss4tut_100-seq_len 32, wordmaxlen 1, nhead 2, nlayers 4, hidden_len 32, emb_len 128, do_layernorm True, optim_type sgd, start_lr 5.0E-04, end_lr 5.0E-06, dropout 0.2, batch_size 4096, total_epochs 100, vocab_len 65.torch"
#model_filename = "runs/mm-ss4tut-karpathy-lnorm_100-seqlen 256, wordlen 1, nhead 6, nlayers 6, hidlen 1024, emblen 384, norm True, optim adamw, startlr 1.0E-03, endlr 1.0E-04, compile False, batch 128, minicnt 2, epochs 5000, dropout 0.0, vocablen 65.torch"
model_filename = "runs/mm-ss4tut-karpathy-v2-seqlen 256, wordlen 1, nhead 6, nlayers 6, hidlen 1024, emblen 384, norm True, optim sgd, startlr 1.0E-03, endlr 1.0E-04, compile False, batch 256, minicnt 1, epochs 5000, dropout 0.0, vocablen 65.torch"
text_filename = "shakespeare.txt"

model, textmap = mxt.load_model_and_textmap(model_filename, text_filename)
model: mxn.TransformerModel2 = model


# %%
for m in [model_utils, mxn, mxt]:
    importlib.reload(m)

all_pairs = textmap.as_pairs()
dataloader = DataLoader(all_pairs, batch_size=1, shuffle=True)
dataloader_it = iter(dataloader)

# %%
input_toks, truth = next(dataloader_it)
print(f"{input_toks.shape=}")

_outputs, out_weights_all = model(input_toks, return_weights=True)

for i, out_weights in enumerate(out_weights_all):
    print(f"{i}. {out_weights=}")

out_weights_all = [w for w in out_weights[0] for out_weights in out_weights_all]

# %%

# attn_scores = torch.cat(out_attns, dim=-2)
# attn_scores = F.softmax(attn_scores, dim=-1)
out_weights_all = [w.detach().cpu() for w in out_weights_all]

labels = [textmap.token_to_vocab[tok.item()] for tok in input_toks[0]]
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