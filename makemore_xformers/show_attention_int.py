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

for m in [model_utils, mxt]:
    importlib.reload(m)

#model_filename = "runs/mm-ss4tut-sgd_20-seq_len 32, wordmaxlen 1, nhead 2, nlayers 2, hidden_len 32, emb_len 64, do_layernorm True, dropout 0.2, batch_size 4096, total_epochs 20, start_lr 5.0E-04, end_lr 5.0E-06, vocab_len 65.torch"
#model_filename = "runs/mm-ss4tut_100-seq_len 32, wordmaxlen 1, nhead 2, nlayers 4, hidden_len 32, emb_len 128, do_layernorm True, optim_type sgd, start_lr 5.0E-04, end_lr 5.0E-06, dropout 0.2, batch_size 4096, total_epochs 100, vocab_len 65.torch"
model_filename = "runs/mm-ss4tut-karpathy-lnorm_100-seqlen 256, wordlen 1, nhead 6, nlayers 6, hidlen 1024, emblen 384, norm True, optim adamw, startlr 1.0E-03, endlr 1.0E-04, compile False, batch 128, minicnt 2, epochs 5000, dropout 0.0, vocablen 65.torch"
text_filename = "shakespeare.txt"

model, textmap = mxt.load_model_and_textmap(model_filename, text_filename)

# %%
all_pairs = textmap.as_pairs()
dataloader = DataLoader(all_pairs, batch_size=1, shuffle=True)
dataloader_it = iter(dataloader)

# %%
input_toks, truth = next(dataloader_it)
print(f"{input_toks.shape=}")
input_embs = model.encoder(input_toks) * math.sqrt(model.emb_len)
input_embs = model.pos_encoder(input_embs)

enc_layer_list: List[nn.TransformerEncoderLayer] = model.transformer_encoder.layers
mh_attn_list: List[nn.MultiheadAttention] = [tel.self_attn for tel in enc_layer_list]

batch_first = mh_attn_list[0].batch_first

if not batch_first:
    input_embs = input_embs.transpose(0, 1)

out_attn_res = [
    mh(input_embs, input_embs, input_embs, need_weights=True)
     for mh in mh_attn_list
]
out_attns = [oar[0] for oar in out_attn_res]
out_attn_weights = [oar[1] for oar in out_attn_res]
if not batch_first:
    out_attns = [out_attn.transpose(0, 1) for out_attn in out_attns]
    # out_attn_weights = [out_attn_weight.transpose(0, 1) for out_attn_weight in out_attn_weights]

for i, (out_attn, out_attn_weight) in enumerate(zip(out_attns, out_attn_weights)):
    print(f"{i}. {out_attn.shape=} {out_attn_weight.shape=}")

# attn_scores = torch.cat(out_attns, dim=-2)
# attn_scores = F.softmax(attn_scores, dim=-1)
weights = [F.softmax(w, -1) for w in out_attn_weights]
weights = [w[0].detach().cpu() for w in weights]

fig = plt.figure(0, figsize=(20, 20))
labels = [textmap.token_to_vocab[tok.item()] for tok in input_toks[0]]
for i, w in enumerate(weights):
    axes = fig.add_subplot(len(weights), 1, i + 1)
    axes.matshow(w)
    axes.set_xticklabels(labels)
    axes.set_yticklabels(labels);

# %%