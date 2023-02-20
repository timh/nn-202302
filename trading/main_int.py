# %%
import io
import sys
import os
import datetime
from typing import List, Tuple, Literal
import importlib

import torch
import torch.optim
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython import display
from PIL import Image

sys.path.insert(0, "..")
import notebook
import trainer

import model

for m in notebook, trainer, model:
    importlib.reload(m)

# %%
device = "cuda"
num_batch = 100
window_len = 200
net_quotes = 50
epochs = 10000
realnet = model.make_net(net_quotes, 4, 50, device)
net = model.TradingModule(10000.0, realnet, device)

all_quotes = model.read_quotes("inap.20151216-5min.txt")
all_quotes_len = all_quotes.shape[-1]
quote_batches = torch.zeros((num_batch, window_len))

rand_start_pos = torch.randint(0, all_quotes_len - window_len, (num_batch,))
for i in range(num_batch):
    start = rand_start_pos[i]
    end = start + window_len
    quote_batches[i] = all_quotes[start:end]

optim = torch.optim.AdamW(params=net.parameters(), lr=1e-3)
print(f"{list(net.parameters())=}")
for i in range(epochs):
    output = net(quote_batches)
    net.zero_grad(set_to_none=True)
    loss = model.loss_fn(output, 0)
    loss.backward()
    optim.step()

    # print(f"{output=}")
    if i % 100 == 0:
        print(f"{i+1}/{epochs}: {loss:.3f}")

# %%
