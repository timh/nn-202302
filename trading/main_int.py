# %%
import io
import sys
import os
import datetime
from typing import List, Tuple, Literal
import importlib

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
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

class DebugTrainer(trainer.Trainer):
    last_train_out: torch.Tensor
    last_train_truth: torch.Tensor

    def on_train_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        self.last_train_in = inputs
        self.last_train_out = outputs
        self.last_train_truth = truth

    def on_val_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        self.last_val_in = inputs
        self.last_val_out = outputs
        self.last_val_truth = truth

    def on_epoch(self, num_epochs: int, epoch: int, train_loss_hist: torch.Tensor, val_loss_hist: torch.Tensor):
        did_print = super().on_epoch(num_epochs, epoch, train_loss_hist, val_loss_hist)
        if did_print:
            last_train_out = self.last_train_out[-1].item()
            last_train_truth = self.last_train_truth[-1].item()
            last_val_out = self.last_val_out[-1].item()
            last_val_truth = self.last_val_truth[-1].item()
            print(f"  train: out {last_train_out:.3f}, truth {last_train_truth:.3f}")
            print(f"    val: out {last_val_out:.3f}, truth {last_val_truth:.3f}")

            model.simulate(net, val_dataloader, 5)
            print()
    

# %%
device = "cuda"
batch_size = 100
net_quotes_len = 50
net = model.make_net(net_quotes_len, 4, 50, device)

all_quotes = model.read_quotes("inap.20151216-5min.txt")

train_data, val_data = model.make_examples(all_quotes, net_quotes_len, 0.7, device=device)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# loss_fn = nn.MSELoss().to(device)
# loss_fn = nn.L1Loss().to(device)
# loss_fn = trainer.MAPELoss
# loss_fn = trainer.RPDLoss
one_cent = torch.tensor(0.01)
def loss_fn(output: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    # truth_diff = truth[-1] - truth[-2]
    # out_diff = output[-1] - output[-2]
    # return torch.mean(torch.abs(out_diff - truth_diff) / truth[-1])
    # res = torch.abs(output - truth) / truth
    # res = (torch.abs(output - truth) ** 2) / truth
    # res = res ** 2
    res = torch.abs(output - truth) * 100.0
    res = torch.mean(res)
    return res

lrs = [
    (1e-4, 1000),
    (3e-5, 1000),
    (1e-5, 1000),
    (5e-6, 1000),
    (1e-6, 1000),
]
optim = torch.optim.AdamW(params=net.parameters(), lr=3e-5)
t = DebugTrainer(net, loss_fn, optim)
for lr, epochs in lrs:
    print(f"{epochs} epochs at lr={lr:.1E}")
    optim = torch.optim.AdamW(params=net.parameters(), lr=lr)
    t.optim = optim
    t.train(epochs, train_dataloader, val_dataloader);

# %%
model.simulate(net, val_dataloader)

# %%
