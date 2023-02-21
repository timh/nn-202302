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
        # if epoch == 10:
        #     print(f"epoch {epoch}")
        #     print(f"  {outputs[:5]=}")
        #     print(f"    {truth[:5]=}")
        #     print(f"         {loss=}")
        self.last_train_out = outputs
        self.last_train_truth = truth

    def on_val_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        # if epoch == 10:
        #     print(f"epoch {epoch}")
        #     print(f"  {outputs[:5]=}")
        #     print(f"    {truth[:5]=}")
        #     print(f"         {loss=}")
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

            model.simulate(net, val_dataloader, 1)

# %%
device = "cuda"
batch_size = 100
net_quotes_len = 20
epochs = 10000
net = model.make_net(net_quotes_len, 4, 50, device)

all_quotes = model.read_quotes("inap.20151216-5min.txt")

train_data, val_data = model.make_examples(all_quotes, net_quotes_len, 0.7, device=device)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

optim = torch.optim.AdamW(params=net.parameters(), lr=1e-4)
# loss_fn = nn.MSELoss().to(device)
# loss_fn = nn.L1Loss().to(device)
# loss_fn = trainer.MAPELoss
loss_fn = trainer.RPDLoss

t = DebugTrainer(net, loss_fn, optim)

t.train(epochs, train_dataloader, val_dataloader);

# %%
model.simulate(net, val_dataloader)

# %%
