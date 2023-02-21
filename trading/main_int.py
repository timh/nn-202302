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

    fig1: Figure = None
    cur_learning_rate: float = 0.0

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

        if self.cur_learning_rate:
            learning_rates = torch.tensor([self.cur_learning_rate])

            self.plot_train.add_data(0, train_loss_hist[epoch:epoch+1])
            self.plot_train.add_data(1, learning_rates)

            self.plot_val.add_data(0, val_loss_hist[epoch:epoch+1])
            self.plot_val.add_data(1, learning_rates)

        if did_print:
            last_train_out = self.last_train_out[-1].item()
            last_train_truth = self.last_train_truth[-1].item()
            last_val_out = self.last_val_out[-1].item()
            last_val_truth = self.last_val_truth[-1].item()
            print(f"  train: out {last_train_out:.3f}, truth {last_train_truth:.3f}")
            print(f"    val: out {last_val_out:.3f}, truth {last_val_truth:.3f}")

            self.plot_train.render(0.8, 20, True)
            self.plot_val.render(0.8, 20, True)

            num_preds, num_print = 50, 10
            actual_quotes, pred_quotes = model.simulate(net, val_dataloader, num_preds)
            actual_quotes = actual_quotes[:num_preds].detach().cpu()
            pred_quotes = pred_quotes[:num_preds].detach().cpu()
            self.axes2.clear()
            self.axes2.set_ylim(bottom=0, top=torch.max(actual_quotes) * 2.0)
            self.axes2.plot(actual_quotes, label="actual")
            self.axes2.plot(pred_quotes, label="pred")
            self.axes2.legend()
            print()

            display.display(self.fig1, self.fig2)

            fig1.savefig("outputs/trading-progress.png")
            fig2.savefig("outputs/trading-predictions.png")

    def train_multi(self, 
                    learning_rates: List[Tuple[float, int]], 
                    train_dataloader: DataLoader, val_dataloader: DataLoader,
                    fig1: Figure, fig2: Figure):
        total_epochs = sum([pair[1] for pair in learning_rates])

        self.fig1 = fig1
        self.plot_train = notebook.Plot(total_epochs, ["train loss", "learning rate"], self.fig1, nrows=2, idx=1)
        self.plot_val = notebook.Plot(total_epochs, ["val loss", "learning rate"], self.fig1, nrows=2, idx=2)

        self.fig2 = fig2
        self.axes2 = fig2.add_subplot(1, 1, 1)

        for lr, epochs in learning_rates:
            print(f"{epochs} epochs at lr={lr:.0E}")
            self.cur_learning_rate = lr
            self.optim = torch.optim.AdamW(params=self.net.parameters(), lr=lr)
            self.train(epochs, train_dataloader, val_dataloader)

one_cent = torch.tensor(0.01)
def loss_fn(output: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    res = torch.abs(output - truth) * 100.0
    res = torch.mean(res)
    return res
   

# %%
device = "cuda"

all_quotes = model.read_quotes("inap.20151216-5min.txt")

batch_size = 100
net_quotes_len = 50
net = model.make_net(net_quotes_len, 6, 50, device)
train_data, val_data = model.make_examples(all_quotes, net_quotes_len, 0.7, device=device)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

fig1 = plt.figure(0, (10, 10))
fig2 = plt.figure(1, (10, 5))

lrs = [
    (1e-4, 2000),
    (5e-5, 2000),
    (1e-5, 2000),
    (5e-6, 2000),
    (1e-6, 2000),
]
optim = torch.optim.AdamW(params=net.parameters())
t = DebugTrainer(net, loss_fn, optim)
t.train_multi(lrs, train_dataloader, val_dataloader, fig1, fig2)

# %%
model.simulate(net, val_dataloader)

# %%
