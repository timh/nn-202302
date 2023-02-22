# %%
import io
import sys
import os
import datetime
from typing import List, Tuple, Literal, Callable
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

    fig_loss: Figure = None
    cur_learning_rate: float = 0.0

    def __init__(self, 
                 nets: List[nn.Module], loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 net_labels: List[str] = []):
        if isinstance(nets, list):
            self.nets = nets
        else:
            self.nets = [nets]
        
        self.optims = [torch.optim.AdamW(n.parameters()) for n in self.nets]
        
        if not len(net_labels):
            net_labels = ["" for _ in self.nets]
        self.net_labels = net_labels
        super().__init__(None, loss_fn, None)

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

            self.plot_train.add_data(self.net_idx, train_loss_hist[epoch:epoch+1])
            self.plot_val.add_data(self.net_idx, val_loss_hist[epoch:epoch+1])

            if self.net_idx == 0:
                self.plot_train.add_data(len(self.nets), learning_rates)
                self.plot_val.add_data(len(self.nets), learning_rates)

        if did_print:
            last_train_out = self.last_train_out[-1].item()
            last_train_truth = self.last_train_truth[-1].item()
            last_val_out = self.last_val_out[-1].item()
            last_val_truth = self.last_val_truth[-1].item()
            print(f"  train: out {last_train_out:.3f}, truth {last_train_truth:.3f}")
            print(f"    val: out {last_val_out:.3f}, truth {last_val_truth:.3f}")

            annotate = epoch == num_epochs - 1
            smoothing = 100
            self.plot_train.render(0.8, smoothing, annotate)
            self.plot_val.render(0.8, smoothing, annotate)

            num_preds, num_print = 0, 10
            actual_quotes, pred_quotes = model.simulate(self.net, val_dataloader, num_preds, num_print)
            if epoch == num_epochs - 1:
                if num_preds:
                    actual_quotes = actual_quotes[:num_preds]
                    pred_quotes = pred_quotes[:num_preds]
                actual_quotes = actual_quotes.detach().cpu()
                pred_quotes = pred_quotes.detach().cpu()
                self.axes2.clear()
                self.axes2.set_ylim(bottom=0, top=torch.max(actual_quotes) * 2.0)
                self.axes2.plot(actual_quotes, label="actual")
                self.axes2.plot(pred_quotes, label="pred")
                self.axes2.legend()
                self.fig_pred.savefig(f"outputs/trading-predictions-{self.net_labels[self.net_idx]}-{self.cur_learning_rate:.0E}.png")
            print()

            # display.display(self.fig_loss, self.fig_pred)

            self.fig_loss.savefig("outputs/trading-progress.png")

    def train_multi(self, 
                    learning_rates: List[Tuple[float, int]], 
                    train_dataloader: DataLoader, val_dataloader: DataLoader,
                    fig_loss: Figure, fig_pred: Figure):
        total_epochs = sum([pair[1] for pair in learning_rates])

        self.fig_loss = fig_loss
        train_labels = [f"train loss {l}" for l in self.net_labels] + ["learning rate"]
        val_labels = [f"val loss {l}" for l in self.net_labels] + ["learning rate"]
        self.plot_train = notebook.Plot(total_epochs, train_labels, self.fig_loss, nrows=2, idx=1)
        self.plot_val = notebook.Plot(total_epochs, val_labels, self.fig_loss, nrows=2, idx=2)

        self.fig_pred = fig_pred
        self.axes2 = fig_pred.add_subplot(1, 1, 1)

        for net_idx, (net, net_label, optim) in enumerate(zip(self.nets, self.net_labels, self.optims)):
            print(f"net #{net_idx} / {net_label}:")
            self.net = net
            self.net_idx = net_idx
            self.optim = optim

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

# all_quotes = model.read_quotes("inap.20151216-5min.txt")
# all_quotes = model.read_quotes("inap.20170106-daily.txt")
all_quotes = model.read_quotes("yf-data/MSFT-2023-01-23-30d-1m.csv")

batch_size = 100
net_quotes_len = 50
train_data, val_data = model.make_examples(all_quotes, net_quotes_len, 0.7, device=device)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

inches = 16
fig_loss = plt.figure(0, (inches, inches))
fig_pred = plt.figure(1, (inches, inches//2))

num_hidden = list(range(2, 10, 2))
hidden_size = [20, 60, 100]
nets = [model.make_net(net_quotes_len, nh, hs, device) for nh in num_hidden for hs in hidden_size]
net_labels = [f"num {nh}, size {hs}" for nh in num_hidden for hs in hidden_size]
# net = model.make_net(net_quotes_len, 6, 50, device)

lrs = [
    (1e-3,  500),
    (1e-4,  500),
    (5e-5, 1000),
    (1e-5, 2000),
    (5e-6, 2000),
    (1e-6, 2000),
]
t = DebugTrainer(nets, loss_fn, net_labels)
t.train_multi(lrs, train_dataloader, val_dataloader, fig_loss, fig_pred)

# %%
model.simulate(net, val_dataloader)

# %%
