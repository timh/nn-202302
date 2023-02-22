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
import experiment
from experiment import Experiment

import model

for m in notebook, trainer, model, experiment:
    importlib.reload(m)

class PredLogger(trainer.GraphLogger):
    fig_pred: Figure
    plot_pred: notebook.Plot
    num_predictions: int

    def __init__(self, exp_epochs: int, num_exps: int, num_predictions: int, fig_loss: Figure, fig_pred: Figure):
        super().__init__(exp_epochs, num_exps, fig_loss)

        self.fig_pred = fig_pred
        self.num_predictions = num_predictions

        blank_labels = [""] * num_exps
        self.plot_pred = \
            notebook.Plot(total_steps=num_predictions, fig=fig_pred, 
                          labels=blank_labels + ["real quotes"], alt_dataset=len(blank_labels),
                          yaxisscale="linear", alt_yaxisscale="linear",
                          yaxisfmt=".2f", alt_yaxisfmt=".2f")

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        self.plot_pred.labels[exp.exp_idx] = "pred " + exp.label
    
    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch(exp, exp_epoch, lr_epoch)

        last_train_out = exp.last_train_out[-1].item()
        last_train_truth = exp.last_train_truth[-1].item()
        last_val_out = exp.last_val_out[-1].item()
        last_val_truth = exp.last_val_truth[-1].item()

        print(f"  train: out {last_train_out:.3f}, truth {last_train_truth:.3f}")
        print(f"    val: out {last_val_out:.3f}, truth {last_val_truth:.3f}")

        self.fig_loss.savefig("outputs/trading-progress.png")

        if exp_epoch == exp.exp_epochs - 1:
            num_print = 5
            actual_quotes, pred_quotes = model.simulate(exp.net, exp.val_dataloader, self.num_predictions, num_print)
            actual_quotes = actual_quotes[:self.num_predictions]
            pred_quotes = pred_quotes[:self.num_predictions]

            # split into a few sections to get a few annotations.
            annotate_every = len(actual_quotes) // 5
            for start in range(0, len(actual_quotes), annotate_every):
                end = start + annotate_every
                self.plot_pred.add_data(exp.exp_idx, pred_quotes[start:end], True)
                if exp.exp_idx == 0:
                    self.plot_pred.add_data(self.num_exps, actual_quotes[start:end], True)
            
                self.plot_pred.render(0, 0)

            self.fig_pred.savefig(f"outputs/trading-predictions.png")

        print()

        display.display(self.fig_loss, self.fig_pred)

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

inches = 16
fig_loss = plt.figure(0, (inches, inches))
fig_pred = plt.figure(1, (inches, inches//2))

num_quotes = [20, 60]
num_hidden = [2, 6]
hidden_size = [20, 60]
lrs = [
    (1e-3,  500),
    (1e-4,  500),
    (5e-5, 1000),
    (1e-5, 2000),
    (5e-6, 2000),
    (1e-6, 2000),
]

# TODO for debugging
lrs = [(lrpair[0], lrpair[1] // 50) for lrpair in lrs]

exp_epochs = sum([lrpair[1] for lrpair in lrs])

get_optimizer_fn = lambda exp, lr: torch.optim.AdamW(exp.net.parameters(), lr=lr)

def experiments():
    for nquote in num_quotes:
        train_data, val_data = model.make_examples(all_quotes, nquote, 0.7, device=device)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        for nhide in num_hidden:
            for hsize in hidden_size:
                label = f"window {nquote}, num {nhide}, size {hsize}"
                net = model.make_net(nquote, nhide, hsize, device)

                exp = Experiment(label=label, net=net, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
                yield exp

num_experiments = len(num_quotes) * len(num_hidden) * len(hidden_size)
tcfg = trainer.TrainerConfig(lrs, get_optimizer_fn, num_experiments, experiments())

# %%
logger = PredLogger(exp_epochs, num_exps=num_experiments, num_predictions=100, fig_loss=fig_loss, fig_pred=fig_pred)
tr = trainer.Trainer(logger)
tr.train(tcfg)

# %%
model.simulate(net, val_dataloader)

# %%
