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
    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch(exp, exp_epoch, lr_epoch)

        last_train_out = exp.last_train_out[-1].item()
        last_train_truth = exp.last_train_truth[-1].item()
        last_val_out = exp.last_val_out[-1].item()
        last_val_truth = exp.last_val_truth[-1].item()

        print(f"  train: out {last_train_out:.3f}, truth {last_train_truth:.3f}")
        print(f"    val: out {last_val_out:.3f}, truth {last_val_truth:.3f}")

        self.fig_loss.savefig("outputs/trading-progress.png")

        # num_preds, num_print = 0, 10
        # actual_quotes, pred_quotes = model.simulate(self.net, val_dataloader, num_preds, num_print)
        # if epoch == num_epochs - 1:
        #     if num_preds:
        #         actual_quotes = actual_quotes[:num_preds]
        #         pred_quotes = pred_quotes[:num_preds]
        #     actual_quotes = actual_quotes.detach().cpu()
        #     pred_quotes = pred_quotes.detach().cpu()
        #     self.axes2.clear()
        #     self.axes2.set_ylim(bottom=0, top=torch.max(actual_quotes) * 2.0)
        #     self.axes2.plot(actual_quotes, label="actual")
        #     self.axes2.plot(pred_quotes, label="pred")
        #     self.axes2.legend()
        #     self.fig_pred.savefig(f"outputs/trading-predictions-{self.net_labels[self.net_idx]}-{self.cur_learning_rate:.0E}.png")
        # print()

        # display.display(self.fig_loss, self.fig_pred)

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
exp_epochs = sum([lrpair[1] for lrpair in lrs])

optim_fn = lambda exp, lr: torch.optim.AdamW(exp.net.parameters(), lr=lr)

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
tcfg = trainer.TrainerConfig(learning_rates=lrs, 
                             get_optimizer_fn=optim_fn, 
                             num_experiments=num_experiments, 
                             experiments=experiments())

logger = PredLogger(exp_epochs, num_exps=num_experiments, fig_loss=fig_loss)
tr = trainer.Trainer(logger)
tr.train(tcfg)

# %%
model.simulate(net, val_dataloader)

# %%
