import random
from typing import Tuple, Callable, Sequence, List, Iterable
from dataclasses import dataclass
from collections import defaultdict
import datetime

import torch, torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import notebook
from experiment import Experiment

# mine
def DistanceLoss(out, truth):
    return torch.abs((truth - out)).mean()

# both are from:
#   https://stats.stackexchange.com/questions/438728/mean-absolute-percentage-error-returning-nan-in-pytorch

# mean absolute percentage error
# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
def MAPELoss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((target - output) / (target + 1e-6)))

# relative percentage difference
# https://en.wikipedia.org/wiki/Relative_change_and_difference
def RPDLoss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))    


@dataclass
class TrainerConfig:
    learning_rates: List[Tuple[float, int]]                                 # LR, num/epochs
    get_optimizer_fn: Callable[[Experiment, float], torch.optim.Optimizer]  # (Experiment, learning rate) -> optimizer

    num_experiments: int
    experiments: Iterable[Experiment]
    _exp_epochs: int = -1

    @property
    def exp_epochs(self) -> int:
        if self._exp_epochs == -1:
            self._exp_epochs = sum([lrpair[1] for lrpair in self.learning_rates])
        return self._exp_epochs

class TrainerLogger:
    def on_exp_start(self, exp: Experiment):
        pass

    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        pass

    def on_exp_end(self, exp: Experiment):
        pass

class Trainer:
    logger: TrainerLogger = None

    def __init__(self, logger: TrainerLogger = None):
        self.logger = logger

    # override this for new behavior after each epoch.
    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        now = datetime.datetime.now()
        if (now - exp.last_print) >= datetime.timedelta(seconds=5) or (lr_epoch == exp.lr_epochs - 1):
            timediff = (now - exp.last_print)

            samples_diff = float(exp.total_nsamples_sofar - exp.last_print_nsamples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(exp.total_batch_sofar - exp.last_print_batch)
            batch_per_sec = batch_diff / timediff.total_seconds()

            train_loss = exp.train_loss_hist[exp_epoch]
            val_loss = exp.val_loss_hist[exp_epoch]
            print(f"epoch {lr_epoch+1}/{exp.lr_epochs}: train loss {train_loss:.5f}, val loss {val_loss:.5f} | samp/sec {samples_per_sec:.3f} | batch/sec {batch_per_sec:.3f}")

            exp.last_print = now
            exp.last_print_nsamples = exp.total_nsamples_sofar
            exp.last_print_batch = exp.total_batch_sofar

            if self.logger is not None:
                self.logger.on_epoch(exp, exp_epoch, lr_epoch)

            return True

        return False
    
    def train(self, tcfg: TrainerConfig):
        for i, exp in enumerate(tcfg.experiments):
            exp.exp_epochs = tcfg.exp_epochs
            exp.train_loss_hist = torch.zeros((exp.exp_epochs,))
            exp.val_loss_hist = torch.zeros_like(exp.train_loss_hist)
            exp.exp_idx = i
            if self.logger is not None:
                self.logger.on_exp_start(exp)

            exp_epoch = 0
            for lridx, (lr, lr_epochs) in enumerate(tcfg.learning_rates):
                exp.last_print = datetime.datetime.now()
                exp.cur_lr = lr
                exp.lr_epochs = lr_epochs
                exp.optim = tcfg.get_optimizer_fn(exp, lr)

                print(f"train {lr_epochs} @ {lr:.0E}")
                for lr_epoch in range(lr_epochs):
                    stepres = exp.step(exp_epoch, lr_epoch)
                    if not stepres:
                        # something went wrong in that step. 
                        break

                    self.on_epoch(exp, exp_epoch, lr_epoch)
                    exp_epoch += 1
            
            if self.logger is not None:
                self.logger.on_exp_end(exp)

class GraphLogger(TrainerLogger):
    fig_loss: Figure
    plot_train: notebook.Plot
    plot_val: notebook.Plot
    num_exps: int

    last_exp_epoch = 0

    def __init__(self, exp_epochs: int, num_exps: int, fig_loss: Figure):
        self.num_exps = num_exps

        # initialize the plots with blank labels, we'll populate them with 
        # experiment names.
        blank_labels = [""] * num_exps

        self.fig_loss = fig_loss
        self.plot_train = notebook.Plot(exp_epochs, blank_labels + ["learning rate"], self.fig_loss, nrows=2, idx=1)
        self.plot_val = notebook.Plot(exp_epochs, blank_labels + ["learning rate"], self.fig_loss, nrows=2, idx=2)

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        self.last_exp_epoch = 0
        self.plot_train.labels[exp.exp_idx] = "train loss " + exp.label
        self.plot_val.labels[exp.exp_idx] = "val loss " + exp.label

    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        start = self.last_exp_epoch
        end = exp_epoch + 1
        # print(f"\033[1;31mGraphLogger: {exp.exp_idx=}  |  {exp_epoch=} {lr_epoch=}  |  {start=} {end=}  |  {exp.exp_epochs=} {exp.lr_epochs=}\033[0m")
        self.last_exp_epoch = exp_epoch + 1

        self.plot_train.add_data(exp.exp_idx, exp.train_loss_hist[start:end])
        self.plot_val.add_data(exp.exp_idx, exp.val_loss_hist[start:end])

        if exp.exp_idx == 0:
            learning_rates = torch.tensor([exp.cur_lr] * (end - start))
            self.plot_train.add_data(self.num_exps, learning_rates)
            self.plot_val.add_data(self.num_exps, learning_rates)

        self.plot_train.render(0.8, 100)
        self.plot_val.render(0.8, 100)

        annotate = (lr_epoch + 1) == exp.lr_epochs
        self.plot_train.render(0.8, 100, annotate)
        self.plot_val.render(0.8, 100, annotate)
