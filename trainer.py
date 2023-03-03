import random
from typing import Tuple, Callable, Sequence, List, Iterable, Dict
from dataclasses import dataclass
from collections import defaultdict
import datetime
import math

import torch, torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.tensorboard as tboard
from accelerate import Accelerator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from IPython import display

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
    experiments: Iterable[Experiment]
    get_optimizer_fn: \
        Callable[[Experiment], 
                 Tuple[torch.optim.Optimizer, 
                       torch.optim.lr_scheduler._LRScheduler]]
    accel: Accelerator = None

class TrainerLogger:
    def on_exp_start(self, exp: Experiment):
        exp.on_start()

    def on_exp_end(self, exp: Experiment):
        exp.on_end()

    def on_epoch_end(self, exp: Experiment, epoch: int):
        pass

    def on_epoch_end_infrequent(self, exp: Experiment, epoch: int):
        pass

class Trainer:
    logger: TrainerLogger = None

    def __init__(self, logger: TrainerLogger = None):
        self.logger = logger

    def on_exp_start(self, exp: Experiment):
        if self.logger is not None:
            self.logger.on_exp_start(exp)

    def on_exp_end(self, exp: Experiment):
        if self.logger is not None:
            self.logger.on_exp_end(exp)

    # override this for new behavior after each epoch.
    def on_epoch_end(self, exp: Experiment, epoch: int):
        if self.logger is not None:
            self.logger.on_epoch_end(exp, epoch)
        
        now = datetime.datetime.now()
        if (now - exp.last_print) >= datetime.timedelta(seconds=10) or epoch == exp.epochs - 1:
            timediff = (now - exp.last_print)

            samples_diff = float(exp.total_nsamples_sofar - exp.last_print_nsamples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(exp.total_batch_sofar - exp.last_print_batch)
            batch_per_sec = batch_diff / timediff.total_seconds()
            epoch_diff = float(epoch - exp.last_print_epoch)
            epoch_per_sec = epoch_diff / timediff.total_seconds()
            if not epoch_per_sec:
                epoch_per_sec = 1
            eta_exp_done_sec = int((exp.epochs - epoch + 1) / epoch_per_sec)
            eta_exp_done_min = eta_exp_done_sec // 60
            eta_exp_done_sec -= eta_exp_done_min * 60

            train_loss = exp.train_loss_hist[epoch]
            val_loss = exp.val_loss_hist[epoch]
            print()
            print(f"epoch {epoch+1}/{exp.epochs}: tloss {train_loss:.5f}, vloss {val_loss:.5f} | samp/s {samples_per_sec:.3f} | epoch/sec {epoch_per_sec:.3f} | eta {eta_exp_done_min}m{eta_exp_done_sec:02}s")

            exp.last_print = now
            exp.last_print_nsamples = exp.total_nsamples_sofar
            exp.last_print_batch = exp.total_batch_sofar
            exp.last_print_epoch = epoch

            if self.logger is not None:
                self.logger.on_epoch_end_infrequent(exp, epoch)

            return True

        return False
    
    def train(self, tcfg: TrainerConfig):
        for exp_idx, exp in enumerate(tcfg.experiments):
            exp.exp_idx = exp_idx
            exp.train_loss_hist = torch.zeros((exp.epochs,))
            exp.val_loss_hist = torch.zeros_like(exp.train_loss_hist)

            exp.last_print = datetime.datetime.now()
            exp.optim, exp.scheduler = tcfg.get_optimizer_fn(exp)
            exp.cur_lr = exp.scheduler.get_lr()[0]
            exp.last_print_nsamples = 0
            exp.last_print_batch = 0
            exp.last_print_epoch = 0

            self.on_exp_start(exp)

            print(f"train #{exp_idx} {exp.label}")
            for epoch in range(exp.epochs):
                stepres = exp.train_epoch(epoch, tcfg.accel)
                if not stepres:
                    # something went wrong in that step. 
                    break

                self.on_epoch_end(exp, epoch)

            self.on_exp_end(exp)

# TODO: this needs a bunch of updates.
class GraphLogger(TrainerLogger):
    fig_loss: Figure
    plot_train: notebook.Plot
    plot_val: notebook.Plot
    num_exps: int
    do_display: bool

    last_exp_epoch = 0

    def __init__(self, exp_epochs: int, num_exps: int, fig_loss: Figure, 
                 do_display = False):
        self.num_exps = num_exps
        self.do_display = do_display

        # initialize the plots with blank labels, we'll populate them with 
        # experiment names.
        blank_labels = [""] * num_exps

        self.fig_loss = fig_loss
        kwargs = dict(
            total_steps=exp_epochs,
            fig=self.fig_loss,
            nrows=2,
            alt_dataset=len(blank_labels),
            alt_yaxisfmt=".1E"
        )
        self.plot_train = notebook.Plot(labels=blank_labels + ["learning rate"], idx=1, **kwargs)
        self.plot_val = notebook.Plot(labels=blank_labels + ["learning rate"], idx=2, **kwargs)

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        self.last_exp_epoch = 0
        self.plot_train.labels[exp.exp_idx] = "train loss " + exp.label
        self.plot_val.labels[exp.exp_idx] = "val loss " + exp.label

    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        start = self.last_exp_epoch
        end = exp_epoch + 1
        # print(f"\033[1;31mGraphLogger: {exp.exp_idx=}  |  {exp_epoch=} {lr_epoch=}  |  {start=} {end=}  |  {exp.exp_epochs=} {exp.lr_epochs=}\033[0m")
        self.last_exp_epoch = exp_epoch + 1

        annotate = (lr_epoch + 1) == exp.lr_epochs
        self.plot_train.add_data(exp.exp_idx, exp.train_loss_hist[start:end], annotate)
        self.plot_val.add_data(exp.exp_idx, exp.val_loss_hist[start:end], annotate)

        if exp.exp_idx == 0:
            learning_rates = torch.tensor([exp.cur_lr] * (end - start))
            self.plot_train.add_data(self.num_exps, learning_rates, annotate)
            self.plot_val.add_data(self.num_exps, learning_rates, annotate)

        self.plot_train.render(0.8, 100)
        self.plot_val.render(0.8, 100)

        if self.do_display:
            display.display(self.fig_loss)

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter
    dirname: str

    def __init__(self, name: str, now: datetime.datetime = None):
        if now is None:
            now = datetime.datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S")
        self.dirname = f"runs/{name}-{timestr}"
        self.writer = tboard.SummaryWriter(log_dir=self.dirname)
    
    def on_exp_end(self, exp: Experiment):
        # r = torch.randint(0, len(exp.last_val_in) - 1, size=(1,))[0].item()
        # print()
        # for p in exp.net.parameters():
        #     print(f"{p.type()=}")
        # print()
        # print(f"{exp.last_val_in[r:r+1]=}")
        # self.writer.add_graph(exp.net, exp.last_val_in[r:r+1])
        pass

    def on_epoch_end(self, exp: Experiment, epoch: int):
        train_loss = exp.train_loss_hist[epoch]
        val_loss = exp.val_loss_hist[epoch]

        self.writer.add_scalars("loss/train", {exp.label: train_loss}, global_step=epoch)
        self.writer.add_scalars("loss/validation", {exp.label: val_loss}, global_step=epoch)
        self.writer.add_scalars("learning rate", {exp.label: exp.cur_lr}, global_step=epoch)

class NanoGPTCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, start_lr: float, min_lr: float, warmup_epochs: int, lr_decay_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.start_lr = start_lr
        self.min_lr = min_lr
        self._step_count = 0

    def get_lr(self) -> float:
        if self._step_count < self.warmup_epochs:
            return [self.start_lr * self._step_count / self.warmup_epochs]
        if self._step_count > self.lr_decay_epochs:
            return [self.min_lr]
        denom = self.lr_decay_epochs - self.warmup_epochs
        if denom == 0:
            denom = 1
        decay_ratio = (self._step_count - self.warmup_epochs) / denom
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.start_lr - self.min_lr)]
    
    def step(self):
        self._step_count += 1
    
    def state_dict(self) -> Dict[str, any]:
        return {
            "warmup_epochs": self.warmup_epochs,
            "lr_decay_epochs": self.lr_decay_epochs,
            "start_lr": self.start_lr,
            "min_lr": self.min_lr,
            "_step_count": self._step_count
        }

 
    