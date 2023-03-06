import random
from typing import Tuple, Callable, Sequence, List, Iterable, Dict
from dataclasses import dataclass
from collections import defaultdict
import datetime
import math

import torch, torch.optim
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch.utils.tensorboard as tboard

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
def MAPELoss(output: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs((target - output) / (target + 1e-6)))

# relative percentage difference
# https://en.wikipedia.org/wiki/Relative_change_and_difference
def RPDLoss(output: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))    


@dataclass
class TrainerConfig:
    experiments: Iterable[Experiment]

    """number of experiments. possibly can't call len() on experiments"""
    nexperiments: int
    get_optimizer_fn: \
        Callable[[Experiment], 
                 Tuple[torch.optim.Optimizer, 
                       torch.optim.lr_scheduler._LRScheduler]]

class TrainerLogger:
    def on_exp_start(self, exp: Experiment):
        pass

    def on_exp_end(self, exp: Experiment):
        pass

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss: float):
        pass

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        pass

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        pass

class Trainer:
    logger: TrainerLogger = None

    total_samples = 0    # samples trained so far
    total_batches = 0    # batches trained so far
    total_epochs = 0     # epochs trained so far

    last_print: datetime.datetime = None
    last_print_total_samples = 0
    last_print_total_batches = 0
    last_print_total_epochs = 0

    val_every: datetime.timedelta
    last_val: datetime.timedelta

    def __init__(self, update_frequency = 10, val_frequency = 60, logger: TrainerLogger = None):
        self.logger = logger
        self.update_frequency = datetime.timedelta(seconds=update_frequency)
        self.val_frequency = datetime.timedelta(seconds=val_frequency)
        self.last_val = datetime.datetime.now()

    def on_exp_start(self, exp: Experiment):
        if self.logger is not None:
            self.logger.on_exp_start(exp)

    def on_exp_end(self, exp: Experiment):
        if self.logger is not None:
            self.logger.on_exp_end(exp)
    
    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        now = datetime.datetime.now()
        if ((now - self.last_print) >= self.update_frequency or
             (batch == batches - 1 and epoch == exp.epochs - 1)):
            timediff = (now - self.last_print)

            samples_diff = float(self.total_samples - self.last_print_total_samples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(self.total_batches - self.last_print_total_batches)
            batch_per_sec = batch_diff / timediff.total_seconds()
            epoch_diff = float(self.total_epochs - self.last_print_total_epochs)
            epoch_per_sec = epoch_diff / timediff.total_seconds()

            if not epoch_per_sec:
                epoch_per_sec = 1
            eta_exp_done_sec = int((exp.epochs - epoch + 1) / epoch_per_sec)
            eta_exp_done_min = eta_exp_done_sec // 60
            eta_exp_done_sec -= eta_exp_done_min * 60

            print(f"epoch {epoch+1}/{exp.epochs} | batch {batch+1}/{batches} | loss {train_loss:.5f} | samp/s {samples_per_sec:.3f} | epoch/sec {epoch_per_sec:.3f} | exp {exp.exp_idx+1}/{self.tcfg.nexperiments} eta {eta_exp_done_min}m{eta_exp_done_sec}s")

            self.last_print = now
            self.last_print_total_samples = self.total_samples
            self.last_print_total_batches = self.total_batches
            self.last_print_total_epochs = self.total_epochs

            if self.logger is not None:
                self.logger.print_status(exp, epoch, batch, batches, train_loss)

    # override this for new behavior after each epoch.
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss: float, device = "cpu"):
        now = datetime.datetime.now()
        if (now - self.last_val >= self.val_frequency) or (epoch == exp.epochs - 1):
            self.last_val = now

            # figure out a validation loss
            with torch.no_grad():
                exp.net.eval()
                num_batches = 0
                val_loss = 0.0
                for batch, (inputs, truth) in enumerate(exp.val_dataloader):
                    inputs, truth = inputs.to(device), truth.to(device)

                    num_batches += 1
                    val_out = exp.net(inputs)
                    loss = exp.loss_fn(val_out, truth)

                    if loss.isnan():
                        print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                        return False

                    val_loss += loss.item()

                    exp.last_val_in = inputs
                    exp.last_val_out = val_out
                    exp.last_val_truth = truth

                val_loss /= num_batches

            exp.val_loss_hist[epoch] = val_loss
            exp.last_val_loss = val_loss
            print(f"epoch {epoch + 1}/{exp.epochs} | validation loss = {val_loss:.5f}")
            
            if self.logger is not None:
                self.logger.update_val_loss(exp, epoch, val_loss)

        if self.logger is not None:
            self.logger.on_epoch_end(exp, epoch, train_loss)
    
    def train(self, tcfg: TrainerConfig, device = "cpu", use_amp = False):
        self.last_print = datetime.datetime.now()
        self.tcfg = tcfg
        for exp_idx, exp in enumerate(tcfg.experiments):
            exp.exp_idx = exp_idx
            exp.train_loss_hist = torch.zeros((exp.epochs,))
            exp.val_loss_hist = torch.zeros_like(exp.train_loss_hist)
            exp.last_val_loss = 0.0

            # enable lazy loading of network
            if exp.net is None:
                exp.net = exp.net_fn()

            exp.optim, exp.scheduler = tcfg.get_optimizer_fn(exp)
            exp.cur_lr = exp.scheduler.get_lr()[0]
            exp.started_at = datetime.datetime.now()

            # reset it between runs.
            if use_amp:
                self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
            else:
                self.scaler = None

            self.on_exp_start(exp)
            if exp.skip:
                print(f"skip experiment #{exp_idx + 1}/{tcfg.nexperiments} | {exp.label}")
                continue

            nparams = sum(p.numel() for p in exp.net.parameters())
            print(f"train #{exp_idx+1}/{tcfg.nexperiments}: {nparams / 1e6:.2f}M params | {exp.label}")
            for epoch in range(exp.epochs):
                stepres = self.train_epoch(exp, epoch, device=device)
                if not stepres:
                    # something went wrong in that step. 
                    break
            
            exp.ended_at = datetime.datetime.now()

            self.on_exp_end(exp)

    def train_epoch(self, exp: Experiment, epoch: int, device: str) -> bool:
        self.total_epochs += 1

        exp.net.train()

        num_batches = len(exp.train_dataloader)
        num_batches_sofar = 0
        train_loss = 0.0
        for batch, (inputs, truth) in enumerate(exp.train_dataloader):
            num_batches_sofar += 1
            self.total_batches += 1
            self.total_samples += len(inputs)

            exp.nsamples += len(inputs)
            exp.nbatches += 1

            inputs, truth = inputs.to(device), truth.to(device)

            if self.scaler is not None:
                with torch.cuda.amp.autocast_mode.autocast():
                    out = exp.net(inputs)
                    loss = exp.loss_fn(out, truth)
            else:
                out = exp.net(inputs)
                loss = exp.loss_fn(out, truth)

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                return False
            train_loss += loss.item()


            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            loss.backward()

            if self.scaler is not None:
                self.scaler.step(exp.optim)
                self.scaler.update()
            else:
                exp.optim.step()
            exp.optim.zero_grad(set_to_none=True)

            exp.last_train_in = inputs
            exp.last_train_out = out
            exp.last_train_truth = truth

            self.print_status(exp, epoch, batch, num_batches, train_loss / num_batches_sofar)

        exp.scheduler.step()
        exp.cur_lr = exp.scheduler.get_lr()[0]

        train_loss /= num_batches
        exp.train_loss_hist[epoch] = train_loss

        self.on_epoch_end(exp, epoch, train_loss, device=device)

        return True

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter
    basename: str
    dirname: str

    def __init__(self, basename: str, now: datetime.datetime = None):
        if now is None:
            now = datetime.datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S")
        self.basename = basename
        self.dirname = f"runs/{self.basename}-{timestr}"
        self.writer = tboard.SummaryWriter(log_dir=self.dirname)
    
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss: float):
        self.writer.add_scalars("loss/train", {exp.label: train_loss}, global_step=epoch)
        self.writer.add_scalars("learning rate", {exp.label: exp.cur_lr}, global_step=epoch)
    
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        self.writer.add_scalars("loss/validation", {exp.label: val_loss}, global_step=epoch)


"""
Scheduler based on nanogpt's cosine decay scheduler:

See https://github.com/karpathy/nanoGPT/blob/master/train.py#L220
"""
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

