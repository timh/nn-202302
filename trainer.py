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
    # TODO pull out the status printing and allow it to be called from within 
    # a minimatch loop, too
    def on_epoch_end(self, exp: Experiment, epoch: int):
        if self.logger is not None:
            self.logger.on_epoch_end(exp, epoch)
        
        now = datetime.datetime.now()
        if (now - exp.last_print) >= datetime.timedelta(seconds=10) or epoch == exp.epochs - 1:
            timediff = (now - exp.last_print)

            samples_diff = float(exp.nsamples - exp.last_print_nsamples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(exp.nbatches - exp.last_print_batch)
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
            exp.last_print_nsamples = exp.nsamples
            exp.last_print_batch = exp.nbatches
            exp.last_print_epoch = epoch

            if self.logger is not None:
                self.logger.on_epoch_end_infrequent(exp, epoch)

            return True

        return False
    
    def train(self, tcfg: TrainerConfig, use_amp = False):
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

            # reset it between runs.
            if use_amp:
                self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
            else:
                self.scaler = None

            self.on_exp_start(exp)

            print(f"train #{exp_idx} {exp.label}")
            for epoch in range(exp.epochs):
                stepres = self.train_epoch(exp, epoch)
                if not stepres:
                    # something went wrong in that step. 
                    break

                self.on_epoch_end(exp, epoch)

            self.on_exp_end(exp)

    def train_epoch(self, exp: Experiment, epoch: int) -> bool:
        train_loss = 0.0

        exp.net.train()
        num_batches = 0

        last_print = datetime.datetime.now()

        total_batches = len(exp.train_dataloader)
        for batch, (inputs, truth) in enumerate(exp.train_dataloader):
            num_batches += 1

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

            now = datetime.datetime.now()
            if (now - last_print) >= datetime.timedelta(seconds=5):
                # TODO merge this with on_epoch_infrequent
                print(f"epoch {epoch+1:4}/{exp.epochs} | batch {batch:3}/{total_batches}  |  train_loss={train_loss/num_batches:.5f}  |  lr={exp.cur_lr:.2E}")
                last_print = now

            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            loss.backward()

            if self.scaler is not None:
                self.scaler.step(exp.optim)
                self.scaler.update()
            else:
                exp.optim.step()
            exp.optim.zero_grad(set_to_none=True)

            exp.nsamples += len(inputs)
            exp.nbatches += 1

            exp.last_train_in = inputs
            exp.last_train_out = out
            exp.last_train_truth = truth
        
        exp.scheduler.step()
        exp.cur_lr = exp.scheduler.get_lr()[0]

        train_loss /= num_batches

        # figure out a validation loss
        with torch.no_grad():
            exp.net.eval()
            num_batches = 0
            val_loss = 0.0
            for batch, (inputs, truth) in enumerate(exp.val_dataloader):
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

        exp.train_loss_hist[epoch] = train_loss
        exp.val_loss_hist[epoch] = val_loss
    
        return True

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter
    dirname: str

    def __init__(self, name: str, now: datetime.datetime = None):
        if now is None:
            now = datetime.datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S")
        self.dirname = f"runs/{name}-{timestr}"
        self.writer = tboard.SummaryWriter(log_dir=self.dirname)
    
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

 
    