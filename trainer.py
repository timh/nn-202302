import random
from typing import Tuple, Callable, Sequence, List, Iterable, Dict
from dataclasses import dataclass
from collections import defaultdict
import datetime
import math
import gc
import warnings

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

# TODO
# Fix use of "epochs". when using the RandomSampler, I'm really doing "steps".
# Use both of these terms properly, which also means extending the logging
# interface to allow for logging at the step.

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


class TrainerLogger:
    def on_exp_start(self, exp: Experiment):
        pass

    def on_exp_end(self, exp: Experiment):
        pass

    """
         epoch: current epoch
         batch: batch num of current epoch
     exp_batch: batch num since beginning of experiment
    train_loss: training loss for this batch only
    """
    def on_batch(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_batch: float):
        pass

    """
    gets called at the end of Trainer.on_epoch_end

         epoch: current (just-ended) epoch
    train_loss: training loss for entire epoch
    """
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        pass

    """
    gets called every trainer.update_frequency.

         epoch: current epoch
         batch: current batch in epoch
     exp_batch: current batch in experiment, i.e., global_steps
    train_loss: training loss for epoch so far
    """
    def print_status(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_epoch: float):
        pass

    """
    gets called (trainer.desired_val_count) times per Experiment.
    val_loss has already been computed.
    """
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        pass

"""
At the end of training an epoch:
- Trainer.on_epoch_end
  - (possibly) Logger.update_val_loss
  - Logger.on_epoch_end

At the end of training an experiment:
- everything in end of epoch training
- Trainer.on_exp_end
  - logger.on_exp_end
  - exp.end
"""
class Trainer:
    experiments: Iterable[Experiment]
    nexperiments: int
    logger: TrainerLogger

    total_samples = 0    # samples trained so far
    total_batches = 0    # batches (steps) trained so far
    total_epochs = 0     # epochs trained so far

    last_epoch_started_at: datetime.datetime = None
    last_print: datetime.datetime = None
    last_print_total_samples = 0
    last_print_total_batches = 0
    last_print_total_epochs = 0

    update_frequency: datetime.timedelta

    def __init__(self, 
                 experiments: Iterable[Experiment], nexperiments: int,
                 update_frequency = 10, 
                 logger: TrainerLogger = None):
        self.experiments = experiments
        self.nexperiments = nexperiments
        self.logger = logger
        self.update_frequency = datetime.timedelta(seconds=update_frequency)
        self.last_val = datetime.datetime.now()

    def on_exp_start(self, exp: Experiment, exp_idx: int):
        exp.start(exp_idx)
        if self.logger is not None:
            self.logger.on_exp_start(exp)

    def on_exp_end(self, exp: Experiment):
        if self.logger is not None:
            self.logger.on_exp_end(exp)

        # TODO: still might be leaking memory. not sure yet. TODO: run more systematic tests.
        exp.end()
        gc.collect()
        torch.cuda.empty_cache()
    
    def print_status(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_epoch: float):
        now = datetime.datetime.now()
        if ((now - self.last_print) >= self.update_frequency or
             (batch == exp.batch_size - 1 and epoch == exp.max_epochs - 1)):
            timediff = (now - self.last_print)

            samples_diff = float(self.total_samples - self.last_print_total_samples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(self.total_batches - self.last_print_total_batches)
            batch_per_sec = batch_diff / timediff.total_seconds()
            epoch_diff = float(self.total_epochs - self.last_print_total_epochs)
            epoch_per_sec = epoch_diff / timediff.total_seconds()

            if not epoch_per_sec:
                epoch_per_sec = 1
            eta_exp_done_sec = int((exp.max_epochs - epoch + 1) / epoch_per_sec)
            eta_exp_done_min = eta_exp_done_sec // 60
            eta_exp_done_sec -= eta_exp_done_min * 60

            print(f"epoch {epoch+1}/{exp.max_epochs} | batch {batch+1}/{exp.batch_size} | loss {train_loss_epoch:.5f} | samp/s {samples_per_sec:.3f} | epoch/sec {epoch_per_sec:.3f} | exp {exp.exp_idx+1}/{self.nexperiments} eta {eta_exp_done_min}m{eta_exp_done_sec}s")

            self.last_print = now
            self.last_print_total_samples = self.total_samples
            self.last_print_total_batches = self.total_batches
            self.last_print_total_epochs = self.total_epochs

            if self.logger is not None:
                self.logger.print_status(exp, epoch, batch, exp_batch, train_loss_epoch)

    # override this for new behavior after each epoch.
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float, device = "cpu"):
        # figure out validation loss
        val_start = datetime.datetime.now()
        with torch.no_grad():
            exp.net.eval()
            exp_batch = 0
            val_loss = 0.0
            for batch, (inputs, truth) in enumerate(exp.val_dataloader):
                inputs, truth = inputs.to(device), truth.to(device)

                exp_batch += 1
                val_out = exp.net(inputs)
                loss = exp.loss_fn(val_out, truth)

                inputs.cpu()
                truth.cpu()

                if loss.isnan():
                    print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                    return False

                val_loss += loss.item()

                exp.last_val_in = inputs
                exp.last_val_out = val_out
                exp.last_val_truth = truth

        val_end = datetime.datetime.now()

        val_loss /= exp_batch
        exp.val_loss_hist[epoch] = val_loss
        exp.lastepoch_val_loss = val_loss

        train_elapsed = (val_start - self.last_epoch_started_at).total_seconds()
        val_elapsed = (val_end - val_start).total_seconds()
        exp_elapsed = (val_end - exp.started_at).total_seconds()

        print(f"epoch {epoch + 1}/{exp.max_epochs} | \033[1mvalidation loss = {val_loss:.5f}\033[0m (train {train_elapsed:.2f}s, val {val_elapsed:.2f}s, exp so far {exp_elapsed:.2f}s)")
        if self.logger is not None:
            self.logger.on_epoch_end(exp, epoch, train_loss_epoch)
            self.logger.update_val_loss(exp, epoch, val_loss)
        print()
        # exp.nepochs += 1
        exp.nepochs = epoch
    
    def train(self, device = "cpu", use_amp = False):
        self.last_print = datetime.datetime.now()
        for exp_idx, exp in enumerate(self.experiments):
            # reset it between runs.
            if use_amp:
                self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
            else:
                self.scaler = None

            self.on_exp_start(exp, exp_idx)
            if exp.skip:
                print(f"skip {exp_idx + 1}/{self.nexperiments} | {exp.label}")
                continue

            print()
            print(f"\033[1mtrain {exp_idx+1}/{self.nexperiments}: {exp.nparams() / 1e6:.3f}M params | {exp.label}\033[0m")
            for epoch in range(exp.max_epochs):
                stepres = self.train_epoch(exp, epoch, device=device)
                if not stepres:
                    # something went wrong in that step. 
                    break
            
            exp.ended_at = datetime.datetime.now()

            self.on_exp_end(exp)

    def train_epoch(self, exp: Experiment, epoch: int, device: str) -> bool:
        self.total_epochs += 1
        self.last_epoch_started_at = datetime.datetime.now()

        exp.net.train()

        total_loss = 0.0
        for batch, (inputs, truth) in enumerate(exp.train_dataloader):
            self.total_batches += 1
            self.total_samples += len(inputs)

            exp.batch_size = len(inputs)
            exp.nsamples += len(inputs)
            # TODO: nbatches is the same as exp_batch, passed below.
            exp.nbatches += 1

            inputs, truth = inputs.to(device), truth.to(device)

            if self.scaler is not None:
                with torch.cuda.amp.autocast_mode.autocast():
                    out = exp.net(inputs)
                    loss = exp.loss_fn(out, truth)
            else:
                out = exp.net(inputs)
                loss = exp.loss_fn(out, truth)

            inputs.cpu()
            truth.cpu()

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                return False
            total_loss += loss.item()

            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            loss.backward()

            if self.scaler is not None:
                self.scaler.step(exp.optim)
                self.scaler.update()
            else:
                exp.optim.step()
            exp.optim.zero_grad(set_to_none=True)

            if exp.train_loss_hist is None:
                exp.train_loss_hist = torch.zeros((exp.batch_size * exp.max_epochs,))
                exp.val_loss_hist = torch.zeros((exp.max_epochs,))

            exp.last_train_in = inputs
            exp.last_train_out = out
            exp.last_train_truth = truth
            exp.train_loss_hist[exp.nbatches] = loss.item()

            if self.logger is not None:
                # TODO: really? passing in exp.nbatches? should clean this up.
                self.logger.on_batch(exp, epoch, batch, exp.nbatches, loss.item())
            self.print_status(exp, epoch, batch, exp.nbatches, total_loss / (batch + 1))

        exp.sched.step()

        total_loss /= (batch + 1)
        exp.lastepoch_train_loss = total_loss

        self.on_epoch_end(exp, epoch, total_loss, device=device)

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
    
    def on_batch(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_batch: float):
        self.writer.add_scalars("batch/tloss", {exp.label: train_loss_batch}, global_step=exp_batch)
        self.writer.add_scalars("batch/lr", {exp.label: exp.cur_lr}, global_step=exp_batch)

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        self.writer.add_scalars("epoch/tloss", {exp.label: train_loss_epoch}, global_step=epoch)
        self.writer.add_scalars("epoch/lr", {exp.label: exp.cur_lr}, global_step=epoch)
    
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        self.writer.add_scalars("epoch/vloss", {exp.label: val_loss}, global_step=epoch)


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

    def get_last_lr(self) -> float:
        return self.get_lr()
    
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

def lazy_optim_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer]:
    if exp.optim_type in ["", "adamw"]:
        optim = torch.optim.AdamW(exp.net.parameters(), lr=exp.startlr)
    elif exp.optim_type == "sgd":
        optim = torch.optim.SGD(exp.net.parameters(), lr=exp.startlr)
    else:
        raise ValueError(f"{exp}: unknown {exp.optim_type=}")
    return optim

def lazy_sched_fn(exp: Experiment) -> Tuple[torch.optim.lr_scheduler._LRScheduler]:
    startlr = exp.startlr
    endlr = getattr(exp, "endlr", None)
    if endlr is None:
        endlr = startlr / 10.0

    if exp.sched_type in ["", "nanogpt"]:
        scheduler = NanoGPTCosineScheduler(exp.optim, startlr, endlr, warmup_epochs=0, lr_decay_epochs=exp.max_epochs)
    elif exp.sched_type in ["constant", "ConstantLR"]:
        scheduler = torch.optim.lr_scheduler.ConstantLR(exp.optim, factor=1.0, total_iters=0)
    elif exp.sched_type in ["step", "StepLR"]:
        gamma = (endlr / startlr) ** (1 / exp.max_epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(exp.optim, 1, gamma=gamma)
    else:
        raise ValueError(f"{exp}: unknown {exp.sched_type=}")

    return scheduler

