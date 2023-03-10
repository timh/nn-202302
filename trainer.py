import random
from typing import Tuple, Callable, Sequence, List, Iterable, Dict
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import datetime
import math
import gc
import warnings

import torch, torch.optim
from torch.utils.data import DataLoader
from torch import nn, Tensor

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from IPython import display

import notebook
from experiment import Experiment

class TrainerLogger:
    def __init__(self, dirname: str):
        self.dirname = dirname

    def _status_path(self, exp: Experiment, subdir: str, epoch: int = 0, suffix = "") -> str:
        filename: List[str] = [exp.label]
        if epoch:
            filename.append(f"epoch_{epoch:04}")
        filename = ",".join(filename)
        path = Path(self.dirname, subdir or "", filename + suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

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
    last_epoch_total_samples: int = None
    last_print: datetime.datetime = None

    last_val_at: datetime.datetime = None
    val_limit_frequency: datetime.timedelta = None

    update_frequency: datetime.timedelta

    def __init__(self, 
                 experiments: Iterable[Experiment], nexperiments: int,
                 update_frequency = 10, val_limit_frequency = 0,
                 logger: TrainerLogger = None):
        self.experiments = experiments
        self.nexperiments = nexperiments
        self.logger = logger
        self.update_frequency = datetime.timedelta(seconds=update_frequency)
        self.val_limit_frequency = datetime.timedelta(seconds=val_limit_frequency)

    def on_exp_start(self, exp: Experiment, exp_idx: int):
        exp.start(exp_idx)
        if self.logger is not None:
            self.logger.on_exp_start(exp)

        if self.val_limit_frequency:
            self.last_val_at = datetime.datetime.now()

        self.nbatches_per_epoch = len(exp.train_dataloader)

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
            timediff = (now - self.last_epoch_started_at)

            # compute per/sec since the beginning of this epoch.
            samples_diff = float(self.total_samples - self.last_epoch_total_samples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_per_sec = samples_per_sec / exp.batch_size
            epoch_per_sec = batch_per_sec / self.nbatches_per_epoch

            if epoch_per_sec < 1:
                sec_per_epoch = 1 / epoch_per_sec
                epoch_rate = f"{sec_per_epoch:.3f}s/epoch"
            else:
                epoch_rate = f"epoch/s {epoch_per_sec:.3f}"

            print(f"epoch {epoch+1}/{exp.max_epochs} | batch {batch+1}/{self.nbatches_per_epoch} | \033[1;32mtrain loss {train_loss_epoch:.5f}\033[0m | samp/s {samples_per_sec:.3f} | {epoch_rate}")

            self.last_print = now
            self.last_print_total_samples = self.total_samples

            if self.logger is not None:
                self.logger.print_status(exp, epoch, batch, exp_batch, train_loss_epoch)

    # override this for new behavior after each epoch.
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float, device = "cpu"):
        # figure out validation loss
        exp.nepochs = epoch + 1
        now = datetime.datetime.now()
        if not self.val_limit_frequency or (now - self.last_val_at) >= self.val_limit_frequency:
            if self.val_limit_frequency:
                self.last_val_at = now

            val_start = now
            with torch.no_grad():
                exp.net.eval()
                exp_batch = 0
                val_loss = 0.0
                for batch, one_tuple in enumerate(exp.val_dataloader):
                    len_input = len(one_tuple) - 1
                    inputs, truth = one_tuple[:len_input], one_tuple[-1]

                    inputs = [inp.to(device) for inp in inputs]
                    truth = truth.to(device)

                    exp_batch += 1
                    val_out = exp.net(*inputs)
                    loss = exp.loss_fn(val_out, truth)

                    if loss.isnan():
                        print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                        return False

                    val_loss += loss.item()

                    exp.last_val_in = inputs
                    exp.last_val_out = val_out
                    exp.last_val_truth = truth

            val_end = datetime.datetime.now()

            val_loss /= exp_batch
            exp.val_loss_hist.append((epoch, val_loss))
            exp.lastepoch_val_loss = val_loss

            train_elapsed = (val_start - self.last_epoch_started_at).total_seconds()
            val_elapsed = (val_end - val_start).total_seconds()
            
            exp_elapsed = (val_end - exp.started_at).total_seconds()
            exp_elapsed_min = int(exp_elapsed / 60)
            exp_elapsed_sec = int(exp_elapsed) % 60

            exp_expected = exp_elapsed * exp.max_epochs / (epoch + 1)
            exp_expected_min = int(exp_expected / 60)
            exp_expected_sec = int(exp_expected) % 60

            print(f"epoch {epoch + 1}/{exp.max_epochs} "
                  f"| \033[1;32mval loss {val_loss:.5f}\033[0m "
                  f"| train {train_elapsed:.2f}s, val {val_elapsed:.2f}s "
                  f"| exp {exp.exp_idx+1}/{self.nexperiments}: {exp_elapsed_min}m{exp_elapsed_sec}s / {exp_expected_min}m{exp_expected_sec}s")
            if self.logger is not None:
                self.logger.update_val_loss(exp, epoch, val_loss)

        if self.logger is not None:
            self.logger.on_epoch_end(exp, epoch, train_loss_epoch)
        print()
    
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
        self.last_epoch_total_samples = self.total_samples

        exp.net.train()

        exp.batch_size = 0

        total_loss = 0.0
        for batch, one_tuple in enumerate(exp.train_dataloader):
            len_input = len(one_tuple) - 1
            inputs, truth = one_tuple[:len_input], one_tuple[-1]

            self.total_batches += 1
            self.total_samples += len(inputs[0])

            exp.nsamples += len(inputs[0])
            exp.batch_size = max(exp.batch_size, len(inputs[0]))

            inputs = [inp.to(device) for inp in inputs]
            truth = truth.to(device)

            if self.scaler is not None:
                with torch.cuda.amp.autocast_mode.autocast():
                    out = exp.net(*inputs)
                    loss = exp.loss_fn(out, truth)
            else:
                out = exp.net(*inputs)
                loss = exp.loss_fn(out, truth)

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                return False

            if self.scaler is not None:
                loss_scaled = self.scaler.scale(loss)
                loss_scaled.backward()
            else:
                loss.backward()
            total_loss += loss.item()

            if self.scaler is not None:
                self.scaler.step(exp.optim)
                self.scaler.update()
            else:
                exp.optim.step()
            exp.optim.zero_grad(set_to_none=True)

            if exp.train_loss_hist is None:
                exp.train_loss_hist = torch.zeros((exp.max_epochs * len(exp.train_dataloader),))
                exp.val_loss_hist = list()

            exp.last_train_in = inputs
            exp.last_train_out = out
            exp.last_train_truth = truth
            exp.train_loss_hist[exp.nbatches] = loss.item()

            # TODO: nbatches is the same as exp_batch, passed below.
            exp.nbatches += 1

            if self.logger is not None:
                # TODO: really? passing in exp.nbatches? should clean this up.
                self.logger.on_batch(exp, epoch, batch, exp.nbatches, loss.item())
            self.print_status(exp, epoch, batch, exp.nbatches, total_loss / (batch + 1))

        exp.sched.step()

        total_loss /= (batch + 1)
        exp.lastepoch_train_loss = total_loss

        self.on_epoch_end(exp, epoch, total_loss, device=device)

        return True

