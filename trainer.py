import random
from typing import Tuple, Callable, Sequence, List
from dataclasses import dataclass
from collections import defaultdict
import datetime
import string

import torch, torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import notebook

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
class TrainConfig:
    net: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (outputs, truth)
    learning_rates: List[Tuple[float, int]]                        # LR, num/epochs
    optimizers: List[torch.optim.Optimizer]                        # one optimizer for each LR
    train_dataloader: DataLoader
    val_dataloader: DataLoader

    cur_lr: float = 0.0
    train_loss_hist: torch.Tensor = None
    val_loss_hist: torch.Tensor = None

    name: str = ""

    last_print: datetime.datetime = None
    last_print_nsamples = 0
    last_print_batch = 0
    last_plot_epoch = 0

    total_nsamples_sofar = 0
    total_batch_sofar = 0

class Trainer:
    fig_loss: Figure = None

    def __init__(self, fig_loss: Figure = None):
        self.fig_loss = fig_loss

    # override this for new behavior after each batch of training samples is
    # processed. this is called after the torch.isnan() check.
    def on_train_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        pass

    # override this for new behavior after each batch of validation samples is
    # processed. this is called after the torch.isnan() check.
    def on_val_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        pass

    # override this for new behavior after each epoch.
    def on_epoch(self, tcfg: TrainConfig, num_epochs: int, epoch: int):
        now = datetime.datetime.now()
        if (now - tcfg.last_print) >= datetime.timedelta(seconds=5) or (epoch == num_epochs - 1):
            timediff = (now - tcfg.last_print)

            samples_diff = float(tcfg.total_nsamples_sofar - tcfg.last_print_nsamples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(tcfg.total_batch_sofar - tcfg.last_print_batch)
            batch_per_sec = batch_diff / timediff.total_seconds()

            train_loss = tcfg.train_loss_hist[epoch]
            val_loss = tcfg.val_loss_hist[epoch]
            print(f"epoch {epoch+1}/{num_epochs}: train loss {train_loss:.5f}, val loss {val_loss:.5f} | samp/sec {samples_per_sec:.3f} | batch/sec {batch_per_sec:.3f}")

            tcfg.last_print = now
            tcfg.last_print_nsamples = tcfg.total_nsamples_sofar
            tcfg.last_print_batch = tcfg.total_batch_sofar

            self.on_epoch_plot(tcfg, num_epochs, tcfg.last_plot_epoch, epoch)

            tcfg.last_plot_epoch = epoch

            return True

        return False
    
    def on_epoch_plot(self, tcfg: TrainConfig, num_epochs: int, epoch_start: int, epoch_end: int):
        if self.fig_loss is not None:
            learning_rates = torch.tensor([self.tcfg.cur_lr] * (epoch_end - epoch_start))

            self.plot_train.add_data(0, tcfg.train_loss_hist[epoch_start:epoch_end + 1])
            self.plot_val.add_data(0, tcfg.val_loss_hist[epoch_start:epoch_end + 1])

            self.plot_train.add_data(1, learning_rates)
            self.plot_val.add_data(1, learning_rates)

            self.plot_train.render(0.8, 100)
            self.plot_val.render(0.8, 100)

            self.plot_train.render(0.8, 100, epoch_end == num_epochs)
            self.plot_val.render(0.8, 100, epoch_end == num_epochs)

    def train(self, tcfg: TrainConfig, fig_loss: Figure, plot_train: notebook.Plot = None, plot_val: notebook.Plot = None):
        total_epochs = sum([lrpair[1] for lrpair in tcfg.learning_rates])

        tcfg.train_loss_hist = torch.zeros((total_epochs,))
        tcfg.val_loss_hist = torch.zeros_like(tcfg.train_loss_hist)
        tcfg.last_print = datetime.datetime.now()

        if fig_loss is not None:
            if plot_train is None:
                plot_train = notebook.Plot(total_epochs, ["train loss", "learning rate"], self.fig_loss, nrows=2, idx=1)
                plot_val = notebook.Plot(total_epochs, ["val loss", "learning rate"], self.fig_loss, nrows=2, idx=2)
            self.plot_train = plot_train
            self.plot_val = plot_val

        epoch = 0
        for lridx, (lr, num_epochs) in enumerate(tcfg.learning_rates):
            tcfg.cur_lr = lr
            tcfg.optim = tcfg.optimizers[lridx]
            print(f"train {num_epochs} @ {lr:.0E}")
            for _epoch in range(num_epochs):
                train_loss = 0.0

                num_samples = 0
                for batch, (inputs, truth) in enumerate(tcfg.train_dataloader):
                    num_samples += len(inputs)
                    out = tcfg.net(inputs)
                    loss = tcfg.loss_fn(out, truth)

                    if loss.isnan():
                        # not sure if there's a way out of this...
                        print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                        return tcfg.train_loss_hist[:epoch], tcfg.val_loss_hist[:epoch]
                    train_loss += loss.item()

                    self.on_train_batch(epoch, inputs, out, truth, loss)

                    loss.backward()
                    tcfg.optim.step()

                    tcfg.total_nsamples_sofar += len(inputs)
                    tcfg.total_batch_sofar += 1

                train_loss /= num_samples

                with torch.no_grad():
                    tcfg.net.eval()
                    num_samples = 0
                    val_loss = 0.0
                    for batch, (inputs, truth) in enumerate(tcfg.val_dataloader):
                        num_samples += len(inputs)
                        val_out = tcfg.net(inputs)
                        loss = tcfg.loss_fn(val_out, truth)

                        if loss.isnan():
                            print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                            return tcfg.train_loss_hist[:epoch], tcfg.val_loss_hist[:epoch]

                        val_loss += loss.item()

                        self.on_val_batch(epoch, inputs, out, truth, loss)

                    val_loss /= num_samples
                    tcfg.net.train()

                tcfg.train_loss_hist[epoch] = train_loss
                tcfg.val_loss_hist[epoch] = val_loss

                self.on_epoch(tcfg, num_epochs, epoch)
                epoch += 1
        
        return tcfg.train_loss_hist, tcfg.val_loss_hist

class TrainerMulti(Trainer):
    num_tconfigs: int

    def __init__(self, fig_loss: Figure = None):
        super().__init__(fig_loss)
    
    def train(self, tconfigs: Sequence[TrainConfig], fig_loss: Figure):
        train_labels = ["train loss " + tcfg.name for tcfg in tconfigs]
        val_labels = ["val loss " + tcfg.name for tcfg in tconfigs]

        total_epochs = [sum([lrpair[1] for lrpair in tcfg.learning_rates]) for tcfg in tconfigs]
        total_epochs = max(total_epochs)
        
        plot_train = notebook.Plot(total_epochs, train_labels + ["learning rate"], fig_loss, nrows=2, idx=1)
        plot_val = notebook.Plot(total_epochs, val_labels + ["learning rate"], fig_loss, nrows=2, idx=2)

        self.num_tconfigs = len(tconfigs)
        for i, tcfg in enumerate(tconfigs):
            tcfg.idx = i
            super().train(tcfg, fig_loss, plot_train, plot_val)

    def on_epoch_plot(self, tcfg: TrainConfig, num_epochs: int, epoch_start: int, epoch_end: int):
        learning_rates = torch.tensor([tcfg.cur_lr] * (epoch_end - epoch_start))

        self.plot_train.add_data(tcfg.idx, tcfg.train_loss_hist[epoch_start : epoch_end  + 1])
        self.plot_val.add_data(tcfg.idx, tcfg.val_loss_hist[epoch_start : epoch_end  + 1])

        self.plot_train.add_data(self.num_tconfigs, learning_rates)
        self.plot_val.add_data(self.num_tconfigs, learning_rates)

        self.plot_train.render(0.8, 100, epoch_end == num_epochs)
        self.plot_val.render(0.8, 100, epoch_end == num_epochs)

