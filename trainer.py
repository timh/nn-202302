import random
from typing import Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import datetime
import string

import torch, torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn

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

class Trainer:
    net: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optim: torch.optim.Optimizer

    # loss_fn(outputs, truth)
    def __init__(self, net: nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optim: torch.optim.Optimizer):
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim
    
    # override this for new behavior after each batch of training samples is
    # processed. this is called after the torch.isnan() check.
    def on_train_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        pass

    # override this for new behavior after each batch of validation samples is
    # processed. this is called after the torch.isnan() check.
    def on_val_batch(self, epoch: int, inputs: torch.Tensor, outputs: torch.Tensor, truth: torch.Tensor, loss: torch.Tensor):
        pass

    # override this for new behavior after each epoch.
    def on_epoch(self, num_epochs: int, epoch: int, train_loss_hist: torch.Tensor, val_loss_hist: torch.Tensor):
        now = datetime.datetime.now()
        if (now - self.last_print) >= datetime.timedelta(seconds=5) or (epoch == num_epochs - 1):
            timediff = (now - self.last_print)

            samples_diff = float(self.total_nsamples_sofar - self.last_print_nsamples)
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_diff = float(self.total_batch_sofar - self.last_print_batch)
            batch_per_sec = batch_diff / timediff.total_seconds()

            train_loss = train_loss_hist[epoch]
            val_loss = val_loss_hist[epoch]
            print(f"epoch {epoch+1}/{num_epochs}: train loss {train_loss:.5f}, val loss {val_loss:.5f} | samp/sec {samples_per_sec:.3f} | batch/sec {batch_per_sec:.3f}")

            self.last_print = now
            self.last_print_nsamples = self.total_nsamples_sofar
            self.last_print_batch = self.total_batch_sofar

            return True

        return False

    def train(self, num_epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader):
        train_loss_hist = torch.zeros((num_epochs,))
        val_loss_hist = torch.zeros_like(train_loss_hist)

        self.last_print = datetime.datetime.now()
        self.last_print_nsamples = 0
        self.last_print_batch = 0

        self.total_nsamples_sofar = 0
        self.total_batch_sofar = 0
        for epoch in range(num_epochs):
            train_loss = 0.0

            num_samples = 0
            for batch, (inputs, truth) in enumerate(train_dataloader):
                num_samples += len(inputs)
                out = self.net(inputs)
                loss = self.loss_fn(out, truth)

                if loss.isnan():
                    # not sure if there's a way out of this...
                    print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                    return train_loss_hist[:epoch], val_loss_hist[:epoch]
                train_loss += loss.item()

                self.on_train_batch(epoch, inputs, out, truth, loss)

                loss.backward()
                self.optim.step()

                self.total_nsamples_sofar += len(inputs)
                self.total_batch_sofar += 1

            train_loss /= num_samples

            with torch.no_grad():
                num_samples = 0
                val_loss = 0.0
                for batch, (inputs, truth) in enumerate(val_dataloader):
                    num_samples += len(inputs)
                    val_out = self.net(inputs)
                    loss = self.loss_fn(val_out, truth)

                    if loss.isnan():
                        print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                        return train_loss_hist[:epoch], val_loss_hist[:epoch]

                    val_loss += loss.item()

                    self.on_val_batch(epoch, inputs, out, truth, loss)

                val_loss /= num_samples

            train_loss_hist[epoch] = train_loss
            val_loss_hist[epoch] = val_loss

            self.on_epoch(num_epochs, epoch, train_loss_hist, val_loss_hist)
        
        return train_loss_hist, val_loss_hist
