from dataclasses import dataclass
from typing import Callable
import datetime

import torch, torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

@dataclass
class Experiment:
    label: str
    net: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (outputs, truth) -> loss
    train_dataloader: DataLoader
    val_dataloader: DataLoader

    optim: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None

    exp_idx: int = 0
    cur_lr: float = 0.0
    train_loss_hist: torch.Tensor = None
    val_loss_hist: torch.Tensor = None

    total_nsamples_sofar = 0
    total_batch_sofar = 0

    exp_epochs = 0
    lr_epochs = 0

    last_train_in: torch.Tensor = None
    last_train_out: torch.Tensor = None
    last_train_truth: torch.Tensor = None
    last_val_in: torch.Tensor = None
    last_val_out: torch.Tensor = None
    last_val_truth: torch.Tensor = None

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None

    def on_start(self):
        self.started_at = datetime.datetime.now()
    
    def on_end(self):
        self.ended_at = datetime.datetime.now()

    def step(self, exp_epoch: int, lr_epoch: int, accel: Accelerator) -> bool:
        train_loss = 0.0

        last_print = datetime.datetime.now()

        self.net.train()
        num_batches = 0
        for batch, (inputs, truth) in enumerate(self.train_dataloader):
            num_batches += 1
            out = self.net(inputs)
            loss = self.loss_fn(out, truth)

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at lr_epoch {lr_epoch} / exp_epoch {exp_epoch}, batch {batch} -- returning!")
                return False
            train_loss += loss.item()

            now = datetime.datetime.now()
            if (now - last_print) >= datetime.timedelta(seconds=5):
                print(f"epoch {exp_epoch+1:4}/{self.exp_epochs} | lr {lr_epoch+1:4}/{self.lr_epochs} | batch {batch:3}  |  train_loss={train_loss/num_batches:.5f}  |  lr={self.cur_lr:.2E}")
                last_print = now

            if accel is not None:
                accel.backward(loss)
            else:
                loss.backward()

            self.optim.step()

            self.total_nsamples_sofar += len(inputs)
            self.total_batch_sofar += 1

            self.last_train_in = inputs
            self.last_train_out = out
            self.last_train_truth = truth
        
        if self.scheduler is not None:
            self.scheduler.step()
            self.cur_lr = self.scheduler.get_lr()[0]

        train_loss /= num_batches

        # figure out a validation loss
        with torch.no_grad():
            self.net.eval()
            num_batches = 0
            val_loss = 0.0
            for batch, (inputs, truth) in enumerate(self.val_dataloader):
                num_batches += 1
                val_out = self.net(inputs)
                loss = self.loss_fn(val_out, truth)

                if loss.isnan():
                    print(f"!! validation loss {loss} at epoch {exp_epoch}, batch {batch} -- returning!")
                    return False

                val_loss += loss.item()

                self.last_val_in = inputs
                self.last_val_out = val_out
                self.last_val_truth = truth

            val_loss /= num_batches

        self.train_loss_hist[exp_epoch] = train_loss
        self.val_loss_hist[exp_epoch] = val_loss
    
        return True
