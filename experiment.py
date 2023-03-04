from dataclasses import dataclass
from typing import Callable, Dict, List
import datetime

import torch, torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

@dataclass(kw_only=True)
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

    nsamples = 0   # samples trained against so far
    nbatches = 0   # batches trained against so far

    last_print: datetime.datetime = None
    last_print_batch = 0
    last_print_nsamples = 0

    epochs = 0

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
    
    def state_dict(self) -> Dict[str, any]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loss_hist": self.train_loss_hist,
            "val_loss_hist": self.val_loss_hist,

            "label": self.label,
            "epochs": self.epochs,
            "exp_idx": self.exp_idx,
            "cur_lr": self.cur_lr,
            "started_at": self.started_at.strftime("%Y%m%d-%H%M%S"),
            "ended_at": self.ended_at.strftime("%Y%m%d-%H%M%S"),
        }

    def train_epoch(self, epoch: int, accel: Accelerator) -> bool:
        train_loss = 0.0

        last_print = datetime.datetime.now()

        self.net.train()
        num_batches = 0
        total_batches = len(self.train_dataloader)
        for batch, (inputs, truth) in enumerate(self.train_dataloader):
            num_batches += 1
            out = self.net(inputs)
            loss = self.loss_fn(out, truth)

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                return False
            train_loss += loss.item()

            now = datetime.datetime.now()
            if (now - last_print) >= datetime.timedelta(seconds=5):
                print(f"epoch {epoch+1:4}/{self.epochs} | batch {batch:3}/{total_batches}  |  train_loss={train_loss/num_batches:.5f}  |  lr={self.cur_lr:.2E}")
                last_print = now

            if accel is not None:
                accel.backward(loss)
            else:
                loss.backward()

            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            self.nsamples += len(inputs)
            self.nbatches += 1

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
                    print(f"!! validation loss {loss} at epoch {epoch}, batch {batch} -- returning!")
                    return False

                val_loss += loss.item()

                self.last_val_in = inputs
                self.last_val_out = val_out
                self.last_val_truth = truth

            val_loss /= num_batches

        self.train_loss_hist[epoch] = train_loss
        self.val_loss_hist[epoch] = val_loss
    
        return True
