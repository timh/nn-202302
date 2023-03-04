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
    epochs = 0     # epochs to be run for this experiment

    last_train_in: torch.Tensor = None
    last_train_out: torch.Tensor = None
    last_train_truth: torch.Tensor = None
    last_val_in: torch.Tensor = None
    last_val_out: torch.Tensor = None
    last_val_truth: torch.Tensor = None

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None

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
