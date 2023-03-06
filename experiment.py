from dataclasses import dataclass
from typing import Callable, Dict, List
import datetime

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

@dataclass(kw_only=True)
class Experiment:
    label: str
    net: nn.Module                               # can be passed in as None. pass net_fn to lazy load
    loss_fn: Callable[[Tensor, Tensor], Tensor]  # (outputs, truth) -> loss

    # epochs to be run for this experiment
    epochs: int

    train_dataloader: DataLoader
    val_dataloader: DataLoader

    """set to True to stop this experiment"""
    skip: bool = False

    net_fn: Callable[[], nn.Module] = None
    optim: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None

    exp_idx: int = 0
    cur_lr: float = 0.0
    train_loss_hist: Tensor = None
    val_loss_hist: Tensor = None
    last_val_loss: float = None

    nsamples = 0   # samples trained against so far
    nbatches = 0   # batches trained against so far

    last_train_in: Tensor = None
    last_train_out: Tensor = None
    last_train_truth: Tensor = None
    last_val_in: Tensor = None
    last_val_out: Tensor = None
    last_val_truth: Tensor = None

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None
    elapsed: float = None
