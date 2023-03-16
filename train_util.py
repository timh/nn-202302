import math
from typing import Tuple, Dict, Literal, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from experiment import Experiment

# mine
def DistanceLoss(out, truth):
    return torch.abs((truth - out)).mean()

# both are from:
#   https://stats.stackexchange.com/questions/438728/mean-absolute-percentage-error-returning-nan-in-pytorch

# mean absolute percentage error
# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
def MAPELoss(output: Tensor, target: Tensor, epsilon=1e-6) -> Tensor:
    return torch.mean(torch.abs((target - output) / (target + epsilon)))

# relative percentage difference
# https://en.wikipedia.org/wiki/Relative_change_and_difference
def RPDLoss(output: Tensor, target: Tensor, epsilon=1e-6) -> Tensor:
    return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2 + epsilon))

def edge_loss_fn(operator: Literal["*", "+"], backing_fn: Callable[[Tensor, Tensor], Tensor], device="cpu") -> Callable[[Tensor, Tensor], Tensor]:
    if operator not in ["*", "+"]:
        raise ValueError(f"invalid {operator=}")

    # basic sobel.
    def build_weight(kernel: Tensor) -> Tensor:
        # (kern, kern) -> (chan, kern, kern)
        withchan1 = torch.stack([kernel] * 3)
        withchan2 = torch.stack([withchan1] * 3)
        return withchan2

    vert_kernel = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]]).float().to(device)
    horiz_kernel = vert_kernel.T

    vert_weight = build_weight(vert_kernel)
    horiz_weight = build_weight(horiz_kernel)

    def edge_hv(img: Tensor) -> Tensor:
        vert = F.conv2d(img, vert_weight, padding=1)
        horiz = F.conv2d(img, horiz_weight, padding=1)
        return (vert + horiz) / 2

    def fn(output: Tensor, truth: Tensor) -> Tensor:
        batch, chan, width, height = output.shape
        output_edges = edge_hv(output)
        truth_edges = edge_hv(truth)
        loss_edges = backing_fn(output_edges, truth_edges)
        loss_backing = backing_fn(output, truth)
        if operator == "*":
            return loss_edges * loss_backing
        return loss_edges + loss_backing

    return fn

def get_loss_fn(loss_type: Literal["l1", "l2", "mse", "distance", "mape", "rpd"] = "l1", 
                device = "cpu") -> Callable[[Tensor, Tensor], Tensor]:
    loss_fns = {
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "mse": F.mse_loss,
        "distance": DistanceLoss,
        "mape": MAPELoss,
        "rpd": RPDLoss,
        "crossent": F.cross_entropy,
    }

    if loss_type.startswith("edge"):
        operator = loss_type[4]
        backing = loss_type[5:]
        backing_fn = loss_fns.get(backing, None)
        if backing_fn is None:
            raise ValueError(f"unknown {loss_type=} after edge")
        loss_fn = edge_loss_fn(operator=operator, backing_fn=backing_fn, device=device)
    else:
        loss_fn = loss_fns.get(loss_type, None)
        if loss_fn is None:
            raise ValueError(f"unknown {loss_type=}")

    return loss_fn

"""
Scheduler based on nanogpt's cosine decay scheduler:

See https://github.com/karpathy/nanoGPT/blob/master/train.py#L220
"""
class NanoGPTCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, start_lr: float, min_lr: float, warmup_epochs: int, lr_decay_epochs: int):
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.start_lr = start_lr
        self.min_lr = min_lr
        self._step_count = 0

    def get_lr(self) -> float:
        if self._step_count < self.warmup_epochs:
            return [self.start_lr * (self._step_count + 1) / self.warmup_epochs]
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
    endlr = exp.endlr
    if endlr is None:
        endlr = startlr / 10.0

    if exp.sched_type in ["", "nanogpt"]:
        scheduler = NanoGPTCosineScheduler(exp.optim, startlr, endlr, 
                                           warmup_epochs=exp.sched_warmup_epochs, 
                                           lr_decay_epochs=exp.max_epochs)
        found_lr = False
        for g in exp.optim.param_groups:
            if 'lr' in g:
                curval = g['lr']
                newlr = scheduler.get_lr()[0]
                print(f"lr {curval:E} -> {newlr:E}")
                g['lr'] = newlr
                found_lr = True
        if not found_lr:
            raise Exception(f"logic error: couldn't find 'lr' in param groups on {exp.optim=}, but none there")
        exp.optim.lr = scheduler.get_lr()
    elif exp.sched_type in ["constant", "ConstantLR"]:
        scheduler = torch.optim.lr_scheduler.ConstantLR(exp.optim, factor=1.0, total_iters=0)
    elif exp.sched_type in ["step", "StepLR"]:
        gamma = (endlr / startlr) ** (1 / exp.max_epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(exp.optim, 1, gamma=gamma)
    else:
        raise ValueError(f"{exp}: unknown {exp.sched_type=}")

    return scheduler

