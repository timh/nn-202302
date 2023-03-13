from typing import Literal, Callable
import torch
from torch import Tensor
import torch.nn.functional as F

# TODO: move scheduler, lazy sched/optim optimizers from trainer into here.

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
