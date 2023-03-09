import sys
from dataclasses import dataclass

from typing import List, Union, Tuple, Literal, Callable
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import model
sys.path.append("..")
import trainer

@dataclass
class _Iter:
    dataset: Dataset
    _start: int
    _end: int
    _idx: int = 0

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            raise Exception("slice not supported")
        
        orig, _ = self.dataset[self._start + idx]

        # amount = torch.rand((1, ))[0].item()
        # noise = model.gen_noise(orig.shape) * amount
        noise = model.gen_noise(orig.shape)

        input_noised_orig = orig + noise
        input_noised_orig.clamp_(min=0, max=1)
        truth = torch.stack([noise, orig], dim=0)

        return input_noised_orig, truth
    
    def __next__(self) -> Tuple[Tensor, Tensor]:
        res = self[self._idx]
        self._idx += 1
        return res
    
    def __len__(self) -> int:
        return self._end - self._start

# %%

def edge_loss_fn(backing_fn: Callable[[Tensor, Tensor], Tensor], device="cpu") -> Callable[[Tensor, Tensor], Tensor]:

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
        # return average of edge- and normal loss
        return (backing_fn(output_edges, truth_edges) + backing_fn(output, truth)) / 2

    return fn


# %%

"""
output: (batch, width, height, chan)
 truth: (batch, 2, width, height, chan)
return: (1,)

"truth" actually contains both the noise that was applied and the original 
src image:
  noise = truth[:, 0, ...]
    src = truth[:, 1, ...]

"""
def twotruth_loss_fn(loss_type: Literal["l1", "l2", "mse", "distance", "mape", "rpd"] = "l1", truth_is_noise: bool = False, device = "cpu") -> Callable[[Tensor, Tensor], Tensor]:
    loss_fns = {
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "mse": F.mse_loss,
        "distance": trainer.DistanceLoss,
        "mape": trainer.MAPELoss,
        "rpd": trainer.RPDLoss,
    }

    if loss_type.startswith("edge"):
        backing = loss_type[4:]
        backing_fn = loss_fns.get(backing, None)
        if backing_fn is None:
            raise ValueError(f"unknown {loss_type=} after edge")
        loss_fn = edge_loss_fn(backing_fn=backing_fn, device=device)
    else:
        loss_fn = loss_fns.get(loss_type, None)
        if loss_fn is None:
            raise ValueError(f"unknown {loss_type=}")

    def fn(output: Tensor, truth: Tensor) -> Tensor:
        batch, ntruth, chan, width, height = truth.shape

        if truth_is_noise:
            truth = truth[:, 0, :, :, :]
        else:
            truth = truth[:, 1, :, :, :]
        truth = truth.view(batch, chan, width, height)
        return loss_fn(output, truth)
    return fn

class NoisedDataset:
    dataset: Dataset

    def __init__(self, base_dataset: Dataset):
        self.dataset = base_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return _Iter(dataset=self.dataset, _start=0, _end=len(self.dataset))
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            start, end, _stride = idx.indices(len(self))
            return _Iter(dataset=self.dataset, _start=start, _end=end)
        return self.dataset[idx]

def load_dataset(image_dirname: str, image_size: int) -> NoisedDataset:
    base_dataset = torchvision.datasets.ImageFolder(
        root=image_dirname,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    return NoisedDataset(base_dataset=base_dataset)

def create_dataloaders(noised_data: NoisedDataset, batch_size: int, 
                       minicnt: int = 0, train_split: float = 0.9, 
                       train_all_data = True, val_all_data = True) -> Tuple[DataLoader, DataLoader]:
    ntrain = int(len(noised_data) * train_split)

    if (not train_all_data or not val_all_data) and minicnt == 0:
        raise ValueError(f"minicnt must be set if {train_all_data=} and/or {val_all_data=} are False")

    train_data = noised_data[:ntrain]
    val_data = noised_data[ntrain:]

    if train_all_data:
        train_dl = DataLoader(train_data, batch_size=batch_size)
    else:
        train_sampler = RandomSampler(train_data, num_samples=batch_size * minicnt)
        train_dl = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

    if val_all_data:
        val_dl = DataLoader(val_data, batch_size=batch_size)
    else:
        val_sampler = RandomSampler(val_data, num_samples=batch_size * minicnt)
        val_dl = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)

    return train_dl, val_dl

