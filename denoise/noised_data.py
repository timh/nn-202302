from dataclasses import dataclass

from typing import List, Union, Tuple, Literal
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from torch import Tensor
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import model

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

"""
inputs: (batch, width, height, chan)
 truth: (batch, 2, width, height, chan)
return: (1,)

"truth" actually contains both the noise that was applied and the original 
src image:
  noise = truth[:, 0, ...]
    src = truth[:, 1, ...]

"""
def twotruth_loss_fn(truth_is_noise: bool, loss_type: Literal["l1", "l2", "mse"] = "l1"):
    if loss_type == "l1":
        loss_fn = F.l1_loss
    elif loss_type in ["l2", "mse"]:
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"unknown {loss_type=}")

    def fn(inputs: Tensor, truth: Tensor) -> Tensor:
        batch, ntruth, chan, width, height = truth.shape

        if truth_is_noise:
            truth = truth[:, 0, :, :, :]
        else:
            truth = truth[:, 1, :, :, :]
        truth = truth.view(batch, chan, width, height)
        return loss_fn(inputs, truth)
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

def create_dataloaders(noised_data: NoisedDataset, batch_size: int, minicnt: int, train_split: float = 0.9, train_all_data = False, val_all_data = False) -> Tuple[DataLoader, DataLoader]:
    ntrain = int(len(noised_data) * 0.9)

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

