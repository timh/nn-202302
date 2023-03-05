from dataclasses import dataclass

from typing import List, Union, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor

class NoisedDataset: pass


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

        # coeff = (torch.rand((1,))[0] - 0.5) * 2
        # noise = (torch.rand(orig.shape, device=orig.device) - 0.5) # * 2.0
        # noise = torch.randn(orig.shape)
        # noise.clamp_(min=0, max=1)
        noise = torch.normal(mean=0, std=0.5, size=orig.shape)

        input_noised_orig = orig + noise
        input_noised_orig.clamp_(min=0, max=1)
        truth = noise

        return input_noised_orig, truth
    
    def __next__(self) -> Tuple[Tensor, Tensor]:
        res = self[self._idx]
        self._idx += 1
        return res
    
    def __len__(self) -> int:
        return self._end - self._start

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

