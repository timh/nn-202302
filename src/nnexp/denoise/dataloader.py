from dataclasses import dataclass
from typing import List, Tuple, Union, Literal
from pathlib import Path
import random
import math

import torch
from torch import Tensor
from torch.utils.data import Dataset

from . import noisegen, clip_cache
from .latent_cache import LatentCache
from .models import vae
from .models.mtypes import VarEncoderOutput

DSItem = Tuple[Tensor, Tensor]
DSItem1OrN = Union[DSItem, List[DSItem]]

@dataclass(kw_only=True)
class DSIter:
    dataset: Dataset
    idx: int = 0

    def __next__(self) -> Tuple[Tensor, Tensor]:
        res = self.dataset[self.idx]
        self.idx += 1
        return res

class DSBase(Dataset):
    base_dataset: Dataset
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
    
    def __len__(self) -> int:
        return len(self.base_dataset)

    def __iter__(self):
        return DSIter(dataset=self)
    
    def __getitem__(self, idx: Union[int, slice]) -> DSItem1OrN:
        if isinstance(idx, slice):
            start, end, stride = idx.indices(len(self))
            indices = list(range(start, end, stride))
            return DSSubset(self, start=start, end=end)

        return self._ds_getitem(idx)
    
    def _ds_getitem(self, idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplemented("override this")


class DSSubset(DSBase):
    base_dataset: DSBase
    start: int
    end: int

    def __init__(self, base_dataset: DSBase, start: int, end: int):
        self.base_dataset = base_dataset
        self.start = start
        self.end = end
    
    def __len__(self) -> int:
        return self.end - self.start
    
    def _ds_getitem(self, idx: int) -> DSItem:
        return self.base_dataset[self.start + idx]

class ShuffleDataset(DSBase):
    idxs: List[int]

    def __init__(self, base_dataset: Dataset):
        super().__init__(base_dataset)
        self.dataset = base_dataset
        self.idxs = list(range(0, len(self.dataset)))
        random.shuffle(self.idxs)
    
    def _ds_getitem(self, idx: int) -> DSItem:
        idx = self.idxs[idx]
        return self.dataset[idx]

def split_dataset(dataset: Dataset, train_split: float = 0.9) -> Tuple[Dataset, Dataset]:
    split_idx = int(len(dataset) * train_split)
    train_ds = DSSubset(dataset, start=0, end=split_idx)
    val_ds = DSSubset(dataset, start=split_idx, end=len(dataset))
    return train_ds, val_ds

@dataclass(kw_only=True)
class _NDLIter:
    _idx: int
    ndataloader: 'NoisedDataLoader'

    def __next__(self) -> List[DSItem]:
        if self._idx >= len(self.ndataloader):
            raise StopIteration()

        res = self.ndataloader[self._idx]
        self._idx += 1
        return res

"""
NoisedDataset: take a backing dataset and apply noise to it.
"""
class NoisedDataLoader:
    sched: noisegen.NoiseSchedule
    unconditional_ratio: Tensor

    dataset: Dataset
    ds_idxs: List[int]

    def __init__(self, *,
                 dataset: Dataset, noise_schedule: noisegen.NoiseSchedule,
                 batch_size: int, shuffle: bool = False,
                 unconditional_ratio: float = None):
        self.batch_size = batch_size
        self.sched = noise_schedule

        self.dataset = dataset
        self.ds_idxs = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.ds_idxs)
        self._idxs = 0

        # zero other ratio - zero the embeds with this probability. to drop some conditional
        # embeddings to unconditional
        self.unconditional_ratio = None
        if unconditional_ratio:
            self.unconditional_ratio = torch.tensor(unconditional_ratio)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self) -> _NDLIter:
        return _NDLIter(_idx=0, ndataloader=self)

    def __getitem__(self, idx: int) -> List[DSItem]:
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))

        input_list = list()
        truth_list = list()

        timestep = torch.randint(low=0, high=self.sched.timesteps, size=(1,))
        rval = torch.rand(size=(1,)).item()
        for idx in range(start_idx, end_idx):
            ds_idx = self.ds_idxs[idx]
            inputs, _truth = self.dataset[ds_idx]
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                orig = inputs[0]

                # e.g., clip_embed
                other_inputs: List[Tensor] = inputs[1:]
                if self.unconditional_ratio and other_inputs and rval < self.unconditional_ratio:
                    other_inputs = [torch.zeros_like(oi) for oi in other_inputs]
            else:
                orig = inputs
                other_inputs = []

            noised_orig, noise, amount, _timestep = self.sched.add_noise(orig=orig, timestep=timestep.item())
            amount = amount.view(amount.shape[:1])

            input_list.append([noised_orig, amount, *other_inputs])
            truth_list.append([noise, orig, timestep, *other_inputs])

        # convert [
        #     [noised_orig1, amount1, embed1],
        #     [noised_orig2, amount2, embed2]
        # ]
        # to
        # [
        #     Tensor(noised_orig1, noised_orig2),
        #     Tensor(amount1, amount2),
        #     Tensor(embed1, embed2),
        # ]
        input_list = [torch.stack(parts) for parts in zip(*input_list)]
        truth_list = [torch.stack(parts) for parts in zip(*truth_list)]
        # truth_list[2] = [timestep.item() for timestep in truth_list[2]]
        return input_list, truth_list

EDSItemType = Literal["encout", "mean+logvar", "sample"]

"""
"""
class EncoderDataset(DSBase):
    all_encouts: List[VarEncoderOutput]

    item_type: EDSItemType
    cache: LatentCache
    _clip_cache: clip_cache.ClipCache = None

    def __init__(self, *,
                 vae_net: vae.VarEncDec, vae_net_path: Path,
                 dataset: Dataset, image_dir: Path = None,
                 item_type: EDSItemType = "sample",
                 clip_model_name: clip_cache.ClipModelName = None,
                 device: str, batch_size: int):
        super().__init__(dataset)

        self.cache = LatentCache(net=vae_net, net_path=vae_net_path,
                                 dataset=dataset,
                                 batch_size=batch_size, device=device)

        self.all_encouts = self.cache.encouts_for_idxs()
        self.item_type = item_type
        if item_type not in ['encout', 'mean+logvar', 'sample']:
            raise ValueError(f"unknown {item_type=}")
        
        if image_dir is not None and clip_model_name is not None:
            self._clip_cache = \
                clip_cache.ClipCache(dataset=dataset, image_dir=image_dir, model_name=clip_model_name,
                                     device=device, batch_size=batch_size)

    def _ds_getitem(self, idx: int) -> DSItem:
        encout = self.all_encouts[idx]
        if self.item_type == 'encout':
            res = encout
        elif self.item_type == 'sample':
            res = encout.sample()
        elif self.item_type == 'mean+logvar':
            res = torch.cat([encout.mean, encout.logvar], dim=0)

        if self._clip_cache is not None:
            embed = self._clip_cache[idx]
            return ([res, embed], [res, embed])

        return (res, res)
    
