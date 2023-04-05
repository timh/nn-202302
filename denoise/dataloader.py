from dataclasses import dataclass
from typing import List, Tuple, Union, Literal
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

import noisegen
from models import vae
from models.mtypes import VarEncoderOutput
from latent_cache import LatentCache

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

"""
NoisedDataset: take a backing dataset and apply noise to it.
"""
class NoisedDataset(DSBase):
    sched: noisegen.NoiseSchedule
    def __init__(self, *,
                 base_dataset: Dataset, noise_schedule: noisegen.NoiseSchedule):
        super().__init__(base_dataset)
        self.sched = noise_schedule
    
    def _ds_getitem(self, idx: int) -> DSItem:
        value = self.base_dataset[idx]
        if isinstance(value, tuple) or isinstance(value, list):
            orig = value[0]
        else:
            orig = value
        
        noised_orig, noise, amount, timestep = self.sched.add_noise(orig=orig)

        truth = torch.stack([noise, orig], dim=0)
        amount = amount.view(amount.shape[:1])

        return [noised_orig, amount], truth, timestep

EDSItemType = Literal["encout", "mean+logvar", "sample"]

"""
"""
class EncoderDataset(DSBase):
    all_encouts: List[VarEncoderOutput]
    item_type: EDSItemType
    cache: LatentCache

    def __init__(self, *,
                 vae_net: vae.VarEncDec, vae_net_path: Path,
                 batch_size: int,
                 base_dataset: Dataset, 
                 item_type: EDSItemType = "sample",
                 device: str):
        super().__init__(base_dataset)
        self.cache = LatentCache(net=vae_net, net_path=vae_net_path,
                                 dataset=base_dataset,
                                 batch_size=batch_size, device=device)

        self.all_encouts = self.cache.encouts_for_idxs()
        self.item_type = item_type
        if item_type not in ['encout', 'mean+logvar', 'sample']:
            raise ValueError(f"unknown {item_type=}")

    def _ds_getitem(self, idx: int) -> DSItem:
        encout = self.all_encouts[idx]
        if self.item_type == 'encout':
            res = encout
        elif self.item_type == 'sample':
            res = encout.sample()
        elif self.item_type == 'mean+logvar':
            res = torch.cat([encout.mean, encout.logvar], dim=0)

        return (res, res)
    
"""

"""
def NoisedEncoderDataLoader(*, 
                            vae_net: vae.VarEncDec, vae_net_path: Path,
                            base_dataset: Dataset, 
                            batch_size: int, enc_batch_size: int = None,
                            noise_schedule: noisegen.NoiseSchedule,
                            eds_item_type: EDSItemType = 'sample',
                            shuffle: bool,
                            device: str):
    enc_batch_size = enc_batch_size or batch_size

    enc_ds = EncoderDataset(vae_net=vae_net, vae_net_path=vae_net_path,
                            item_type=eds_item_type,
                            batch_size=enc_batch_size, base_dataset=base_dataset,
                            device=device)
    noised_ds = NoisedDataset(base_dataset=enc_ds, noise_schedule=noise_schedule)
    noise_dl = DataLoader(dataset=noised_ds, batch_size=batch_size, shuffle=shuffle)
    return noise_dl

