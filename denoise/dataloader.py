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
import clip_cache

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
    split_idx = torch.randint(low=0, high=len(dataset), size=(1,)).item()
    train_ds = DSSubset(dataset, start=0, end=split_idx)
    val_ds = DSSubset(dataset, start=split_idx, end=len(dataset))
    return train_ds, val_ds

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
        value, _truth = self.base_dataset[idx]
        if isinstance(value, tuple) or isinstance(value, list):
            orig = value[0]
            # e.g., clip_embed
            other_inputs = value[1:]
        else:
            orig = value
            other_inputs = []
        
        noised_orig, noise, amount, timestep = self.sched.add_noise(orig=orig)
        amount = amount.view(amount.shape[:1])

        return [noised_orig, amount, *other_inputs], [noise, orig, timestep, *other_inputs]

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
    
"""

"""
class NoisedEncoderDataLoader(DataLoader):
    encoder_ds: EncoderDataset
    noised_ds: NoisedDataset

    def __init__(self, *,
                 vae_net: vae.VarEncDec, vae_net_path: Path,
                 dataset: Dataset, image_dir: Path,
                 batch_size: int, enc_batch_size: int = None,
                 noise_schedule: noisegen.NoiseSchedule,
                 eds_item_type: EDSItemType = 'sample',
                 shuffle: bool,
                 clip_model_name: clip_cache.ClipModelName = None,
                 device: str):
        enc_batch_size = enc_batch_size or batch_size

        self.encoder_ds = \
            EncoderDataset(vae_net=vae_net, vae_net_path=vae_net_path,
                           dataset=dataset, image_dir=image_dir,
                           item_type=eds_item_type,
                           clip_model_name=clip_model_name,
                           device=device, batch_size=enc_batch_size)
        self.noised_ds = \
            NoisedDataset(base_dataset=self.encoder_ds, noise_schedule=noise_schedule)

        super().__init__(dataset=self.noised_ds, batch_size=batch_size, shuffle=shuffle)
    
    def get_clip_emblen(self) -> int:
        if self.encoder_ds.clip_cache is None:
            raise ValueError(f"called without clip embedding enabled")
        return self.encoder_ds.clip_cache[0].shape[0]


