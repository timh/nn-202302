from dataclasses import dataclass
from typing import List, Tuple, Union
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

import noisegen
from models import vae
from models.mtypes import VarEncoderOutput
from latent_cache import LatentCache

# train VAE:
# - vae wants batches of IMAGES
# = DataLoader(
#       DataSet()
#   )
#
# train denoise:
# - denoise wants batches of NOISED LATENTS
# = DataLoader(
#       batch=batch,
#       NoisedDataset(
#           EncoderDataset(
#               batch=batch,
#           )
#       )
#   )
# = EncoderDataset(
#       batch=batch, 
#       NoisedDataset()
#   )
#

# gen_samples / make_anim:
# - want direct access to backing dataset

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

        noise, amount = self.sched.noise(size=orig.shape)

        input_noised_orig = (1 - amount) * orig + amount * noise
        truth = torch.stack([noise, orig], dim=0)
        amount = amount.view(amount.shape[:1])

        return input_noised_orig, amount, truth

"""
"""
class EncoderDataset(DSBase):
    all_encouts: List[VarEncoderOutput]
    return_encout: bool

    def __init__(self, *,
                 vae_net: vae.VarEncDec, vae_net_path: Path,
                 batch_size: int,
                 base_dataset: Dataset, 
                 return_encout: bool,
                 device: str):
        super().__init__(base_dataset)
        cache = LatentCache(net=vae_net, net_path=vae_net_path,
                            dataset=base_dataset,
                            batch_size=batch_size, device=device)

        self.all_encouts = cache.encouts_for_idxs()
        self.return_encout = return_encout

    def _ds_getitem(self, idx: int) -> DSItem:
        encout = self.all_encouts[idx]
        if self.return_encout:
            return (encout, encout)
        sample = encout.sample()
        return (sample, sample)
    
"""

"""
def NoisedEncoderDataLoader(*, 
                            vae_net: vae.VarEncDec, vae_net_path: Path,
                            base_dataset: Dataset, batch_size: int,
                            noise_schedule: noisegen.NoiseSchedule,
                            shuffle: bool,
                            device: str):
    enc_ds = EncoderDataset(vae_net=vae_net, vae_net_path=vae_net_path,
                            return_encout=False,
                            batch_size=batch_size, base_dataset=base_dataset,
                            device=device)
    noised_ds = NoisedDataset(base_dataset=enc_ds, noise_schedule=noise_schedule)
    noise_dl = DataLoader(dataset=noised_ds, batch_size=batch_size, shuffle=shuffle)
    return noise_dl

