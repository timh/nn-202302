import sys
from dataclasses import dataclass

from typing import List, Union, Tuple, Literal, Callable
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

sys.path.append("..")
import trainer
import train_util

def gen_noise_rand(size: Tuple) -> Tensor:
    return torch.rand(size=size) - 0.5

def gen_noise_normal(size: Tuple) -> Tensor:
    return torch.normal(mean=0, std=0.5, size=size)

default_gen_noise = gen_noise_rand

def gen_amount_01() -> Tensor:
    return torch.rand(1)

def gen_amount_range(amount_min: Tensor, amount_max: Tensor) -> Callable[[], Tensor]:
    def fn() -> Tensor:
        amount = torch.rand(1)
        diff = amount_max - amount_min
        amount = amount * diff + amount_min
        return amount
    return fn

default_gen_amount = gen_amount_01
    
"""
generate an image from pure noise.
"""
def generate(net: nn.Module, 
             inputs: Tensor,
             num_steps: int,
             truth_is_noise: bool, use_timestep: bool) -> Tensor:
    if use_timestep:
        timestep = torch.zeros((1, 1), device=inputs.device)
        timestep[0, 0] = 1.0 / num_steps

    net.eval()
    if num_steps <= 1:
        return inputs
    
    # TODO: this doesn't do the right math for use_timestep, i don't think.
    with torch.no_grad():
        for step in range(num_steps - 1):
            if use_timestep:
                net_inputs = [inputs, timestep]
            else:
                net_inputs = [inputs]

            if truth_is_noise:
                out_noise: Tensor = net(*net_inputs)
                if use_timestep:
                    out = inputs - out_noise
                else:
                    keep_noise_amount = (step + 1) / num_steps
                    out = inputs - keep_noise_amount * out_noise
            else:
                out_denoised: Tensor = net(*net_inputs)
                if use_timestep:
                    out = out_denoised
                else:
                    keep_output = (step + 1) / num_steps
                    out = (out_denoised * keep_output) + (out * (1 - keep_output))

            inputs = out
    return out

@dataclass(kw_only=True)
class _Iter:
    ndataset: 'NoisedDataset'
    _start: int
    _end: int
    _idx: int = 0

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            raise Exception("slice not supported")

        ds_value = self.ndataset.dataset[self._start + idx]
        if isinstance(ds_value, tuple) or isinstance(ds_value, list):
            orig = ds_value[0]
        else:
            orig = ds_value

        noise: Tensor = None
        amount: Tensor = None
        if self.ndataset.use_noise_steps:
            nsteps = torch.randint(low=1, high=self.ndataset.use_noise_steps, size=(1,))[0].item()
            noise = torch.zeros_like(orig)
            for _ in range(nsteps):
                noise += self.ndataset.noise_fn(orig.shape) * torch.tensor(1.0 / nsteps)
            amount = torch.tensor(nsteps / self.ndataset.use_noise_steps)
        elif self.ndataset.use_timestep:
            amount = self.ndataset.amount_fn()
            noise = self.ndataset.noise_fn(orig.shape) * amount
        else:
            noise = self.ndataset.noise_fn(orig.shape)

        input_noised_orig = orig + noise
        # input_noised_orig.clamp_(min=0, max=1)
        truth = torch.stack([noise, orig], dim=0)

        if amount is not None:
            return input_noised_orig, amount, truth
        
        return input_noised_orig, truth
    
    def __next__(self) -> Tuple[Tensor, Tensor]:
        res = self[self._idx]
        self._idx += 1
        return res
    
    def __len__(self) -> int:
        return self._end - self._start

class NoisedDataset:
    dataset: Dataset
    use_timestep: bool
    noise_fn: Callable[[Tuple], Tensor]

    def __init__(self, *,
                 base_dataset: Dataset, use_timestep: bool, 
                 use_noise_steps: int = 0,
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tensor]):
        self.dataset = base_dataset
        self.use_timestep = use_timestep
        self.use_noise_steps = use_noise_steps
        self.noise_fn = noise_fn
        self.amount_fn = amount_fn

        if use_noise_steps and amount_fn is not None:
            raise ValueError(f"{use_noise_steps=} can't be used with amount_fn != None")
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return _Iter(ndataset=self, _start=0, _end=len(self.dataset))
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            start, end, _stride = idx.indices(len(self))
            return _Iter(ndataset=self, _start=start, _end=end)
        return _Iter(ndataset=self, _start=0, _end=len(self))[idx]



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
    loss_fn = train_util.get_loss_fn(loss_type, device)

    def fn(output: Tensor, truth: Tensor) -> Tensor:
        batch, ntruth, chan, width, height = truth.shape

        if truth_is_noise:
            truth = truth[:, 0, :, :, :]
        else:
            truth = truth[:, 1, :, :, :]
        truth = truth.view(batch, chan, width, height)
        return loss_fn(output, truth)
    return fn

def load_dataset(image_dirname: str, image_size: int, use_timestep: bool,
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tuple]) -> NoisedDataset:
    base_dataset = torchvision.datasets.ImageFolder(
        root=image_dirname,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    return NoisedDataset(base_dataset=base_dataset, use_timestep=use_timestep,
                         noise_fn=noise_fn, amount_fn=amount_fn)

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

