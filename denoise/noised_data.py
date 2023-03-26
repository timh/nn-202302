import sys
from dataclasses import dataclass

from typing import List, Union, Tuple, Literal, Callable
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from torch import Tensor, nn

import torchvision
from torchvision import transforms

sys.path.append("..")
import train_util
import noisegen

"""
generate an image from pure noise.
"""
# TODO: update to follow the diffusion paper.
def generate(net: nn.Module, 
             inputs: Tensor,
             num_steps: int,
             truth_is_noise: bool) -> Tensor:
    timestep = torch.zeros((1, 1), device=inputs.device)
    timestep[0, 0] = 1.0 / num_steps

    net.eval()
    if num_steps <= 1:
        return inputs
    
    with torch.no_grad():
        for step in range(num_steps - 1):
            net_inputs = [inputs, timestep]

            if truth_is_noise:
                out_noise: Tensor = net(*net_inputs)
                out = inputs - out_noise
            else:
                out = net(*net_inputs)

            inputs = out
    return out

"""
output: (batch, width, height, chan)
 truth: (batch, 2, width, height, chan)
return: (1,)

"truth" actually contains both the noise that was applied and the original 
src image:
  noise = truth[:, 0, ...]
    src = truth[:, 1, ...]

"""
def twotruth_loss_fn(loss_type: Literal["l1", "l2", "mse", "distance", "mape", "rpd"] = "l1", 
                     truth_is_noise: bool = False,
                     device = "cpu") -> Callable[[Tensor, Tensor], Tensor]:
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

# def load_dataset(image_dirname: str, image_size: int, 
#                  noise_fn: Callable[[Tuple], Tensor]) -> NoisedDataset:
#     base_dataset = torchvision.datasets.ImageFolder(
#         root=image_dirname,
#         transform=transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))

#     return NoisedDataset(base_dataset=base_dataset, 
#                          noise_fn=noise_fn)

# def create_dataloaders(noised_data: NoisedDataset, batch_size: int, 
#                        minicnt: int = 0, train_split: float = 0.9, 
#                        train_all_data = True, val_all_data = True) -> Tuple[DataLoader, DataLoader]:
#     ntrain = int(len(noised_data) * train_split)

#     if (not train_all_data or not val_all_data) and minicnt == 0:
#         raise ValueError(f"minicnt must be set if {train_all_data=} and/or {val_all_data=} are False")

#     train_data = noised_data[:ntrain]
#     val_data = noised_data[ntrain:]

#     if train_all_data:
#         train_dl = DataLoader(train_data, batch_size=batch_size)
#     else:
#         train_sampler = RandomSampler(train_data, num_samples=batch_size * minicnt)
#         train_dl = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

#     if val_all_data:
#         val_dl = DataLoader(val_data, batch_size=batch_size)
#     else:
#         val_sampler = RandomSampler(val_data, num_samples=batch_size * minicnt)
#         val_dl = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)

#     return train_dl, val_dl


