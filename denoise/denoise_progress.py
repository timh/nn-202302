# %%
import datetime
import sys
import math
from pathlib import Path
from typing import Deque, Tuple, List, Callable
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch
from torch import Tensor
from torchvision import transforms

import noised_data
sys.path.append("..")
from experiment import Experiment
import image_util
from loggers import image_progress

def norm(inputs: Tensor) -> Tensor:
    gray = inputs.mean(dim=0)
    out = torch.zeros_like(inputs)

    out[0] = -(torch.minimum(gray, torch.tensor(0)).clamp(min=-1))
    out[1] = inputs[1].clamp(min=0, max=1)
    out[2] = torch.maximum(gray, torch.tensor(0)).clamp(max=1)
    return out

"""
always  noised_input    (noise + src)
always  truth_src       (truth)
 maybe  input - output  (noise + src) - (predicted noise)
 maybe  truth_noise     (just noise)
always  output          (either predicted noise or denoised src)
"""
class DenoiseProgress(image_progress.ImageProgressGenerator):
    truth_is_noise: bool
    use_timestep: bool
    noise_fn: Callable[[Tuple], Tensor] = None
    amount_fn: Callable[[], Tensor] = None
    device: str
    image_size: int

    _steps: List[int]

    _sample_idxs: List[int] = None

    def __init__(self, truth_is_noise: bool, use_timestep: bool, disable_noise: bool,
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tensor],
                 device: str):
        self.truth_is_noise = truth_is_noise
        self.use_timestep = use_timestep
        self.disable_noise = disable_noise
        self.noise_fn = noise_fn
        self.amount_fn = amount_fn
        self.device = device
        # self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self._normalize = norm

    def on_exp_start(self, exp: Experiment, nrows: int):
        dataset = exp.val_dataloader.dataset
        first_input = dataset[0][0]

        # pick the same sample indexes for each experiment.
        if self._sample_idxs is None:
            self._sample_idxs = [i.item() for i in torch.randint(0, len(dataset), (nrows,))]

        self.image_size = first_input.shape[-1]
        self._steps = [2, 5, 10, 20, 50]

    def get_col_labels(self) -> List[str]:
        if self.disable_noise:
            return ["src", "output"]

        if self.truth_is_noise:
            res = ["noised input", "truth (src)", "output (in-out)", "truth (noise)", "output (noise)"]
        else:
            res = ["noised input", "truth (src)", "output (denoised)"]
        
        for steps in enumerate(self._steps):
            res.append(f"noise @{steps}")
        
        return res
    
    def get_images(self, exp: Experiment, epoch: int, row: int) -> List[Tensor]:
        input, timestep, truth_noise, truth_src = self._pick_image(exp, row)
        input_list = [input.unsqueeze(0).to(self.device)]

        # first, the outputs based on the dataset.
        if self.disable_noise:
            out = exp.net(*input_list)
            out = out.clamp(min=0, max=1)
            return [truth_src, out[0]]

        if self.use_timestep:
            # add timestep to inputs
            input_list.append(timestep.unsqueeze(0).to(self.device))

        if self.truth_is_noise:
            out = exp.net(*input_list)
            in_out = (input - out).clamp(min=0, max=1)
            truth_noise = self._normalize(truth_noise)
            out = self._normalize(out)

            res = [input, truth_src, in_out, truth_noise, out[0]]
        else:
            # truth is original image, input is noised image.
            out = exp.net(*input_list)
            res = [input, truth_src, out[0]]

        # then comes images based on noise imagination
        noise_in = self.noise_fn((1, 3, self.image_size, self.image_size)).to(self.device)

        for i, steps in enumerate(self._steps):
            out = noised_data.generate(net=exp.net, 
                                       num_steps=steps, size=self.image_size, 
                                       truth_is_noise=self.truth_is_noise,
                                       use_timestep=self.use_timestep,
                                       inputs=noise_in,
                                       noise_fn=self.noise_fn, amount_fn=self.amount_fn,
                                       device=self.device)
            out.clamp_(min=0, max=1)
            res.append(out[0])
        
        return res


    """
    Returns (input, timestep, truth_noise, truth_src). Some may be None based on 
    self.use_timestep / self.truth_is_noise.

    They have not had .to called, and have no batch dimension.

    Sizes:
    - (nchan, size, size)   input, truth_noise, truth_src
    - (1,)                  timesteps
    """
    def _pick_image(self, exp: Experiment, row: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dataset = exp.val_dataloader.dataset
        sample_idx = self._sample_idxs[row]

        # if use timestep, 
        if self.disable_noise:
            input_src, truth_src = dataset[sample_idx]
            return input_src, None, None, truth_src

        if self.use_timestep:
            noised_input, timesteps, twotruth = dataset[sample_idx]
        else:
            noised_input, twotruth = dataset[sample_idx]
            timesteps = None

        # take one from batch.
        if self.truth_is_noise:
            truth_noise, truth_src = twotruth
        else:
            truth_src = twotruth[1]
            truth_noise = None
        
        #      noised_input: (nchan, size, size)
        #   timesteps: (1,)                   - if use_timestep
        # truth_noise: (nchan, size, size)    - if truth_is_noise
        return noised_input, timesteps, truth_noise, truth_src

