# %%
import datetime
import random
import sys
import math
from pathlib import Path
from typing import Deque, Tuple, List, Union, Callable
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
    decoder_fn: Callable[[Tensor], Tensor]

    steps: List[int]

    dataset_idxs: List[int] = None

    def __init__(self, truth_is_noise: bool, use_timestep: bool, 
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tensor],
                 decoder_fn: Callable[[Tensor], Tensor],
                 device: str):
        self.truth_is_noise = truth_is_noise
        self.use_timestep = use_timestep
        self.noise_fn = noise_fn
        self.amount_fn = amount_fn
        self.device = device
        self.decoder_fn = decoder_fn
        self._normalize = norm
    
    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [exp.describe(include_loss=False) for exp in exps]
    
    def get_col_headers(self) -> List[str]:
        return ["original"]

    def get_col_header_images(self, row: int) -> List[Tensor]:
        _input, _timestep, _truth_noise, truth_src = self._pick_image(row)
        image_t = self.decoder_fn(truth_src).detach()
        return image_t

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataset = exp.train_dataloader.dataset
        first_input = self.dataset[0][0]

        # pick the same sample indexes for each experiment.
        if self.dataset_idxs is None:
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            self.dataset_idxs = all_idxs[:nrows]

        self.image_size = first_input.shape[-1]
        self.steps = [2, 5, 10, 20, 50]

    def get_num_images_per_exp(self) -> int:
        return len(self.get_image_labels())
        
    def get_image_labels(self) -> List[str]:
        if self.truth_is_noise:
            res = ["noised input", "output (in-out)", "truth (noise)", "output (noise)"]
        else:
            res = ["noised input", "output (denoised)"]
        
        # for steps in enumerate(self.steps):
        #     res.append(f"noise @{steps}")
        
        return res
    
    def get_images(self, exp: Experiment, row: int) -> List[Tensor]:
        # def decode(t: Tensor) -> Tensor:
        #     if t is not None:
        #         return self.decoder_fn(t).detach()
        #     return None

        input, timestep, truth_noise, truth_src = self._pick_image(row)

        input_list = [input]
        if self.use_timestep:
            # add timestep to inputs
            input_list.append(timestep)

        if self.truth_is_noise:
            out = exp.net(*input_list)
            in_out = (input - out)
            # in_out = (input - out).clamp(min=0, max=1)
            # truth_noise = self._normalize(truth_noise)
            # out = self._normalize(out)

            res = [input, in_out, truth_noise, out]
        else:
            # truth is original image, input is noised image.
            out = exp.net(*input_list)
            res = [input, out]

        res = [self.decoder_fn(latent).detach() for latent in res]
        res = [image[0] for image in res]
        # input = decode(input)
        # truth_noise = decode(truth_noise)
        # truth_src = decode(truth_src)

        # # then comes images based on noise imagination
        # noise_in = self.noise_fn((1, 3, self.image_size, self.image_size)).to(self.device)

        # for i, steps in enumerate(self.steps):
        #     out = noised_data.generate(net=exp.net, 
        #                                num_steps=steps, size=self.image_size, 
        #                                truth_is_noise=self.truth_is_noise,
        #                                use_timestep=self.use_timestep,
        #                                inputs=noise_in,
        #                                noise_fn=self.noise_fn, amount_fn=self.amount_fn,
        #                                device=self.device)
        #     out.clamp_(min=0, max=1)
        #     res.append(out[0])
        
        return res


    """
    Returns (input, timestep, truth_noise, truth_src). Some may be None based on 
    self.use_timestep / self.truth_is_noise.

    They are returned on self.device, and have a batch dimension of 1.

    Sizes:
    - (1, nchan, size, size)   input, truth_noise, truth_src
    - (1,)                     timestep
    """
    def _pick_image(self, row: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ds_idx = self.dataset_idxs[row]

        # if use timestep, 
        if self.use_timestep:
            noised_input, timestep, twotruth = \
                [t.detach().to(self.device) for t in self.dataset[ds_idx]]
            timestep = timestep.unsqueeze(0)
        else:
            noised_input, twotruth = \
                [t.detach().to(self.device) for t in self.dataset[ds_idx]]
            timestep = None
        
        noised_input = noised_input.unsqueeze(0)

        # take one from batch.
        if self.truth_is_noise:
            truth_noise, truth_src = [t.unsqueeze(0) for t in twotruth]
        else:
            truth_src = twotruth[1].unsqueeze(0)
            truth_noise = None

        #      noised_input: (nchan, size, size)
        #    timestep: (1,)                   - if use_timestep
        # truth_noise: (nchan, size, size)    - if truth_is_noise
        return noised_input, timestep, truth_noise, truth_src

