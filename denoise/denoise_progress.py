# %%
import datetime
import random
import sys
import math
from pathlib import Path
from typing import Deque, Tuple, List, Dict, Union, Callable

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

    # save inputs for each row across experiments. 
    # these have a batch dimension, are detached, and on the CPU.
    tensors_for_row: List[List[Tensor]] = None

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

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [exp.describe(include_loss=False) for exp in exps]
    
    def get_fixed_labels(self) -> List[str]:
        if self.truth_is_noise:
            res = ["original", "input noise"]
        else:
            res = ["original", "noised input"]
        return res

    def get_fixed_images(self, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        input_t, timestep, _truth_noise, truth_src_t = self._get_inputs_for_row(row)

        input_image_t = self.decoder_fn(input_t).detach()
        input_image_t.requires_grad_(False)

        input_image_anno = ""
        if timestep is not None:
            input_image_anno = f"timestep {timestep[0].item():.3f}"

        truth_image_t = self.decoder_fn(truth_src_t).detach()
        truth_image_t.requires_grad_(False)

        return [
            truth_image_t[0],
            (input_image_t[0], input_image_anno),
        ]

    def get_exp_num_cols(self) -> int:
        return len(self.get_exp_col_labels())
        
    def get_exp_col_labels(self) -> List[str]:
        if self.truth_is_noise:
            res = ["output (in-out)", "truth (noise)", "output (noise)"]
        else:
            res = ["denoised output"]
        
        # for steps in enumerate(self.steps):
        #     res.append(f"noise @{steps}")
        
        return res
    
    def get_exp_images(self, exp: Experiment, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        input, timestep, truth_noise, truth_src = self._get_inputs_for_row(row)

        input_list = [input]
        if self.use_timestep:
            # add timestep to inputs
            input_list.append(timestep)

        exp.net.eval()
        if self.truth_is_noise:
            out = exp.net(*input_list)
            exp.loss_fn(out)
            in_out = (input - out)

            res = [in_out, truth_noise, out]
        else:
            # truth is original image, input is noised image.
            out = exp.net(*input_list)
            res = [out]

        # decode all the latents, and detach them for rendering.
        res = [self.decoder_fn(latent) for latent in res]
        for image_t in res:
            image_t.detach()

        # remove batch dimension for result, returning tensors with size
        # (chan, width, height)
        res = [image_t[0] for image_t in res]

        # add train loss to the last output, which is always 'out'
        res[-1] = (res[-1], f"tloss {exp.lastepoch_train_loss:.3f}")

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

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataset = exp.train_dataloader.dataset
        first_input = self.dataset[0][0]

        if self.tensors_for_row is None:
            self.tensors_for_row = [list() for _ in range(nrows)]
    
        # pick the same sample indexes for each experiment.
        if self.dataset_idxs is None:
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            self.dataset_idxs = all_idxs[:nrows]

        self.image_size = first_input.shape[-1]
        # self.steps = [2, 5, 10, 20, 50]

    """
    Returns (input, timestep, truth_noise, truth_src). Some may be None based on 
    self.use_timestep / self.truth_is_noise.

    They are returned on self.device, and have a batch dimension of 1.

    Sizes:
    - (1, nchan, size, size)   input, truth_noise, truth_src
    - (1,)                     timestep
    """
    def _get_inputs_for_row(self, row: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # return memoized input for consistency across experiments.
        if not len(self.tensors_for_row[row]):
            ds_idx = self.dataset_idxs[row]

            # if use timestep, 
            if self.use_timestep:
                noised_input, timestep, twotruth = self.dataset[ds_idx]
            else:
                noised_input, twotruth = self.dataset[ds_idx]
                timestep = None
            
            # take one from batch.
            if self.truth_is_noise:
                truth_noise, truth_src = twotruth
            else:
                truth_src = twotruth[1]
                truth_noise = None
            
            def unsqueeze(t: Tensor) -> Tensor:
                if t is None:
                    return None

            # memoize the inputs for this row so we can use the same ones across
            # experiments.
            memo = [self._prepare(t)
                    for t in [noised_input, timestep, truth_noise, truth_src]]
            self.tensors_for_row[row] = memo
        
        #      noised_input: (nchan, size, size)
        #    timestep: (1,)                   - if use_timestep
        # truth_noise: (nchan, size, size)    - if truth_is_noise
        res = [self._to_device(t) 
               for t in self.tensors_for_row[row]]
        return res

    """
    prepare the tensor. unsqueeze, detach, and call requires_grad_(False). it
    stays on the device that it started on.
    """
    def _prepare(self, t: Tensor) -> Tensor:
        if t is None:
            return None
        t.detach()
        t.requires_grad_(False)
        return t.unsqueeze(0)
    
    def _to_device(self, t: Tensor) -> Tensor:
        if t is None:
            return None
        return t.to(self.device)

    