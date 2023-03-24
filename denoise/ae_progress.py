# %%
import datetime
import sys
import math
import random
from pathlib import Path
from typing import Deque, Tuple, List, Union, Callable
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

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
class AutoencoderProgress(image_progress.ImageProgressGenerator):
    device: str
    image_size: int

    dataset: Dataset
    dataset_idxs: List[int] = None

    def __init__(self, device: str):
        self.device = device

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataset = exp.train_dataloader.dataset
        first_input = self.dataset[0][0]

        # pick the same sample indexes for each experiment.
        if self.dataset_idxs is None:
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            self.dataset_idxs = all_idxs[:nrows]

        self.image_size = first_input.shape[-1]

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [exp.describe(include_loss=False) for exp in exps]
    
    def get_fixed_labels(self) -> List[str]:
        return ["original"]
    
    def _get_orig(self, row: int) -> Tensor:
        ds_idx = self.dataset_idxs[row]
        image = self.dataset[ds_idx][0]
        return image.detach()

    def get_fixed_images(self, row: int) -> List[Union[Tensor, Tuple[Tensor, str]]]:
        return [self._get_orig(row)]

    def get_exp_images(self, exp: Experiment, row: int) -> List[Union[Tensor, Tuple[Tensor, str]]]:
        image = self._get_orig(row)
        image = image.unsqueeze(0)
        image = image.detach().to(self.device)
    
        exp.net.eval()
        out: Tensor = exp.net(image).detach()
        out.clamp_(min=0, max=1)
        exp.net.train()

        loss = exp.loss_fn(out, image)

        return [(out[0], f"loss {loss:.3f}")]
