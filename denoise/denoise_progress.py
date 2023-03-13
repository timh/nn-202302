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

import model
import noised_data
sys.path.append("..")
from experiment import Experiment
import image_util

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
class DenoiseProgress:
    truth_is_noise: bool
    use_timestep: bool
    noise_fn: Callable[[Tuple], Tensor] = None
    amount_fn: Callable[[], Tensor] = None
    device: str

    image_size: int

    _steps: List[int]

    _margin_x = 2
    _margin_y = 2
    _margin_noise_y = 20
    _spacing_x: int
    _spacing_y: int
    _ymin_noise: int
    _sample_idxs: List[int]

    _path: Path
    _img: Image.Image
    _draw: ImageDraw.ImageDraw
    _font: ImageFont.ImageFont
    _to_image: transforms.ToPILImage

    def __init__(self, truth_is_noise: bool, use_timestep: bool, disable_noise: bool,
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tensor],
                 device: str):
        self.truth_is_noise = truth_is_noise
        self.use_timestep = use_timestep
        self.disable_noise = disable_noise
        self.noise_fn = noise_fn
        self.amount_fn = amount_fn
        self.device = device
        self._to_image = transforms.ToPILImage("RGB")
        # self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self._normalize = norm

    def on_exp_start(self, exp: Experiment, ncols: int, path: Path):
        dataset = exp.val_dataloader.dataset
        first_input = dataset[0][0]

        # pick the same sample indexes for each experiment.
        self._sample_idxs = [i.item() for i in torch.randint(0, len(dataset), (ncols,))]

        self._path = path

        self.image_size = first_input.shape[-1]
        self._spacing_y = (self.image_size + self._margin_y)
        self._spacing_x = (self.image_size + self._margin_x)
        self._ymin_out = self._spacing_y * 2                            # 1 input + 1 truth = 2
        self._ymin_noise = self._spacing_y * 3 + self._margin_noise_y   # 1 input + 1 truth + 1 out = 3
        self._steps = [2, 5, 10, 20, 50]

        # determine output image dimensions.
        if self.disable_noise:
            # input, src, output
            nrows_top = 3
        elif self.truth_is_noise:
            # noised_input, noise_truth, src, input - output, output
            nrows_top = 5
        else:
            # noised_input, src, output
            nrows_top = 3
        
        top_height = self._spacing_y * nrows_top
        if self.disable_noise:
            bot_height = 0
        else:
            bot_height = self._spacing_y * len(self._steps)
        self._ymin_noise = top_height + self._margin_noise_y
        img_height = top_height + self._margin_noise_y + bot_height
        img_width = self._spacing_x * ncols

        font_size = math.ceil(self.image_size / 10)   # heuristic
        padding = 4
        self._font = ImageFont.truetype(Roboto, font_size)
        label_list, label_height = image_util.experiment_labels([exp], self.image_size * 4, self._font)
        label = label_list[0]

        img_height += label_height + padding * 2

        self._img = Image.new("RGB", (img_width, img_height))
        self._draw = ImageDraw.ImageDraw(self._img)
        self._title_xy = (10, int(img_height - label_height - padding))
        self._draw.text(xy=self._title_xy, text=label, font=self._font, fill='white')
        self._img.save(self._path)

    """
     sample_idx: if None, pick randomly from the dataset. otherwise, use given.
       noise_in: if None, generate random noise for the imagination part.
    timestep_in: if None, and use_timestep, generate a random timestep.
    """
    def add_column(self, exp: Experiment, epoch: int, col: int, 
                   sample_idx: int = None,
                   noise_in: Tensor = None):
        start = datetime.datetime.now()

        # update the title
        label_list, _ = image_util.experiment_labels([exp], self.image_size * 4, self._font)
        label = label_list[0]
        box = (self._title_xy[0], self._title_xy[1], self._img.width, self._img.height)
        self._draw.rectangle(xy=box, fill='black')
        self._draw.text(xy=self._title_xy, text=label, font=self._font, fill='white')

        input, timestep, truth_noise, truth_src = self._pick_image(exp, col, sample_idx)
        input_list = [input.unsqueeze(0).to(self.device)]
        if self.disable_noise:
            noise_str = ""
        elif self.use_timestep:
            input_list.append(timestep.unsqueeze(0).to(self.device))
            noise_str = f"@{timestep[0]:.2f}"
        else:
            noise_str = ""
        
        x = col * self._spacing_x
        if self.disable_noise:
            out = exp.net(*input_list)
            out = out[0].detach().cpu()

            rows = [(f"true input\n"              , input),
                    ( "truth (src)"               , truth_src),
                    ( "output"                    , out.clamp(min=0, max=1))]
            for row, (title, tensor) in enumerate(rows):
                y = row * self._spacing_y
                img = self._to_image(tensor)
                self._img.paste(img, (x, y))
                self._drawtext((x, y), title)

        elif self.truth_is_noise:
            out = exp.net(*input_list)
            out = out[0].detach().cpu()

            tn = self._normalize(truth_noise)
            rows = [(f"noised input\n{noise_str}" , input),
                    ( "truth (src)"               , truth_src),
                    ( "output\n(in - out)"        , (input - out).clamp(min=0, max=1)),
                    (f"truth\n(noise {noise_str})", self._normalize(truth_noise)),
                    (f"output\n(pred. noise)"     , self._normalize(out))]
            for row, (title, tensor) in enumerate(rows):
                y = row * self._spacing_y
                img = self._to_image(tensor)
                self._img.paste(img, (x, y))
                self._drawtext((x, y), title)
        else:
            out = exp.net(*input_list)
            out = out[0].detach().cpu()

            rows = [(f"noised input {noise_str}", input),
                    ( "truth (src)"             , truth_src),
                    ( "output (denoised)"       , out)]
            for row, (title, tensor) in enumerate(rows):
                y = row * self._spacing_y
                img = self._to_image(tensor)
                self._img.paste(img, (x, y))
                self._drawtext((x, y), title)


        # then comes some noise imagination
        if not self.disable_noise:
            if noise_in is None:
                noise_in = self.noise_fn((1, 3, self.image_size, self.image_size)).to(self.device)

            for i, steps in enumerate(self._steps):
                y = self._ymin_noise + (i * self._spacing_y)
                out = noised_data.generate(net=exp.net, 
                                        num_steps=steps, size=self.image_size, 
                                        truth_is_noise=self.truth_is_noise,
                                        use_timestep=self.use_timestep,
                                        inputs=noise_in,
                                        noise_fn=self.noise_fn, amount_fn=self.amount_fn,
                                        device=self.device)
                out.clamp_(min=0, max=1)
                out = out[0].detach().cpu()
                self._img.paste(self._to_image(out), (x, y))
                text = f"noise @{steps}"
                self._drawtext(xy=(x, y), text=text)

        self._img.save(self._path)
        end = datetime.datetime.now()
        elapsed = (end - start).total_seconds()
        print(f"  updated progress image in {elapsed:.2f}s")

    """
    Returns input, timesteps, truth_noise, truth_src. Some may be None
    based on self.use_timestep / self.truth_is_noise.
    They have not had .to called, and have no batch dimension.

    Sizes:
    - (nchan, size, size)   input, truth_noise, truth_src
    - (1,)                  timesteps
    """
    def _pick_image(self, exp: Experiment, col: int, sample_idx: int = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dataset = exp.val_dataloader.dataset
        if sample_idx is None:
            sample_idx = self._sample_idxs[col]
            # sample_idx = torch.randint(0, len(dataset), (1,)).item()

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

    def _drawtext(self, xy: Tuple[int, int], text: str):
        self._draw.text(xy=xy, text=text, font=self._font, fill="black")
        self._draw.text(xy=(xy[0] + 2, xy[1] + 2), text=text, font=self._font, fill="white")

