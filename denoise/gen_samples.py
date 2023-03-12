# %%
import sys
import math
from pathlib import Path
from typing import List, Union, Literal, Dict
from collections import deque
import re
import csv
import datetime
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
import tqdm
import argparse

import torch
from torch import Tensor
from torchvision import transforms

sys.path.append("..")
import model
import model_sd
from experiment import Experiment
import loadsave
import noised_data
import image_util

device = "cuda"

_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font_size = 10
_font: ImageFont.ImageFont = ImageFont.truetype(Roboto, _font_size)
_padding = 2
_minx = 50
_miny = 0

def _create_image(nrows: int, ncols: int, image_size: int, title_height: int):
    global _img, _draw, _font, _miny

    _miny = title_height + _padding

    width = ncols * (image_size + _padding) + _minx
    height = nrows * (image_size + _padding) + _miny
    print(f"{width=} {height=}")
    _img = Image.new("RGB", (width, height))
    _draw = ImageDraw.ImageDraw(_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("-m", "--mode", type=str, choices=["latent", "random", "steps"])
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-i", "--image_size", default=128, type=int)
    parser.add_argument("-s", "--steps", default=20, type=int, help="number of steps (for 'random', 'latent')")
    parser.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
    parser.add_argument("--amount_min", type=float, default=0.0)
    parser.add_argument("--amount_max", type=float, default=1.0)

    cfg = parser.parse_args()

    checkpoints = loadsave.find_checkpoints(only_paths=cfg.pattern)

    checkpoints = list(reversed(sorted(checkpoints, key=lambda cptup: cptup[1].saved_at)))
    if cfg.mode == "latent":
        skipped_checkpoints = [(path, exp) for path, exp in checkpoints if exp.emblen == 0]
        checkpoints = [(path, exp) for path, exp in checkpoints if exp.emblen != 0]
        print(f"skipping because mode=latent and checkpoint emblen=0; they don't use any latent representation:")
        [print(f"  {exp.label}") for path, exp in skipped_checkpoints]
        print()

    experiments = [exp for path, exp in checkpoints]
    [print(exp.label) for exp in experiments]

    image_size = cfg.image_size
    nchannels = 3
    ncols = len(checkpoints)
    padded_image_size = image_size + _padding

    # noise and amount functions.
    if cfg.noise_fn == "rand":
        noise_fn = noised_data.gen_noise_rand
    elif cfg.noise_fn == "normal":
        noise_fn = noised_data.gen_noise_normal
    else:
        raise ValueError(f"unknown {cfg.noise_fn=}")

    amount_fn = noised_data.gen_amount_range(cfg.amount_min, cfg.amount_max)

    if cfg.mode == "steps":
        steps_list = [1, 2, 5, 10, 20, 40, 80]
        row_labels = [f"{s} steps" for s in steps_list]
        nrows = len(steps_list)
        filename = cfg.output or "gen-steps.png"

        # uniform distribution for pixel space.
        inputs = noise_fn((1, nchannels, image_size, image_size)).to(device)
    elif cfg.mode == "latent":
        num_steps = cfg.steps
        nrows = 10
        row_labels = [f"latent {i}" for i in range(nrows)]
        filename = cfg.output or "gen-latent.png"
        inputs_for_emblen: Dict[int, Tensor] = dict()
    else: # random
        num_steps = cfg.steps
        nrows = 10
        row_labels = [f"rand {i}" for i in range(nrows)]
        filename = cfg.output or "gen-random.png"
        # uniform distribution for pixel space.
        inputs = noise_fn((nrows, 1, nchannels, image_size, image_size)).to(device)

    to_pil = transforms.ToPILImage("RGB")

    print(f"   ncols: {ncols}")
    print(f"   nrows: {nrows}")
    print(f"    mode: {cfg.mode}")
    print(f"filename: {filename}")


    # generate column headers
    
    col_titles, max_title_height = \
        image_util.experiment_labels(experiments, image_size + _padding, font=_font)

    _create_image(nrows=nrows, ncols=ncols, image_size=image_size, title_height=max_title_height)

    # draw row headers
    for row, row_label in enumerate(row_labels):
        xy = (0, _miny + row * padded_image_size)
        _draw.text(xy=xy, text=row_label, font=_font, fill="white")

    # draw col titles
    for col, col_title in enumerate(col_titles):
        xy = (_minx + col * padded_image_size, 0)
        _draw.text(xy=xy, text=col_title, font=_font, fill="white")

    # walk through and generate the images
    for col, (path, exp) in tqdm.tqdm(list(enumerate(checkpoints))):
        use_timestep = False
        with open(path, "rb") as cp_file:
            state_dict = torch.load(path)
            if state_dict['net_class'] == 'ConvEncDec':
                exp.net = model.ConvEncDec.new_from_state_dict(state_dict['net']).to(device)
            elif exp.net_class in ['Model', 'OptimizedModule']:
                exp.net = model_sd.Model(ch=exp.ch, out_ch=exp.out_ch, 
                                         num_res_blocks=exp.num_res_blocks,
                                         in_channels=exp.in_channels, resolution=exp.resolution).to(device)
                use_timestep = getattr(exp.net, 'use_timestep', None)
            else:
                raise Exception("unknown net_class " + state_dict['net_class'] + "/" + exp.net_class)

        if cfg.mode == "latent":
            if exp.emblen not in inputs_for_emblen:
                # gaussian distribution for latent space.
                inputs_for_emblen[exp.emblen] = torch.normal(0.0, 0.5, (nrows, 1, exp.emblen), device=device)
            inputs = inputs_for_emblen[exp.emblen]

        for row in range(nrows):
            if cfg.mode == "latent":
                steps = num_steps
                out = exp.net.decoder(inputs[row])
            elif cfg.mode == "random":
                steps = num_steps
                out = noised_data.generate(net=exp.net, num_steps=steps, size=image_size, 
                                           truth_is_noise=exp.truth_is_noise, use_timestep=use_timestep,
                                           noise_fn=noise_fn, amount_fn=amount_fn,
                                           inputs=inputs[row], 
                                           device=device)
            else:
                steps = steps_list[row]
                out = noised_data.generate(net=exp.net, num_steps=steps, size=image_size, 
                                           truth_is_noise=exp.truth_is_noise, use_timestep=use_timestep,
                                           noise_fn=noise_fn, amount_fn=amount_fn,
                                           inputs=inputs, 
                                           device=device)

            # draw this image
            out = to_pil(out[0].detach().cpu())
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)
    
    _img.save(filename)