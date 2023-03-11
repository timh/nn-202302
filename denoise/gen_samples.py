# %%
import sys
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

device = "cuda"

_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font: ImageFont.ImageFont = None
_font_size = 10
_padding = 2
_minx = 50
_miny = 160

def _create_image(nrows: int, ncols: int, image_size: int):
    global _img, _draw, _font
    width = ncols * (image_size + _padding) + _minx
    height = nrows * (image_size + _padding) + _miny
    print(f"{width=} {height=}")
    _img = Image.new("RGB", (width, height))
    _draw = ImageDraw.ImageDraw(_img)
    _font = ImageFont.truetype(Roboto, _font_size)

def _build_ago_str(now: datetime.datetime, exp: Experiment):
    ago = int((now - exp.saved_at).total_seconds())
    ago_secs = ago % 60
    ago_mins = (ago // 60) % 60
    ago_hours = (ago // 3600)
    ago = deque([(val, desc) for val, desc in zip([ago_hours, ago_mins, ago_secs], ["h", "m", "s"])])
    while not ago[0][0]:
        ago.popleft()
    ago_str = " ".join([f"{val}{desc}" for val, desc in ago])
    return ago_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("-m", "--mode", type=str, choices=["latent", "random", "steps"])
    parser.add_argument("-o", "--output", default=None)

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

    image_size = experiments[0].image_size
    nchannels = experiments[0].nchannels
    image_size_list = [exp.image_size for exp in experiments]
    nchannels_list = [exp.nchannels for exp in experiments]
    if not all([onesz == image_size for onesz in image_size_list[1:]]):
        raise Exception(f"not all dirs have image_size={image_size}: {image_size_list}")
    if not all([onenc == nchannels for onenc in nchannels_list[1:]]):
        raise Exception(f"not all dirs have nchannels={nchannels}: {nchannels_list}")
    ncols = len(checkpoints)
    padded_image_size = image_size + _padding

    if cfg.mode == "steps":
        steps_list = [2, 5, 10, 20, 40, 80]
        row_labels = [f"{s} steps" for s in steps_list]
        nrows = len(steps_list)
        filename = cfg.output or "gen-steps.png"

        # uniform distribution for pixel space.
        inputs = torch.rand((1, nchannels, image_size, image_size), device=device)
    elif cfg.mode == "latent":
        num_steps = 50
        nrows = 10
        row_labels = [f"latent {i}" for i in range(nrows)]
        filename = cfg.output or "gen-latent.png"
        inputs_for_emblen: Dict[int, Tensor] = dict()
    else: # random
        num_steps = 50
        nrows = 10
        row_labels = [f"rand {i}" for i in range(nrows)]
        filename = cfg.output or "gen-random.png"
        # uniform distribution for pixel space.
        inputs = torch.rand((ncols, 1, nchannels, image_size, image_size), device=device)

    to_pil = transforms.ToPILImage("RGB")

    print(f"   ncols: {ncols}")
    print(f"   nrows: {nrows}")
    print(f"    mode: {cfg.mode}")
    print(f"filename: {filename}")

    _create_image(nrows=nrows, ncols=ncols, image_size=image_size)

    # draw row headers
    for row, row_label in enumerate(row_labels):
        xy = (0, _miny + row * padded_image_size)
        _draw.text(xy=xy, text=row_label, font=_font, fill="white")

    # draw column headers
    now = datetime.datetime.now()
    for col, (path, exp) in enumerate(checkpoints):
        # draw the title for this checkpoint
        endlr_str = f" {exp.endlr}" if exp.endlr else ""
        ago_str = _build_ago_str(now, exp)

        label_list = exp.label.split(",")
        label_len = len(label_list)
        num_splits = 5
        label_startend = [(i, i+1) for i in range(num_splits)]
        label_parts = [",".join(label_list[start:end]) for start, end in label_startend]
        label_str = "  \n".join(label_parts)
        title = (f"{label_str}\n"
                 f"startlr {exp.startlr:.1E} {endlr_str}\n"
                 f"nepoch {exp.nepochs}\n"
                 f"tloss {exp.lastepoch_train_loss:.3f}\n"
                 f"vloss {exp.lastepoch_val_loss:.3f}\n"
                 f"{ago_str} ago")

        xy = (_minx + col * padded_image_size, 0)
        _draw.text(xy=xy, text=title, font=_font, fill="white")

    for col, (path, exp) in tqdm.tqdm(list(enumerate(checkpoints))):
        nchannels = exp.nchannels

        with open(path, "rb") as cp_file:
            state_dict = torch.load(path)
            exp.net = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to(device)

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
                out = model.generate(exp=exp, num_steps=steps, size=image_size, truth_is_noise=False, input=inputs[row], device=device)
            else:
                steps = steps_list[row]
                out = model.generate(exp=exp, num_steps=steps, size=image_size, truth_is_noise=False, input=inputs, device=device)

            # draw this image
            out = to_pil(out[0].detach().cpu())
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)
    
    _img.save(filename)