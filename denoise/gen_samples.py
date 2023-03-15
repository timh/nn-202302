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
import noised_data
import image_util
import model_util, dn_util

device = "cuda"

_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font: ImageFont.ImageFont = None
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
    parser.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
    parser.add_argument("-m", "--mode", default="steps", choices=["latent", "random", "steps", "interp"])
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-s", "--steps", default=20, type=int, help="number of steps (for 'random', 'latent')")
    parser.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
    parser.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
    parser.add_argument("-i", "--output_image_size", type=int, default=None)
    parser.add_argument("--amount_min", type=float, default=0.0)
    parser.add_argument("--amount_max", type=float, default=1.0)

    cfg = parser.parse_args()
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern, re.DOTALL)

    checkpoints = model_util.find_checkpoints(only_paths=cfg.pattern, attr_matchers=cfg.attribute_matchers)
    checkpoints = list(reversed(sorted(checkpoints, key=lambda cp_tuple: cp_tuple[1].saved_at)))
    experiments = [exp for path, exp in checkpoints]
    [print(exp.label) for exp in experiments]

    image_size = max([exp.net_image_size for exp in experiments])
    output_image_size = cfg.output_image_size or image_size
    nchannels = 3
    ncols = len(checkpoints)
    padded_image_size = output_image_size + _padding

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
        inputs_for_latent: Dict[List[int], Tensor] = dict()

    elif cfg.mode == "interp":
        nrows = 10 + 2
        row_labels = ["src 0", "src 1"]
        row_labels.extend([f"latent {i}" for i in range(nrows - 2)])
        filename = cfg.output or "gen-interp.png"

        img_idxs = None

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


    font_size = max(10, output_image_size // 20)
    _font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    # generate column headers
    max_width = output_image_size + _padding
    col_titles, max_title_height = \
        image_util.experiment_labels(experiments, max_width=max_width, font=_font)

    _create_image(nrows=nrows, ncols=ncols, image_size=output_image_size, title_height=max_title_height)

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
            model_dict = torch.load(path)
            exp.net = dn_util.load_model(model_dict).to(device)

        if cfg.mode == "latent":
            if not isinstance(exp.net, model.ConvEncDec):
                raise NotImplementedError("not implemented for != ConvEncDec")
                # gaussian distribution for latent space.

            ldkey = str(exp.net_latent_dim)
            if ldkey not in inputs_for_latent:
                inputs_for_latent[ldkey] = torch.normal(0.0, 0.5, (nrows, 1, *exp.net_latent_dim), device=device)
            inputs = inputs_for_latent[ldkey]


        elif cfg.mode == "interp":
            _, val_dl = dn_util.get_dataloaders(disable_noise=True, 
                                                image_size=exp.net_image_size, 
                                                image_dir=cfg.image_dir, batch_size=1)
            dataset = val_dl.dataset
            if img_idxs is None:
                img_idxs = [i.item() for i in torch.randint(0, len(dataset), (2,))]
            img_tensors = [dataset[img_idx][1].unsqueeze(0).to(device) for img_idx in img_idxs]

            latent0 = exp.net.encoder(img_tensors[0])
            latent1 = exp.net.encoder(img_tensors[1])
            # print(f"{latent0.mean()=} {latent0.std()=}")

        for row in range(nrows):
            if cfg.mode == "latent":
                try:
                    if isinstance(exp.net, model.ConvEncDec):
                        out = exp.net.decoder(inputs[row])
                    elif isinstance(exp.net, model_sd.Model):
                        out = exp.net.decoder(inputs[row])
                except Exception as e:
                    print(f"error processing {path}:", file=sys.stderr)
                    raise e

            elif cfg.mode == "interp":
                if row == 0:
                    out = img_tensors[0]
                elif row == nrows - 1:
                    out = img_tensors[1]
                else:
                    imgidx = row - 1
                    nimages = nrows - 2
                    latent0_part = (nimages - imgidx - 1) / nimages
                    latent1_part = (imgidx + 1) / nimages
                    latent_in = latent0 * latent0_part + latent1 * latent1_part
                    out = exp.net.decoder(latent_in)

            elif cfg.mode == "random":
                out = noised_data.generate(net=exp.net, num_steps=num_steps, size=exp.net.image_size, 
                                           truth_is_noise=False, use_timestep=use_timestep,
                                           noise_fn=noise_fn, amount_fn=amount_fn,
                                           inputs=inputs[row], 
                                           device=device)
            else:
                out = noised_data.generate(net=exp.net, num_steps=steps_list[row], size=exp.net.image_size, 
                                           truth_is_noise=exp.truth_is_noise, use_timestep=use_timestep,
                                           noise_fn=noise_fn, amount_fn=amount_fn,
                                           inputs=inputs, 
                                           device=device)

            # make the image tensor into an image then make all images the same
            # (output) size.
            out: Image.Image = to_pil(out[0].detach().cpu())
            if exp.net.image_size != output_image_size:
                out = out.resize((output_image_size, output_image_size), resample=Image.Resampling.BICUBIC)

            # draw this image
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)
    
    _img.save(filename)