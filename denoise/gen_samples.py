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
import model_sd
import model_new
from experiment import Experiment
import noised_data
import image_util
import model_util
import dn_util
import cmdline


_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font: ImageFont.ImageFont = None
_padding = 2
_minx = 50
_miny = 0

class Config(cmdline.QueryConfig):
    mode: str
    output: str
    steps: int
    image_dir: str
    noise_fn: str
    output_image_size: int
    amount_min: float
    amount_max: float


    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="steps", choices=["latent", "random", "steps", "interp", "autoencode"])
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--steps", default=20, type=int, help="number of steps (for 'random', 'latent')")
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--amount_min", type=float, default=0.0)
        self.add_argument("--amount_max", type=float, default=1.0)


def _create_image(nrows: int, ncols: int, image_size: int, title_height: int):
    global _img, _draw, _font, _miny

    _miny = title_height + _padding

    width = ncols * (image_size + _padding) + _minx
    height = nrows * (image_size + _padding) + _miny
    print(f"{width=} {height=}")
    _img = Image.new("RGB", (width, height))
    _draw = ImageDraw.ImageDraw(_img)

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    checkpoints = cfg.list_checkpoints()
    exps = [exp for _path, exp in checkpoints]

    image_size = max([exp.net_image_size for exp in exps])
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
        inputs = noise_fn((1, nchannels, image_size, image_size)).to(cfg.device)

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
        inputs = noise_fn((nrows, 1, nchannels, image_size, image_size)).to(cfg.device)

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
        image_util.fit_exp_descrs(exps, max_width=max_width, font=_font)

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
            try:
                model_dict = torch.load(path)
                exp.net = dn_util.load_model(model_dict).to(cfg.device)
                if isinstance(exp.net, model_new.VarEncDec):
                    encoder_fn = exp.net.encode
                    decoder_fn = exp.net.decode
                else:
                    encoder_fn = exp.net.encoder
                    decoder_fn = exp.net.decoder
            except Exception as e:
                print(f"error processing {path}:", file=sys.stderr)
                raise e

        if cfg.mode == "latent":
            if isinstance(exp.net, model_new.VarEncDec):
                latent_dim = exp.net.encoder_out_dim
            else:
                raise NotImplementedError(f"not implemented for {type(exp.net)}")
                # gaussian distribution for latent space.

            ldkey = str(latent_dim)
            if ldkey not in inputs_for_latent:
                inputs_for_latent[ldkey] = torch.normal(0.0, 0.5, (nrows, 1, *latent_dim), device=cfg.device)
            inputs = inputs_for_latent[ldkey]

        elif cfg.mode == "interp":
            if isinstance(exp.net, model_new.VarEncDec):
                dataloader, _ = image_util.get_dataloaders(image_size=exp.net_image_size,
                                                           image_dir=cfg.image_dir, batch_size=1)
            else:
                dataloader, _ = dn_util.get_dataloaders(disable_noise=True, 
                                                        image_size=exp.net_image_size, 
                                                        image_dir=cfg.image_dir, batch_size=1)
            dataset = dataloader.dataset
            if img_idxs is None:
                img_idxs = [i.item() for i in torch.randint(0, len(dataset), (2,))]
            img_tensors = [dataset[img_idx][1].unsqueeze(0).to(cfg.device) for img_idx in img_idxs]

            latent0 = encoder_fn(img_tensors[0])
            latent1 = encoder_fn(img_tensors[1])
            # print(f"{latent0.mean()=} {latent0.std()=}")
        
        for row in range(nrows):
            if cfg.mode == "latent":
                try:
                    out = decoder_fn(inputs[row])
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
                    out = decoder_fn(latent_in)

            elif cfg.mode == "random":
                out = noised_data.generate(net=exp.net, num_steps=num_steps,
                                           truth_is_noise=False, use_timestep=use_timestep,
                                           inputs=inputs[row])
            else:
                truth_is_noise = getattr(exp, 'truth_is_noise', False)
                out = noised_data.generate(net=exp.net, num_steps=steps_list[row],
                                           truth_is_noise=truth_is_noise, use_timestep=use_timestep,
                                           inputs=inputs)

            # make the image tensor into an image then make all images the same
            # (output) size.
            out: Image.Image = to_pil(out[0].detach().cpu())
            if exp.net.image_size != output_image_size:
                out = out.resize((output_image_size, output_image_size), resample=Image.Resampling.BICUBIC)

            # draw this image
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)
    
    _img.save(filename)