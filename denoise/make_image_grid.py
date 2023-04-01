# %%
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import numpy as np
import torch
import tqdm
from pathlib import Path

import sys
sys.path.append("..")
from experiment import Experiment
import image_util
import checkpoint_util
import dn_util
from models import vae
import cmdline

class Config(cmdline.QueryConfig):
    image_dir: str
    image_size: int
    shortcode: str

    def __init__(self):
        super().__init__()
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", default=64, type=int)
        self.add_argument("-c", "--shortcode", default=None)

# checkpoints, num_images, num_frames, frames_per_pair
if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    dl_image_size = cfg.image_size
    image_size = cfg.image_size

    net: vae.VarEncDec = None
    exp: Experiment = None
    if cfg.shortcode:
        exps = checkpoint_util.list_experiments()
        exps = [exp for exp in exps if exp.shortcode == cfg.shortcode]
        if not len(exps):
            raise Exception(f"can't find VarEncDec with shortcode {cfg.shortcode}")

        exp = exps[0]
        best_run = exp.run_best_loss('tloss')
        cp_path = best_run.checkpoint_path
        state_dict = torch.load(cp_path)
        net = dn_util.load_model(state_dict)
        dl_image_size = net.image_size

    dataloader, _ = image_util.get_dataloaders(image_size=dl_image_size,
                                               image_dir=cfg.image_dir, batch_size=cfg.batch_size, 
                                               train_split=1.0, shuffle=False)
    dataset = dataloader.dataset

    pad_image = 2
    pad_ten = 10
    pad_size = image_size + pad_image

    ncols = int(len(dataset) ** 0.5) // 10 * 10
    ncols_ten = ncols // 10
    nrows = math.ceil(len(dataset) / ncols)
    width = pad_size * ncols + pad_ten * (ncols // 10)
    height = pad_size * nrows + pad_ten * (nrows // 10)
    print(f"{width=} {height=}")

    image = Image.new("RGB", (width, height))

    font_size = max(10, image_size // 10)
    font = ImageFont.truetype(fonts.ttf.Roboto, font_size)
    draw = ImageDraw.ImageDraw(image)

    dl_iter = iter(dataloader)
    for batch in tqdm.tqdm(range(len(dataloader))):
        images, _truth = next(dl_iter)

        if net is not None:
            images = net.forward(images)

        for img_idx, img_t in enumerate(images):
            img_idx = batch * cfg.batch_size + img_idx

            row = img_idx // ncols
            col = img_idx % ncols
            text = str(img_idx)
            imgx = pad_size * col + pad_ten * (col // 10)
            imgy = pad_size * row + pad_ten * (row // 10)

            img = image_util.tensor_to_pil(img_t, image_size)
            image.paste(img, (imgx, imgy))

            left, top, right, bot = draw.textbbox((0, 0), text=text, font=font)
            text_ul = (imgx, imgy + image_size - bot)
            text_br = (text_ul[0] + right, text_ul[1] + bot)
            draw.rectangle((text_ul, text_br), fill='black')
            draw.text(text_ul, text=text, font=font, fill='white')

    image_dir_basename = "image-grid--" + Path(cfg.image_dir).name + f"x{cfg.image_size}"
    if exp is not None:
        image.save(f"{image_dir_basename}--{exp.shortcode}.png")
    else:
        image.save(f"{image_dir_basename}.png")

# %%
