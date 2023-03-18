# %%
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
import tqdm

import sys
sys.path.append("..")
import dn_util


# checkpoints, num_images, num_frames, frames_per_pair
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
    parser.add_argument("-i", "--image_size", default=64, type=int)
    cfg = parser.parse_args()

    image_size = cfg.image_size
    dataloader, _ = dn_util.get_dataloaders(disable_noise=True, 
                                            image_size=image_size,
                                            image_dir=cfg.image_dir, batch_size=1, 
                                            shuffle=False)

    to_image = transforms.ToPILImage()

    pad_image = 2
    pad_ten = 10
    pad_size = image_size + pad_image

    ncols = int(len(dataloader) ** 0.5) // 10 * 10
    ncols_ten = ncols // 10
    nrows = math.ceil(len(dataloader) / ncols)
    width = pad_size * ncols + pad_ten * (ncols // 10)
    height = pad_size * nrows + pad_ten * (nrows // 10)
    print(f"{width=} {height=}")

    image = Image.new("RGB", (width, height))

    font_size = max(10, image_size // 10)
    font = ImageFont.truetype(fonts.ttf.Roboto, font_size)
    draw = ImageDraw.ImageDraw(image)

    dl_iter = iter(dataloader)
    for idx in tqdm.tqdm(range(len(dataloader))):
        input, truth = next(dl_iter)
        row = idx // ncols
        col = idx % ncols
        text = str(idx)
        imgx = pad_size * col + pad_ten * (col // 10)
        imgy = pad_size * row + pad_ten * (row // 10)

        input_img = to_image(input[0])
        image.paste(input_img, (imgx, imgy))

        left, top, right, bot = draw.textbbox((0, 0), text=text, font=font)
        text_ul = (imgx, imgy + image_size - bot)
        text_br = (text_ul[0] + right, text_ul[1] + bot)
        draw.rectangle((text_ul, text_br), fill='black')
        draw.text(text_ul, text=text, font=font, fill='white')

    image.save("image-grid.png")

# %%
