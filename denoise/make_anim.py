from typing import List, Literal, Generator
from pathlib import Path
import datetime
import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf
import numpy as np
import cv2

import sys
sys.path.append("..")

import torch
from torch import Tensor

import dn_util
import image_util
from experiment import Experiment, ExpRun
import cmdline
import imagegen

# python make_anim.py -b 4 -nc VarEncDec -a 'net.image_size = 256' 'nepochs > 100' 
#   -n 5 --fpp 120 --fields shortcode net_latent_dim -s time
class Config(cmdline.QueryConfig):
    image_dir: str
    image_size: int

    num_images: int
    num_frames: int
    frames_per_pair: int
    fps: int

    use_subdir: bool
    use_fancy_filenames: bool
    filename_fields: List[str]

    # std_add: float
    # mean_add_rand: float
    # mean_add_rand_frames: float
    # by_std: bool

    draw_mode: Literal["real", "latent"] = None
    # find_close: bool
    # find_far: bool

    dataset_idxs: List[int]

    def __init__(self):
        super().__init__()
        self.add_argument("-d", "--image_dir", default="images.alex+1star-1024")
        self.add_argument("-i", "--image_size", type=int, default=None)
        self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("-I", "--dataset_idxs", type=int, nargs="+", default=None, help="specify the image positions in the dataset")
        self.add_argument("-n", "--num_images", type=int, default=2)
        self.add_argument("--draw", dest='draw_mode', default=None, choices=["real", "latent"])
        # self.add_argument("--find_close", action='store_true', default=False, help="find (num_images-1) more images than the first, each the closest to the previous")
        # self.add_argument("--find_far", action='store_true', default=False, help="find (num_images-1) more images than [0], each the farthest from the previous")
        self.add_argument("--use_subdir", default=False, action='store_true')
        self.add_argument("--fancy_filenames", dest='use_fancy_filenames', default=False, action='store_true')
        self.add_argument("--fields", dest='filename_fields', default=list(), nargs='+', type=str)

        # self.add_argument("--std_add", type=float, default=None)
        # self.add_argument("--mean_add_rand", type=float, default=None)
        # self.add_argument("--mean_add_rand_frames", type=float, default=None)

        self.add_argument("--fpp", "--frames_per_pair", dest='frames_per_pair', type=int, default=30)
        self.add_argument("--fps", type=int, default=30)
    
    def parse_args(self) -> 'Config':
        super().parse_args()

        # bogus dataset just to get the count.
        dataset, _ = image_util.get_datasets(image_size=64, image_dir=self.image_dir, train_split=1.0)
        all_ds_idxs = list(range(len(dataset)))
        random.shuffle(all_ds_idxs)

        # if find_close is set, we'll fill out the dataset_idxs list
        if self.dataset_idxs:
            self.num_images = len(self.dataset_idxs)
        else:
            self.dataset_idxs = all_ds_idxs[:self.num_images]
        
        self.num_frames = (self.num_images - 1) * self.frames_per_pair
        self.frames_per_pair = self.num_frames // (self.num_images - 1)

        print(f"     num frames: {self.num_frames}")
        print(f"frames per pair: {self.frames_per_pair}")
        print(f"     batch size: {self.batch_size}")

        if self.use_subdir:
            self.animdir = Path("animations", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.animdir = Path("animations")
        self.animdir.mkdir(parents=True, exist_ok=True)

        return self

    def make_path(self, exp: Experiment, exp_idx: int) -> Path:
        parts: List[str] = [f"makeanim_{exp_idx}"]
        if self.use_fancy_filenames:
            parts = [
                f"makeanim_{exp.shortcode}_{exp.nepochs}",
                exp.label,
                # str(cfg.dataset_idxs[0]),
                # *(["fclose"] if cfg.find_close else []),
                # *(["ffar"] if cfg.find_far else []),
                f"tloss_{exp.last_train_loss:.3f}"
            ]
        elif self.filename_fields:
            parts.extend([f"{field}_{getattr(exp, field)}" for field in cfg.filename_fields])

        return Path(self.animdir, ",".join(parts) + ".mp4")

"""
take the decoded frame and make a bigger picture, showing dataset indices and experiment title.
"""
title_font: ImageFont.ImageFont = None
frame_str_font: ImageFont.ImageFont = None
last_exp: Experiment = None
last_exp_descr: str = None
def annotate(cfg: Config, exp: Experiment, frame: int, image: Image.Image):
    global title_font, frame_str_font, last_exp, last_exp_descr

    # TODO: use the stuff in image_util

    image_size = image.width

    title_font_size = 10
    frame_str_font_size = 15
    if title_font is None:
        title_font = ImageFont.truetype(fonts.ttf.Roboto, title_font_size)
        frame_str_font = ImageFont.truetype(fonts.ttf.Roboto, frame_str_font_size)

    if exp != last_exp:
        last_exp_descr = dn_util.exp_descr(exp, include_label=False)
        last_exp = exp
    title, title_height = \
        image_util.fit_strings(last_exp_descr, max_width=image_size, font=title_font, list_indent="")
    frame_str_height = int(frame_str_font_size * 4 / 3)

    extra_height = frame_str_height + title_height

    anno_image = Image.new("RGB", (image_size, image_size + extra_height))
    anno_image.paste(image, box=(0, extra_height))
    draw = ImageDraw.ImageDraw(anno_image)

    # draw the frame numbers for start and end, fading start from white to black
    # and end from black to white
    frame_in_pair = frame % cfg.frames_per_pair
    end_mult = frame_in_pair / (cfg.frames_per_pair - 1)
    start_mult = 1.0 - end_mult
    start_val = int(start_mult * 255)
    end_val = int(end_mult * 255)
    start_color = (start_val,) * 3
    end_color = (end_val,) * 3

    imgidx = frame // cfg.frames_per_pair
    dsidx_start, dsidx_end = cfg.dataset_idxs[imgidx:imgidx + 2]
    start_x = imgidx * image_size / cfg.num_images
    end_x = (imgidx + 1) * image_size / cfg.num_images
    draw.text(xy=(start_x, title_height), text=str(dsidx_start), font=frame_str_font, fill=start_color)
    draw.text(xy=(end_x, title_height), text=str(dsidx_end), font=frame_str_font, fill=end_color)

    # draw the title 
    draw.text(xy=(0, 0), text=title, font=title_font, fill='white')

    return anno_image

def draw_numbers_real(text: str, image_size: int, exp_gen: imagegen.ImageGenExp) -> Tensor:
    font = ImageFont.truetype(fonts.ttf.Roboto, image_size)
    
    image = Image.new("RGB", (image_size, image_size))
    draw = ImageDraw.ImageDraw(im=image)
    draw.text(xy=(0, 0), text=text, font=font, fill='white')
    image_t = image_util.pil_to_tensor(image=image, net_size=image_size)
    return exp_gen.cache.samples_for_images([image_t])[0]

def draw_numbers_latent(text: str, draw_chans: List[int], latent_dim: List[int]) -> Tensor:
    chan, size, _size = latent_dim
    font = ImageFont.truetype(fonts.ttf.Roboto, size // 3)
    
    image = Image.new("F", (size, size))   # 32-bit floating point for each pixel
    draw = ImageDraw.ImageDraw(im=image)
    draw.text(xy=(0, 0), text=text, font=font, fill='white')

    latent = torch.zeros(latent_dim)
    for dchan in draw_chans:
        latent[dchan] = image_util.pil_to_tensor(image=image, net_size=size)

    return latent


_igen: imagegen.ImageGen = None
def gen_frame_latents(cfg: Config, 
                      exp: Experiment, run: ExpRun) -> Generator[Image.Image, None, None]:
    global _igen
    image_size = cfg.image_size or exp.net_image_size
    if _igen is None:
        _igen = imagegen.ImageGen(image_dir=cfg.image_dir, output_image_size=image_size, 
                                  device=cfg.device, batch_size=cfg.batch_size)

    exp_gen = _igen.for_run(exp, run)
    for i in range(cfg.num_images - 1):
        start_idx = cfg.dataset_idxs[i]
        end_idx = cfg.dataset_idxs[i + 1]

        if cfg.draw_mode == 'real':
            start = draw_numbers_real(str(i), image_size, exp_gen)
            end = draw_numbers_real(str(i + 1), image_size, exp_gen)
        
        elif cfg.draw_mode == 'latent':
            latent_dim = exp_gen.latent_dim
            latent_chan = latent_dim[0]
            ncombos = 2 ** latent_chan
            combo_start = i % ncombos
            combo_end = (i + 1) % ncombos
            draw_chan_start: List[int] = list()
            draw_chan_end: List[int] = list()
            for bit in range(latent_chan):
                mask = 2 ** bit
                if combo_start & mask:
                    draw_chan_start.append(bit)
                if combo_end & mask:
                    draw_chan_end.append(bit)

            text_start = "latent\n" + "-".join(["1" if chan in draw_chan_start else "0" for chan in list(reversed(range(latent_chan)))])
            text_end = "latent\n" + "-".join(["1" if chan in draw_chan_end else "0" for chan in list(reversed(range(latent_chan)))])

            start = draw_numbers_latent(text_start, draw_chan_start, latent_dim)
            end = draw_numbers_latent(text_end, draw_chan_end, latent_dim)
        
        else:
            start, end = exp_gen.get_image_latents(image_idxs=[start_idx, end_idx])

        frames = exp_gen.gen_lerp(start=start, end=end, steps=cfg.frames_per_pair)
        frame_it = iter(frames)
        for _ in tqdm.tqdm(list(range(cfg.frames_per_pair))):
            yield next(frame_it)

def main():
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()
    for exp_idx, exp in enumerate(exps):
        animpath = cfg.make_path(exp, exp_idx)

        print()
        print(f"{exp_idx+1}/{len(exps)} {animpath}:")

        anim_out: cv2.VideoWriter = None
        for frame, image in enumerate(gen_frame_latents(cfg, exp, exp.get_run())):
            image = annotate(cfg, exp, frame, image)
            if anim_out is None:
                anim_out = cv2.VideoWriter(str(animpath),
                                           cv2.VideoWriter_fourcc(*'mp4v'), 
                                           cfg.fps, 
                                           (image.width, image.height))

            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            anim_out.write(image_cv)
        anim_out.release()

if __name__ == "__main__":
    main()
