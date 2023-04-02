# %%
from typing import List, Tuple
import types
from pathlib import Path
import datetime
import argparse
import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

import sys
sys.path.append("..")
from models.mtypes import VarEncoderOutput
import dn_util
import checkpoint_util
import image_util
from experiment import Experiment
import cmdline
from latent_cache import LatentCache
from models import vae

class Config(cmdline.QueryConfig):
    image_dir: str
    image_size: int
    limit_dataset: int

    num_images: int
    num_frames: int
    frames_per_pair: int
    fps: int
    no_subdir: bool

    std_add: float
    mean_add_rand: float
    mean_add_rand_frames: float
    by_std: bool
    
    find_close: bool
    find_far: bool

    do_loop: bool

    dataset_idxs: List[int]
    dataset_encouts: List[VarEncoderOutput] = None
    all_dataset_veo: VarEncoderOutput

    def __init__(self):
        super().__init__()
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", type=int, default=None)
        self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("-I", "--dataset_idxs", type=int, nargs="+", default=None, help="specify the image positions in the dataset")
        self.add_argument("-n", "--num_images", type=int, default=2)
        self.add_argument("--find_close", action='store_true', default=False, help="find (num_images-1) more images than the first, each the closest to the previous")
        self.add_argument("--find_far", action='store_true', default=False, help="find (num_images-1) more images than [0], each the farthest from the previous")
        self.add_argument("--no_subdir", default=False, action='store_true')

        self.add_argument("--std_add", type=float, default=None)
        self.add_argument("--mean_add_rand", type=float, default=None)
        self.add_argument("--mean_add_rand_frames", type=float, default=None)
        self.add_argument("--by_std", default=False, action='store_true',
                          help="adjust adds/mults to mean & std by the starting std")

        self.add_argument("--loop", dest='do_loop', default=False, action='store_true')
        self.add_argument("--frames_per_pair", type=int, default=30)
        self.add_argument("--fps", type=int, default=30)
    
    def parse_args(self) -> 'Config':
        super().parse_args()

        # if find_close is set, we'll fill out the dataset_idxs list
        if self.dataset_idxs and not self.find_close and not self.find_far:
            self.num_images = len(self.dataset_idxs)
        
        if self.do_loop:
            if self.dataset_idxs:
                self.dataset_idxs.append(self.dataset_idxs[0])
            self.num_images += 1

        self.num_frames = (self.num_images - 1) * self.frames_per_pair
        self.frames_per_pair = self.num_frames // (self.num_images - 1)

        if self.find_close and self.find_far:
            self.error("can't set both --find_far and --find_close")

        print(f"num frames: {self.num_frames}")
        print(f"batch size: {self.batch_size}")

        return self
    
    def get_dataset(self, image_size: int) -> Dataset:
        train_ds, _ = \
            image_util.get_datasets(image_size=image_size, image_dir=self.image_dir,
                                    train_split=1.0,
                                    limit_dataset=self.limit_dataset)
        return train_ds


    def startend_mults(self, frame: int) -> Tuple[float, float]:
        frame_in_pair = frame % self.frames_per_pair
        start_mult = (self.frames_per_pair - frame_in_pair - 1) / (self.frames_per_pair - 1)
        end_mult = frame_in_pair / (self.frames_per_pair - 1)
        return start_mult, end_mult

    def img_idx(self, frame: int) -> int:
        imgidx = frame // self.frames_per_pair
        # print(f"{frame=} {self.frames_per_pair=} {imgidx=}")
        return imgidx

"""
take the decoded frame and make a bigger picture, showing dataset indices and experiment title.
"""
title_font: ImageFont.ImageFont = None
frame_str_font: ImageFont.ImageFont = None
def annotate(cfg: Config, exp: Experiment, frame: int, image: Image.Image):
    global title_font, frame_str_font

    # TODO: use the stuff in image_util

    image_size = image.width

    title_font_size = 10
    frame_str_font_size = 15
    if title_font is None:
        title_font = ImageFont.truetype(fonts.ttf.Roboto, title_font_size)
        frame_str_font = ImageFont.truetype(fonts.ttf.Roboto, frame_str_font_size)

    field_names = \
        {'nepochs': 'epochs', 
         'net_layers_str': "layers", 'net_encoder_kernel_size': "enc_kern", 
         'image_dir': "image_dir", 'loss_type': "loss",
         'last_val_loss': 'vloss', 'last_train_loss': 'tloss'}
    title_fields: List[str] = list()
    for fieldidx, (field, short) in enumerate(field_names.items()):
        val = getattr(exp, field)
        if isinstance(val, float):
            val = format(val, ".4f")
        if field == 'nepochs':
            val = str(val) + "\n"
        elif fieldidx < len(field_names) - 1:
            val = str(val) + ","
        title_fields.append(f"{short} {val}")
    
    title, title_height = \
        image_util.fit_strings(title_fields, max_width=image_size, font=title_font, list_indent="")
    frame_str_height = int(frame_str_font_size * 4 / 3)

    extra_height = frame_str_height + title_height

    anno_image = Image.new("RGB", (image_size, image_size + extra_height))
    anno_image.paste(image, box=(0, extra_height))
    draw = ImageDraw.ImageDraw(anno_image)

    # draw the frame numbers for start and end, fading start from white to black
    # and end from black to white
    imgidx = cfg.img_idx(frame)
    start_mult, end_mult = cfg.startend_mults(frame)
    dsidx_start, dsidx_end = cfg.dataset_idxs[imgidx:imgidx + 2]
    start_val = int(start_mult * 255)
    end_val = int(end_mult * 255)
    start_x = imgidx * image_size / cfg.num_images
    end_x = (imgidx + 1) * image_size / cfg.num_images
    draw.text(xy=(start_x, title_height), text=str(dsidx_start), font=frame_str_font, fill=(start_val, start_val, start_val))
    draw.text(xy=(end_x, title_height), text=str(dsidx_end), font=frame_str_font, fill=(end_val, end_val, end_val))

    # draw the title 
    draw.text(xy=(0, 0), text=title, font=title_font, fill='white')

    return anno_image

"""
compute the latent input for each frame. generates (num_frames) latents.
"""
def generate_frame_latents(cfg: Config): # -> Generator[Tensor]:
    encout_prev = cfg.dataset_encouts[0]
    r: Tensor = None
    last_imgidx = -1
    for frame in range(cfg.num_frames):
        imgidx = cfg.img_idx(frame)
        start_encout, end_encout = cfg.dataset_encouts[imgidx : imgidx + 2]

        start_mult, end_mult = cfg.startend_mults(frame)
        encout_lerp = start_encout * start_mult + end_encout * end_mult

        if cfg.mean_add_rand_frames:
            if r is None or imgidx != last_imgidx:
                r = torch.randn_like(encout_lerp.mean) * cfg.mean_add_rand_frames
            # add = end_encout.std * r
            add = (encout_lerp.std) * r
            encout = encout_prev.copy(mean=encout_prev.mean + add)
            encout = (encout + encout_lerp) * 0.5
        else:
            encout = encout_lerp

        yield encout.sample()
        encout_prev = encout

def setup_experiments(cfg: Config): # -> Generator[Experiment, image_latents.ImageLatents]:
    exps = cfg.list_experiments()
    for exp_idx, exp in enumerate(exps):
        cfg.dataset_encouts = None
        cfg.all_dataset_veo = None
        path = exp.get_run().checkpoint_path
        with open(path, "rb") as file:
            state_dict = torch.load(path)
            exp.net: vae.VarEncDec = dn_util.load_model(state_dict)
            exp.net = exp.net.to(cfg.device)
            exp.net.eval()

        # image_size = cfg.image_size or exp.net_image_size
        dataset = cfg.get_dataset(exp.net_image_size)

        cache = LatentCache(net=exp.net, net_path=path, batch_size=cfg.batch_size,
                            dataset=dataset, device=cfg.device)
        
        if cfg.by_std:
            all_encouts = cache.encouts_for_idxs()
            all_means = torch.stack([eo.mean for eo in all_encouts])
            mean = all_means.mean(dim=0)
            std = all_means.std(dim=0)
            print(f"{mean.shape=} {std.shape=}")
            print(f"{mean.mean()=} {std.mean()=}")
            cfg.all_dataset_veo = VarEncoderOutput(mean=mean, std=std)

        if not cfg.dataset_idxs:
            all_dataset_idxs = list(range(len(dataset)))
            random.shuffle(all_dataset_idxs)
            cfg.dataset_idxs = all_dataset_idxs[:cfg.num_images]
            if cfg.do_loop:
                cfg.dataset_idxs[-1] = cfg.dataset_idxs[0]
        
        if cfg.find_close or cfg.find_far:
            if exp_idx == 0:
                src_idx = cfg.dataset_idxs[0]
                src_encout = cache.encouts_for_idxs([src_idx])[0]

                # [1] = closest to [0] that's not [0]
                # [2] = closest to [1] that's not [0 or 1]
                # ...
                new_idxs = [src_idx]
                new_encouts = [src_encout]
                while len(new_idxs) < cfg.num_images:
                    results = cache.find_closest_n(src_idx=new_idxs[-1], 
                                                    src_encout=new_encouts[-1],
                                                    n=cfg.num_images)
                    # print(f"{results[0][0]=} {results[-1][0]=}")
                    if cfg.find_far:
                        results = reversed(results)
                    for close_idx, close_encout in results:
                        if close_idx not in new_idxs:
                            new_idxs.append(close_idx)
                            new_encouts.append(close_encout)
                            break
                
                cfg.dataset_idxs = new_idxs
                cfg.dataset_encouts = new_encouts
            else:
                images = cache.get_images(cfg.dataset_idxs)
                cfg.dataset_encouts = cache.encouts_for_images(images)

        else: #if cfg.dataset_encouts is None:
            images = cache.get_images(cfg.dataset_idxs)
            cfg.dataset_encouts = cache.encouts_for_images(images)
        
        if cfg.std_add is not None:
            for i, eo in enumerate(cfg.dataset_encouts):
                std_add = cfg.std_add
                if cfg.by_std:
                    std_add = cfg.all_dataset_veo.std * std_add
                cfg.dataset_encouts[i] = eo.copy(std=eo.std + std_add)

        if cfg.mean_add_rand:
            r_shape = cfg.dataset_encouts[0].mean.shape
            for i, eo in enumerate(cfg.dataset_encouts):
                r = torch.randn(size=r_shape)
                if cfg.by_std:
                    r = cfg.all_dataset_veo.std * r
                r = r * cfg.mean_add_rand
                cfg.dataset_encouts[i] = eo.copy(mean=eo.mean + r)

        print("--dataset_idxs", " ".join(map(str, cfg.dataset_idxs)))

        yield exp, cache

if __name__ == "__main__":
    cfg = Config().parse_args()

    if cfg.no_subdir:
        animdir = Path("animations")
    else:
        animdir = Path("animations", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    animdir.mkdir(parents=True, exist_ok=True)

    exps = cfg.list_experiments()
    for exp_idx, (exp, cache) in enumerate(setup_experiments(cfg)):
        # build output path and video container
        image_size = cfg.image_size or exp.net.image_size
        parts: List[str] = list()

        parts.extend([str(cfg.dataset_idxs[0]),
                    #   *(["btwn"] if cfg.walk_between else []),
                    #   *(["towards"] if cfg.walk_towards else []),
                    #   *([f"wmult_{cfg.walk_mult:.3f}"] if cfg.walk_between or cfg.walk_after else []),
                      *(["fclose"] if cfg.find_close else []),
                      *(["ffar"] if cfg.find_far else []),
                      f"tloss_{exp.last_train_loss:.3f}",
                      f"vloss_{exp.last_val_loss:.3f}",
                      exp.label])
        animpath = Path(animdir, f"{exp_idx}-" + ",".join(parts) + ".mp4")
        print()
        print(f"{exp_idx+1}/{len(exps)} {animpath}:")

        frame_latents_all = list(generate_frame_latents(cfg))
        frame_latents_batches: List[Tensor] = list()
        while len(frame_latents_all) > 0:
            frame_batch = frame_latents_all[:cfg.batch_size]
            frame_latents_batches.append(frame_batch)
            frame_latents_all = frame_latents_all[cfg.batch_size:]

        # process the batches and output the frames.
        anim_out = None
        for batch_nr, frame_batch in tqdm.tqdm(list(enumerate(frame_latents_batches))):
            out = cache.decode(frame_batch)

            # make the image tensor into an image then make all images the same
            # (output) size.
            for sample_num, sample_out in enumerate(out):
                frame = batch_nr * cfg.batch_size + sample_num
                image = image_util.tensor_to_pil(sample_out.detach().cpu(), image_size)
                image = annotate(cfg, exp, frame, image)
                if anim_out is None:
                    anim_out = cv2.VideoWriter(str(animpath),
                                               cv2.VideoWriter_fourcc(*'mp4v'), 
                                               cfg.fps, 
                                               (image.width, image.height))

                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                anim_out.write(image_cv)
        anim_out.release()

# %%
