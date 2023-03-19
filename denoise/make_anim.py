# %%
from typing import List, Dict, Deque, Tuple, Callable
from collections import deque
from pathlib import Path
import datetime
import argparse
import tqdm
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
import cv2

import sys
sys.path.append("..")
import model_new
import dn_util
import model_util
import image_util
from experiment import Experiment

device = "cuda"

class Config(argparse.Namespace):
    checkpoints: List[Tuple[Path, Experiment]]
    image_dir: str
    image_size: int
    num_images: int
    num_frames: int
    frames_per_pair: int
    fps: int

    dataset_idxs: List[int]

    random: bool
    walk_between: bool
    walk_towards: bool
    walk_after: int
    walk_mult: float
    find_close: bool

    do_loop: bool
    sort_key: str

    batch_size: int

    def startend_mults(self, frame: int) -> Tuple[float, float]:
        frame_in_pair = frame % self.frames_per_pair
        start_mult = (self.frames_per_pair - frame_in_pair - 1) / (self.frames_per_pair - 1)
        end_mult = frame_in_pair / (self.frames_per_pair - 1)
        return start_mult, end_mult

    def img_idx(self, frame: int) -> int:
        imgidx = frame // self.frames_per_pair
        # print(f"{frame=} {self.frames_per_pair=} {imgidx=}")
        return imgidx

def latents_for_images(image_tensors: List[Tensor], encoder_fn: Callable[[Tensor], Tensor]) -> List[Tensor]:
    image_tensors = [image_tensor.unsqueeze(0).to(device) 
                     for image_tensor in image_tensors]
    dslatents = [encoder_fn(image_tensor) for image_tensor in image_tensors]
    return dslatents


"""
take the decoded frame and make a bigger picture, showing dataset indices and experiment title.
"""
title_font: ImageFont.ImageFont = None
frame_str_font: ImageFont.ImageFont = None
def annotate(cfg: Config, exp: Experiment, frame: int, image: Image.Image):
    global title_font, frame_str_font

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
         'lastepoch_val_loss': 'vloss', 'lastepoch_train_loss': 'tloss',
         'lastepoch_bl_loss': 'bl_loss', 'lastepoch_bl_loss_true': 'blt_loss'}
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

# checkpoints, num_images, num_frames, frames_per_pair
DEFAULT_SORT_KEY = 'lastepoch_val_loss'
def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", dest='batch_size', type=int, default=16)
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
    parser.add_argument("-s", "--sort_key", type=str, default=DEFAULT_SORT_KEY)
    parser.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
    parser.add_argument("-i", "--image_size", type=int, default=None)
    parser.add_argument("-I", "--dataset_idxs", type=int, nargs="+", default=None, help="specify the image positions in the dataset")
    parser.add_argument("-n", "--num_images", type=int, default=2)
    parser.add_argument("--find_close", action='store_true', default=False, help="find (num_images-1) more images close to the first input")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--random", default=False, action='store_true', help="use random latent images instead of loading them")
    parser.add_argument("--walk_between", default=False, action='store_true', help="randomly perturb the walk from one image to the next")
    parser.add_argument("--walk_towards", default=False, action='store_true', help="when walking between, adjust the random walk to be weight towards the next image")
    parser.add_argument("--walk_after", default=None, type=int, help="after N images, randomly walk for the rest of the time")
    parser.add_argument("--walk_mult", default=0.2, type=float, help="amount to randomly perturb the walk by")

    parser.add_argument("--loop", dest='do_loop', default=False, action='store_true')
    parser.add_argument("--frames_per_pair", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)

    cfg: Config = parser.parse_args(namespace=Config())
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern, re.DOTALL)
    
    cfg.checkpoints = model_util.find_checkpoints(only_paths=cfg.pattern, attr_matchers=cfg.attribute_matchers)
    cfg.checkpoints = list(sorted(cfg.checkpoints, key=lambda cp_tuple: getattr(cp_tuple[1], cfg.sort_key)))

    experiments = [exp for path, exp in cfg.checkpoints]
    for i, exp in enumerate(experiments):
        print(f"{i+1}.", exp.label)
    print()

    if cfg.dataset_idxs:
        # if find_close is set, we'll fill out the dataset_idxs list
        if not cfg.find_close:
            cfg.num_images = len(cfg.dataset_idxs)
    
    if cfg.do_loop:
        if cfg.dataset_idxs is not None:
            cfg.dataset_idxs.append(cfg.dataset_idxs[0])
        cfg.num_images += 1

    if cfg.num_frames is None:
        cfg.num_frames = (cfg.num_images - 1) * cfg.frames_per_pair
    cfg.frames_per_pair = cfg.num_frames // (cfg.num_images - 1)

    if cfg.walk_towards:
        cfg.walk_between = True
        print(f"weight random walk towards end image")

    if cfg.walk_between:
        print(f"walking between images at {cfg.walk_mult=}")
    elif cfg.walk_after is not None:
        print(f"walking after {cfg.walk_after=} with {cfg.walk_mult=}")
    
    print(f"num frames: {cfg.num_frames}")
    print(f"batch size: {cfg.batch_size}")

    return cfg

"""
compute the latent input for each frame. generates (num_frames) latents.
"""
def compute_latents(cfg: Config, dslatents: List[Tensor]):
    latent_last_frame = dslatents[0]
    for frame in range(cfg.num_frames):
        imgidx = cfg.img_idx(frame)
        start_mult, end_mult = cfg.startend_mults(frame)

        start_latent, end_latent = dslatents[imgidx:imgidx + 2]
        latent_lerp = start_mult * start_latent + end_mult * end_latent

        if cfg.walk_between:
            if cfg.walk_towards:
                # make the random number scaled by the difference between start and end
                # r = r * (end_latent - start_latent)
                r = torch.randn_like(start_latent) * cfg.walk_mult
                r = r * F.softmax(end_latent - latent_last_frame, dim=2)
                # r = F.softmax(r, dim=2)
            else:
                r = torch.randn_like(start_latent) * cfg.walk_mult

            latent_walk = latent_last_frame + r

            # scale the latent effect down the closer we get to the end
            # latent = latent_walk * start_mult + latent_lerp * end_mult

            # just pick 1/2 and 1/2 between lerp (where we should be) and random
            latent = latent_walk * 0.5 + latent_lerp * 0.5
        elif cfg.walk_after is not None and imgidx >= cfg.walk_after:
            r = torch.randn_like(start_latent) * cfg.walk_mult
            latent = latent_last_frame + r
        else:
            latent = latent_lerp

        yield latent
        latent_last_frame = latent

def load_encoder_decoder(path: Path) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    with open(path, "rb") as cp_file:
        try:
            model_dict = torch.load(path)
            net = dn_util.load_model(model_dict).to(device)
            net.eval()
            if isinstance(net, model_new.VarEncDec):
                encoder_fn = net.encode
                decoder_fn = net.decode
            else:
                encoder_fn = net.encoder
                decoder_fn = net.decoder
        except Exception as e:
            print(f"error processing {path}:", file=sys.stderr)
            raise e
    
    return encoder_fn, decoder_fn

def get_datasets(cfg: Config):
    for cp_idx, (path, exp) in enumerate(cfg.checkpoints):
        encoder_fn, decoder_fn = load_encoder_decoder(path)

        image_size = cfg.image_size or exp.net_image_size
        dataloader, _ = dn_util.get_dataloaders(disable_noise=True, 
                                                image_size=exp.net_image_size, 
                                                image_dir=cfg.image_dir, batch_size=cfg.batch_size,
                                                shuffle=False)
        dataset = dataloader.dataset
        dslatents: List[Tensor] = None
        if cfg.random:
            cfg.dataset_idxs = list(range(cfg.num_images))
            dslatents = [torch.randn(exp.net.encoder_out_dim).unsqueeze(0).to(device) 
                         for _ in range(cfg.num_images)]
        else:
            if not cfg.dataset_idxs:
                cfg.dataset_idxs = [ridx.item() for ridx in torch.randint(0, len(dataset), (cfg.num_images,))]
                if cfg.do_loop:
                    cfg.dataset_idxs[-1] = cfg.dataset_idxs[0]
        
        if cfg.find_close and cp_idx == 0:
            best_distance: Deque[Tuple[Tensor, int]] = deque()

            src_idx = cfg.dataset_idxs[0]
            src_latent = latents_for_images([dataset[src_idx][0]], encoder_fn)[0]
            print(f"looking for closest images to {src_idx:}")

            dataloader_it = iter(dataloader)
            results: List[Tuple[float, int, Tensor]] = list()

            imgidx = 0
            for batch_nr in tqdm.tqdm(list(enumerate(range(len(dataloader))))):
                input, _truth = next(dataloader_it)
                latent_outs = encoder_fn(input.to(device)).detach()
                for lat_idx, latent_out in enumerate(latent_outs):
                    if imgidx + lat_idx == src_idx:
                        continue
                    distance = ((latent_out - src_latent) ** 2).sum() ** 0.5
                    results.append((distance.item(), imgidx + lat_idx, latent_out.unsqueeze(0)))
                imgidx += len(latent_outs)
            
            best_pairs = sorted(results)[:cfg.num_images - 1]
            best_latents = [bp[2] for bp in best_pairs]
            best_ds_idxs = [bp[1] for bp in best_pairs]
            cfg.dataset_idxs = [src_idx] + best_ds_idxs
            dslatents = [src_latent] + best_latents
            # print(f"{cfg.num_images - 1} images closest to {src_idx}:", ", ".join(map(str, best_ds_idxs)))

        print("--dataset_idxs", " ".join(map(str, cfg.dataset_idxs)))

        if dslatents is None:
            image_tensors = [dataset[dsidx][1] for dsidx in cfg.dataset_idxs]
            dslatents = latents_for_images(image_tensors, encoder_fn)

        yield cp_idx, exp, dslatents, decoder_fn, image_size

if __name__ == "__main__":
    cfg = parse_args()

    animdir = Path("animations", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    animdir.mkdir(parents=True, exist_ok=True)

    for cp_idx, exp, dslatents, decoder_fn, image_size in get_datasets(cfg):
        # build output path and video container
        parts: List[str] = list()
        if cfg.sort_key != DEFAULT_SORT_KEY:
            sort_val = getattr(exp, cfg.sort_key)
            if isinstance(sort_val, float):
                sort_val = format(sort_val, ".4f")
            parts.append(f"{cfg.sort_key}_{sort_val}")

        parts.extend([str(cfg.dataset_idxs[0]),
                      *(["btwn"] if cfg.walk_between else []),
                      *(["towards"] if cfg.walk_towards else []),
                      *(["fclose"] if cfg.find_close else []),
                      *([f"wmult_{cfg.walk_mult:.3f}"] if cfg.walk_between or cfg.walk_after else []),
                      f"tloss_{exp.lastepoch_train_loss:.3f}",
                      f"vloss_{exp.lastepoch_val_loss:.3f}",
                      exp.label])
        animpath = Path(animdir, f"{cp_idx}-" + ",".join(parts) + ".mp4")
        print()
        print(f"{cp_idx+1}/{len(cfg.checkpoints)} {animpath}:")

        # compute latents for all the frames, in batch.
        batch_latents_in: List[Tensor] = list()
        for frame, latent_in in enumerate(compute_latents(cfg, dslatents)):
            batch_num = frame // cfg.batch_size
            sample_num = frame % cfg.batch_size
            if batch_num >= len(batch_latents_in):
                nbatch = min(cfg.num_frames - frame, cfg.batch_size)
                batch_latent_size = (nbatch, *[d for d in latent_in.shape[1:]])
                batch_latents_in.append(torch.zeros(batch_latent_size, device=device))
            batch_latents_in[batch_num][sample_num] = latent_in[0]

        # process the batches and output the frames.
        anim_out = None
        for batch_nr, batch_latents_in in tqdm.tqdm(list(enumerate(batch_latents_in))):
            out = decoder_fn(batch_latents_in)

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
