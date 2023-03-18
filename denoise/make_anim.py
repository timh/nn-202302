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
import model_new
from experiment import Experiment
import noised_data
import image_util
import model_util, dn_util

device = "cuda"

to_pil_xform = transforms.ToPILImage("RGB")
def to_pil(image_tensor: Tensor, image_size: int) -> Image.Image:
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]

    image: Image.Image = to_pil_xform(image_tensor)
    if image_size != image.width:
        image = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    
    return image

_to_tensor_xform = transforms.ToTensor()
def to_tensor(image: Image.Image, net_size: int) -> Tensor:
    if net_size != image.width:
        image = image.resize((net_size, net_size), resample=Image.Resampling.BICUBIC)
    return _to_tensor_xform(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
    parser.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
    parser.add_argument("-i", "--image_size", type=int, default=None)
    parser.add_argument("-I", "--image_idxs", type=int, nargs="+", default=None)
    parser.add_argument("-N", "--num_images", type=int, default=2)
    parser.add_argument("-n", "--num_frames", type=int, default=None)
    parser.add_argument("--loop", dest='do_loop', default=False, action='store_true')
    parser.add_argument("--frames_per_pair", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)

    cfg = parser.parse_args()
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern, re.DOTALL)
    
    checkpoints = model_util.find_checkpoints(only_paths=cfg.pattern, attr_matchers=cfg.attribute_matchers)
    checkpoints = list(reversed(sorted(checkpoints, key=lambda cp_tuple: cp_tuple[1].started_at)))
    experiments = [exp for path, exp in checkpoints]
    [print(f"{i+1}.", exp.label) for i, exp in enumerate(experiments)]

    image_idxs: List[int] = None
    if cfg.image_idxs:
        image_idxs = cfg.image_idxs
        num_images = len(image_idxs)
    else:
        num_images = cfg.num_images
    
    if cfg.do_loop:
        if num_images % 2 == 1:
            parser.error(f"need even number of images, not {num_images=}, to loop; the first image added at the end")
        if image_idxs is not None:
            image_idxs.append(image_idxs[0])
        num_images += 1

    num_frames = cfg.num_frames or (num_images - 1) * cfg.frames_per_pair
    frames_per_pair = num_frames // (num_images - 1)

    for i, (path, exp) in enumerate(checkpoints):
        parts = [f"tloss_{exp.lastepoch_train_loss:.3f}",
                 f"vloss_{exp.lastepoch_val_loss:.3f}",
                 exp.label]
        # parts.append("-I " + " ".join(map(str, image_idxs)))
        filename = f"anim_{i}--" + ",".join(parts) + ".gif"
        print()
        print(f"{i+1}/{len(checkpoints)} {filename}:")

        with open(path, "rb") as cp_file:
            try:
                model_dict = torch.load(path)
                exp.net = dn_util.load_model(model_dict).to(device)
                if isinstance(exp.net, model_new.VarEncDec):
                    encoder_fn = exp.net.encode
                    decoder_fn = exp.net.decode
                else:
                    encoder_fn = exp.net.encoder
                    decoder_fn = exp.net.decoder
            except Exception as e:
                print(f"error processing {path}:", file=sys.stderr)
                raise e


        image_size = cfg.image_size or exp.net_image_size
        dataloader, _ = dn_util.get_dataloaders(disable_noise=True, 
                                                image_size=exp.net_image_size, 
                                                image_dir=cfg.image_dir, batch_size=1)
        dataset = dataloader.dataset
        if not cfg.image_idxs:
            image_idxs = [i.item() for i in torch.randint(0, len(dataset), (num_images,))]
            if cfg.do_loop:
                image_idxs[-1] = image_idxs[0]
            print(f"--image_idxs", " ".join(map(str, image_idxs)))

        image_tensors = [dataset[image_idx][1] for image_idx in image_idxs]
        image_tensors = [image_tensor.unsqueeze(0).to(device) for image_tensor in image_tensors]
        latents = [encoder_fn(image_tensor) for image_tensor in image_tensors]

        image_frames: List[Image.Image] = list()
        for frame in tqdm.tqdm(range(num_frames)):
            start_idx = math.floor(frame / frames_per_pair)
            end_idx = math.ceil(frame / frames_per_pair)
            start_latent, end_latent = latents[start_idx], latents[end_idx]

            frame_in_pair = frame % frames_per_pair
            start_mult = (frames_per_pair - frame_in_pair - 1) / (frames_per_pair - 1)
            end_mult = frame_in_pair / (frames_per_pair - 1)

            # print(f"{frame_in_pair=} {start_idx=} {end_idx=} start_mult={start_mult:.3f} end_mult={end_mult:.3f}")
            latent_in = start_mult * start_latent + end_mult * end_latent
            out = decoder_fn(latent_in)

            # make the image tensor into an image then make all images the same
            # (output) size.
            frame_image = to_pil(out[0].detach().cpu(), image_size)

            # annotate the image
            extra_height = 20
            font_size = extra_height
            font = ImageFont.truetype(Roboto, font_size)
            frame_bigger = Image.new("RGB", (image_size, image_size + extra_height))
            frame_bigger.paste(frame_image, box=(0, 0))
            draw = ImageDraw.ImageDraw(frame_bigger)
            imgidx_start, imgidx_end = image_idxs[start_idx], image_idxs[end_idx]
            start_val = int(start_mult * 256)
            end_val = int(end_mult * 256)
            if start_idx % 2 == 1:
                imgidx_start, imgidx_end = imgidx_end, imgidx_start
                start_val, end_val = end_val, start_val
            draw.rectangle(xy=(0, image_size, image_size, image_size), fill='gray')
            draw.text(xy=(0, image_size), text=str(imgidx_start), font=font, fill=(start_val, start_val, start_val))
            draw.text(xy=(image_size // 2, image_size), text=str(imgidx_end), font=font, fill=(end_val, end_val, end_val))
            
            # image_frames.append(frame_image)
            image_frames.append(frame_bigger)


        out_image = image_frames[0]
        out_image.save(filename, format="GIF", append_images=image_frames[1:], save_all=True, duration=1.0 / cfg.fps, loop=0)
        print(f"  saved {filename}")

# %%
