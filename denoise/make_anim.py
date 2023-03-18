# %%
from typing import List, Tuple
from pathlib import Path
import argparse
import tqdm
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
import cv2

import sys
sys.path.append("..")
import model
import model_new
import dn_util
import model_util
from experiment import Experiment

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
    walk_after: int
    walk_mult: float

    do_loop: bool

    batch_size: int

    def startend_mults(self, frame: int) -> Tuple[float, float]:
        frame_in_pair = frame % self.frames_per_pair
        start_mult = (self.frames_per_pair - frame_in_pair - 1) / (self.frames_per_pair - 1)
        end_mult = frame_in_pair / (self.frames_per_pair - 1)
        return start_mult, end_mult

    def img_idx(self, frame: int) -> int:
        imgidx = frame // self.frames_per_pair
        return imgidx


def annotate(cfg: Config, frame: int, image: Image.Image):
    # annotate the image
    extra_height = 20
    font_size = int(extra_height * 2 / 3)
    font = ImageFont.truetype(fonts.ttf.Roboto, font_size)

    anno_image = Image.new("RGB", (image_size, image_size + extra_height))
    anno_image.paste(image, box=(0, 0))
    draw = ImageDraw.ImageDraw(anno_image)

    imgidx = cfg.img_idx(frame)
    start_mult, end_mult = cfg.startend_mults(frame)
    dsidx_start, dsidx_end = dataset_idxs[imgidx:imgidx + 2]
    start_val = int(start_mult * 255)
    end_val = int(end_mult * 255)
    # print(f"{start_val=} {end_val=}")
    start_x = imgidx * image_size / cfg.num_images
    end_x = (imgidx + 1) * image_size / cfg.num_images
    draw.rectangle(xy=(0, image_size, image_size, image_size), fill='gray')
    draw.text(xy=(start_x, image_size), text=str(dsidx_start), font=font, fill=(start_val, start_val, start_val))
    draw.text(xy=(end_x, image_size), text=str(dsidx_end), font=font, fill=(end_val, end_val, end_val))

    return anno_image

# checkpoints, num_images, num_frames, frames_per_pair
def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", dest='batch_size', type=int, default=16)
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
    parser.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
    parser.add_argument("-i", "--image_size", type=int, default=None)
    parser.add_argument("-I", "--dataset_idxs", type=int, nargs="+", default=None, help="specify the image positions in the dataset")
    parser.add_argument("-n", "--num_images", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--random", default=False, action='store_true', help="use random latent images instead of loading them")
    parser.add_argument("--walk_between", default=False, action='store_true', help="randomly perturb the walk from one image to the next")
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
    cfg.checkpoints = list(sorted(cfg.checkpoints, key=lambda cp_tuple: cp_tuple[1].lastepoch_val_loss))

    experiments = [exp for path, exp in cfg.checkpoints]
    for i, exp in enumerate(experiments):
        print(f"{i+1}.", exp.label)
    print()

    dataset_idxs: List[int] = None
    if cfg.dataset_idxs:
        cfg.num_images = len(dataset_idxs)
    
    if cfg.do_loop:
        if cfg.dataset_idxs is not None:
            cfg.dataset_idxs.append(dataset_idxs[0])
        cfg.num_images += 1

    if cfg.num_frames is None:
        cfg.num_frames = (cfg.num_images - 1) * cfg.frames_per_pair
    cfg.frames_per_pair = cfg.num_frames // (cfg.num_images - 1)

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

if __name__ == "__main__":
    cfg = parse_args()

    for i, (path, exp) in enumerate(cfg.checkpoints):
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
                                                image_dir=cfg.image_dir, batch_size=1, shuffle=False)
        dataset = dataloader.dataset
        if cfg.random:
            dataset_idxs = list(range(cfg.num_images))
            dslatents = [torch.randn(exp.net.encoder_out_dim).unsqueeze(0).to(device) for _ in range(cfg.num_images)]
        else:
            if not cfg.dataset_idxs:
                dataset_idxs = [i.item() for i in torch.randint(0, len(dataset), (cfg.num_images,))]
                if cfg.do_loop:
                    dataset_idxs[-1] = dataset_idxs[0]
                print(f"--dataset_idxs", " ".join(map(str, dataset_idxs)))

            image_tensors = [dataset[dsidx][1] for dsidx in dataset_idxs]
            image_tensors = [image_tensor.unsqueeze(0).to(device) for image_tensor in image_tensors]
            dslatents = [encoder_fn(image_tensor) for image_tensor in image_tensors]


        # build filename and video container
        parts = [f"tloss_{exp.lastepoch_train_loss:.3f}",
                 f"vloss_{exp.lastepoch_val_loss:.3f}",
                 exp.label]
        filename = f"anim_{i}--" + ",".join(parts) + ".mp4"
        print()
        print(f"{i+1}/{len(cfg.checkpoints)} {filename}:")

        # compute latents for all the frames, in batch.
        batch_latents_in: List[Tensor] = list()
        for frame, latent_in in enumerate(compute_latents(cfg, dslatents)):
            batch_num = frame // cfg.batch_size
            sample_num = frame % cfg.batch_size
            if batch_num >= len(batch_latents_in):
                batch_latent_size = (cfg.batch_size, *[d for d in latent_in.shape[1:]])
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
                image = to_pil(sample_out.detach().cpu(), image_size)
                image = annotate(cfg, frame, image)
                if anim_out is None:
                    anim_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, 
                                                (image.width, image.height))

                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                anim_out.write(image_cv)
        anim_out.release()

# %%
