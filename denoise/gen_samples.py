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
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append("..")
from experiment import Experiment
import image_util
import model_util
import dn_util
import cmdline
from models import vae, denoise, unet, ae_simple, linear
import dataloader
from models.mtypes import VarEncoderOutput

from latent_cache import LatentCache

# TODO this code is a big mess.

_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font: ImageFont.ImageFont = None
_padding = 2
_minx = 0
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
        self.add_argument("-m", "--mode", default="interp", choices=["rand-latent", "interp", "roundtrip"])
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--steps", default=20, type=int, help="number of steps (for 'random', 'latent')")
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--amount_min", type=float, default=0.0)
        self.add_argument("--amount_max", type=float, default=1.0)

class State:
    # across experiments
    rand_latent_for_dim: Dict[List[int], List[int]] = dict()
    all_image_idxs: List[int] = None
    img_dataset: Dataset = None

    # per experiment
    path: Path
    exp: Experiment
    net: Union[vae.VarEncDec, denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]
    vae_net: vae.VarEncDec = None
    img_dataset: Dataset = None
    lat_dataset: dataloader.EncoderDataset = None

    # 
    cache_img2lat: LatentCache
    cache_lat2lat: LatentCache
    latent_dim: List[int]             # dimension of inner latents
    latents: List[VarEncoderOutput]   # inner latents - 

    def setup(self, exp: Experiment, nrows: int):
        self.exp = exp
        path = self.exp.cur_run().checkpoint_path
        self.path = path

        try:
            model_dict = torch.load(path)
            net = dn_util.load_model(model_dict).to(cfg.device)
            net.eval()
            self.net = net
        except Exception as e:
            print(f"error processing {path}:", file=sys.stderr)
            raise e

        if getattr(exp, 'is_denoiser', None):
            vae_path = exp.vae_path
            try:
                vae_dict = torch.load(vae_path)
                vae_net = dn_util.load_model(vae_dict).to(cfg.device)
                vae_net.eval()
                self.vae_net = vae_net
                image_size = vae_net.image_size

            except Exception as e:
                print(f"error processing {vae_path}:", file=sys.stderr)
                raise e
        else:
            self.vae_net = None
            image_size = net.image_size
                
        if self.img_dataset is None:
            self.img_dataset, _ = \
                image_util.get_datasets(image_size=image_size, 
                                        image_dir=cfg.image_dir,
                                        train_split=1.0)

        latent_dim: List[int] = None
        cache_lat2lat: LatentCache = None
        cache_img2lat: LatentCache = None
        lat_dataset: dataloader.EncoderDataset = None

        if isinstance(net, vae.VarEncDec):
            latent_dim = net.latent_dim
            cache_img2lat = \
                LatentCache(net=net, net_path=path, batch_size=cfg.batch_size,
                            dataset=self.img_dataset, device=cfg.device)

        elif isinstance(net, denoise.DenoiseModel):
            latent_dim = net.bottleneck_dim
    
            lat_dataset = \
                dataloader.EncoderDataset(vae_net=vae_net, vae_net_path=vae_path,
                                          batch_size=cfg.batch_size, 
                                          base_dataset=self.img_dataset,
                                          device=cfg.device)
            cache_lat2lat = \
                LatentCache(net=net, net_path=path, dataset=lat_dataset,
                            batch_size=cfg.batch_size, device=cfg.device)
            cache_img2lat = \
                LatentCache(net=vae_net, net_path=vae_path, dataset=lat_dataset, 
                            batch_size=cfg.batch_size, device=cfg.device)
        elif getattr(exp, 'is_denoiser'):
            latent_dim = vae_net.latent_dim
    
            cache_img2lat = \
                LatentCache(net=vae_net, net_path=vae_path, dataset=self.img_dataset, 
                            batch_size=cfg.batch_size, device=cfg.device)
            
        else:
            raise Exception(f"not implemented for {type(net)}")

        self.latent_dim = latent_dim
        self.cache_lat2lat = cache_lat2lat
        self.cache_img2lat = cache_img2lat
        self.lat_dataset = lat_dataset

        if cfg.mode == "rand-latent":
            if latent_dim not in self.rand_latent_for_dim:
                self.latents = torch.randn(latent_dim, device=cfg.device)
                self.rand_latent_for_dim[latent_dim] = self.latents
            else:
                self.latents = self.rand_latent_for_dim[latent_dim]

        elif cfg.mode in ["interp", "roundtrip"]:
            if self.all_image_idxs is None:
                import random
                self.all_image_idxs = list(range(len(self.img_dataset)))
                random.shuffle(self.all_image_idxs)
            
            if cfg.mode == "interp":
                # first_latent, last_latent = self.imglat.encode(self.all_image_idxs[:2])
                first_latent, last_latent = self.to_latent(self.all_image_idxs[:2])

                self.latents = [first_latent]
                nimages = nrows - 2
                for i in range(nimages):
                    first_part = (nimages - i - 1) / nimages
                    last_part = (i + 1) / nimages
                    latent = first_latent * first_part + last_latent * last_part
                    self.latents.append(latent)
                self.latents.append(last_latent)
            else:
                self.latents = self.to_latent(self.all_image_idxs[:nrows])

    def to_latent(self, img_idxs: List[int]) -> List[VarEncoderOutput]:
        if isinstance(self.net, denoise.DenoiseModel):
            return self.cache_lat2lat.encouts_for_idxs(img_idxs)
        
        return self.cache_img2lat.encouts_for_idxs(img_idxs)

    def to_image_t(self, latent: VarEncoderOutput) -> Tensor:
        if getattr(self.exp, 'is_denoiser', None):
            if getattr(self.exp, 'predict_stats', None):
                mean_logvar = latent.cat_mean_logvar()
                mean_logvar = mean_logvar.unsqueeze(0).to(cfg.device)
                dec_mean_logvar = self.net(mean_logvar)[0]
                dec_veo = VarEncoderOutput.from_cat(dec_mean_logvar)

                sample = latent.sample(device=cfg.device) - dec_veo.sample(device=cfg.device)
            else:
                dec_in = latent.sample().unsqueeze(0).to(cfg.device)
                sample = self.net(dec_in)[0]

            return self.cache_img2lat.decode([sample])[0]
            
        elif isinstance(self.net, denoise.DenoiseModel):
            sample = latent.sample().unsqueeze(0).to(cfg.device)
            dec_out = self.net.decode(sample)[0]
            return self.cache_img2lat.decode([dec_out])[0]
        
        return self.cache_img2lat.decode([latent.sample()])[0]

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

    exps = cfg.list_experiments()
    exps = [exp for exp in exps if getattr(exp, 'vae_path', None) and getattr(exp, 'image_size', None)]

    image_size = min([exp.image_size for exp in exps])
    print(f"image_size = {image_size}")
    for i, exp in enumerate(exps):
        print(f"{i + 1}. {exp.shortcode}: {exp.net_dim=}, exp.id_fields =", ", ".join(exp.id_fields()))

    # image_size = max([exp.net_image_size for exp in exps])
    # image_size = 512
    # image_size = 256
    output_image_size = cfg.output_image_size or image_size
    nchannels = 3
    ncols = len(exps)
    padded_image_size = output_image_size + _padding

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

    elif cfg.mode == "interp":
        nrows = 10 + 2
        row_labels = ["first"]
        row_labels.extend([f"interp {i}" for i in range(nrows - 2)])
        row_labels.append("last")
        filename = cfg.output or "gen-interp.png"

    elif cfg.mode == "roundtrip":
        nrows = 10
        row_labels = [f"img {i}" for i in range(nrows)]
        filename = cfg.output or "gen-roundtrip.png"

    else: # random
        num_steps = cfg.steps
        nrows = 10
        row_labels = [f"rand {i}" for i in range(nrows)]
        filename = cfg.output or "gen-random.png"
        # uniform distribution for pixel space.
        inputs = noise_fn((nrows, 1, nchannels, image_size, image_size)).to(cfg.device)

    print(f"   ncols: {ncols}")
    print(f"   nrows: {nrows}")
    print(f"    mode: {cfg.mode}")
    print(f"filename: {filename}")

    font_size = max(10, output_image_size // 20)
    _font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    # generate column headers
    max_width = output_image_size + _padding
    exp_descrs: List[str] = list()
    for exp in exps:
        descr = image_util.exp_descr(exp, 
                                     include_loss=False, include_label=False,
                                     extra_field_map={'saved_at_relative': 'rel', 'loss_type': 'loss', 'nepochs': 'nepochs'})
        descr[-1] += ","
        descr.append(exp.net_class)
        if exp.net_class == 'Unet':
            descr.append(f"dim {exp.net_dim},")
            descr.append("dim_mults " + "-".join(map(str, exp.net_dim_mults)) + ",")
            descr.append(f"rnblks {exp.net_resnet_block_groups},")
            descr.append(f"selfcond {exp.net_self_condition},")
        elif exp.net_class == 'AEDenoise':
            descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))
        elif exp.net_class == 'DenoiseLinear':
            descr.append(f"nlayers {exp.net_nlayers}")
            descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))

        descr.append(f"tloss {exp.last_train_loss:.3f}")
        exp_descrs.append(descr)

    col_titles, max_title_height = \
        image_util.fit_strings_multi(exp_descrs, max_width=max_width, font=_font)
    
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
    state = State()
    for col, exp in tqdm.tqdm(list(enumerate(exps))):
        state.setup(exp, nrows)
        for row in range(nrows):
            latent = state.latents[row]
            out_t = state.to_image_t(latent)
            out = image_util.tensor_to_pil(out_t, output_image_size)

            # draw this image
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)
    
    _img.save(filename)