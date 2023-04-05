import sys
from pathlib import Path
from typing import List, Union, Literal, Dict
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append("..")
from experiment import Experiment
import image_util
import dn_util
import cmdline
from models import vae, denoise, unet, ae_simple, linear
import dataloader
import noisegen
from models.mtypes import VarEncoderOutput

from latent_cache import LatentCache

# TODO this code is a big mess.

_img: Image.Image = None
_draw: ImageDraw.ImageDraw = None
_font: ImageFont.ImageFont = None
_padding = 2
_minx = 0
_miny = 0

Mode = Literal["rand-latent", "interp", "roundtrip", "denoise-random", "denoise-steps", "denoise-images"]
Model = Union[vae.VarEncDec, denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]

class Config(cmdline.QueryConfig):
    mode: Mode
    output: str
    image_dir: str
    output_image_size: int

    steps: int
    add_noise_steps: int
    nrows: int
    denoise_steps_list: List[int]

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="interp", 
                          choices=["rand-latent", "interp", "roundtrip", 
                                   "denoise-random", "denoise-steps", "denoise-images"])
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--steps", default=300, type=int, 
                          help="number of denoising timesteps")
        self.add_argument("--add_noise_steps", default=150, type=int, 
                          help="how much noise to add before denoising, in timesteps")
        self.add_argument("--rows", dest='nrows', default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
    
    def parse_args(self) -> 'Config':
        res = super().parse_args()
        self.denoise_steps_list = torch.linspace(start=1, end=cfg.steps - 1, steps=self.nrows).int()
        return res

class State:
    # across experiments
    rand_latents_for_dim: Dict[str, List[VarEncoderOutput]] = dict()
    all_image_idxs: List[int] = None
    img_dataset: Dataset = None

    # per experiment
    exp: Experiment
    net: Model
    net_path: Path
    vae_net: vae.VarEncDec = None
    vae_net_path: Path
    img_dataset: Dataset = None
    lat_dataset: dataloader.EncoderDataset = None
    noise_sched: noisegen.NoiseSchedule
    noised_dataset: dataloader.NoisedDataset

    # 
    cache_img2lat: LatentCache
    cache_lat2lat: LatentCache
    latent_dim: List[int]             # dimension of inner latents
    latents: List[VarEncoderOutput]   # inner latents
    image_size: int                   # size needed for cache_img2lat

    denoise: bool

    def setup(self, exp: Experiment):
        self.exp = exp
        self.net_path = self.exp.get_run().checkpoint_path

        try:
            model_dict = torch.load(self.net_path)
            self.net = dn_util.load_model(model_dict).to(cfg.device)
            self.net.eval()
        except Exception as e:
            print(f"error processing {self.net_path}:", file=sys.stderr)
            raise e

        if getattr(exp, 'is_denoiser', None):
            self.vae_net_path = exp.vae_path
            try:
                vae_dict = torch.load(self.vae_net_path)
                vae_net = dn_util.load_model(vae_dict).to(cfg.device)
                vae_net.eval()
                self.vae_net = vae_net
                self.image_size = vae_net.image_size

                if cfg.mode in ['denoise-images', 'denoise-steps', 'denoise-random']:
                    self.denoise = True

            except Exception as e:
                print(f"error processing {self.vae_net_path}:", file=sys.stderr)
                raise e
        else:
            self.vae_net = None
            self.image_size = self.net.image_size

            if cfg.mode in ['denoise-images', 'denoise-steps', 'denoise-random']:
                raise Exception(f"can't do mode {cfg.mode} with net {exp.shortcode} cuz it's not a denoising model")

        self._setup_loaders()

        if cfg.mode in ["rand-latent", "denoise-random", "denoise-steps"]:
            gen_size = [self.latent_dim[0] * 2, *self.latent_dim[1:]]
            gen_size_str = str(gen_size)

            if gen_size_str in self.rand_latents_for_dim:
                self.latents = self.rand_latents_for_dim[gen_size_str]
                return

            # denoise_steps uses the same latent for all rows, and instead varies
            # the denoise steps. rand-latent and denoise-random use different latents 
            # and the same, fixed number of steps for each row.
            if cfg.mode == "denoise-steps":
                mean_logvar = torch.randn(gen_size, device=cfg.device)
                veo = VarEncoderOutput.from_cat(mean_logvar)
                veos = [veo for _ in range(cfg.nrows)]
                self.rand_latents_for_dim[gen_size_str] = veos
                self.latents = veos
                return

            gen_size = [cfg.nrows, *gen_size]
            mean_logvar_list = torch.randn(gen_size, device=cfg.device)

            veos: List[VarEncoderOutput] = list()
            for i, mean_logvar in enumerate(mean_logvar_list):
                veo = VarEncoderOutput.from_cat(mean_logvar)
                veos.append(veo)
            
            self.rand_latents_for_dim[gen_size_str] = veos
            self.latents = veos

        elif cfg.mode in ["interp", "roundtrip", "denoise-images"]:
            if self.all_image_idxs is None:
                import random
                self.all_image_idxs = list(range(len(self.img_dataset)))
                random.shuffle(self.all_image_idxs)
            
            if cfg.mode == "interp":
                first_latent, last_latent = self.to_latent(self.all_image_idxs[:2])

                self.latents = [first_latent]
                nimages = cfg.nrows - 2
                for i in range(nimages):
                    first_part = (nimages - i - 1) / nimages
                    last_part = (i + 1) / nimages
                    latent = first_latent * first_part + last_latent * last_part
                    self.latents.append(latent)
                self.latents.append(last_latent)
            else:
                self.latents = self.to_latent(self.all_image_idxs[:cfg.nrows])

    def _setup_loaders(self):
        self.latent_dim = None
        self.cache_lat2lat = None
        self.cache_img2lat = None
        self.noised_dataset = None

        if self.img_dataset is None:
            self.img_dataset, _ = \
                image_util.get_datasets(image_size=self.image_size, 
                                        image_dir=cfg.image_dir,
                                        train_split=1.0)

        if isinstance(self.net, vae.VarEncDec):
            self.latent_dim = self.net.latent_dim
            self.cache_img2lat = \
                LatentCache(net=self.net, net_path=self.net_path, 
                            batch_size=cfg.batch_size,
                            dataset=self.img_dataset, device=cfg.device)

        elif isinstance(self.net, denoise.DenoiseModel):
            self.latent_dim = self.net.bottleneck_dim
    
            lat_dataset = \
                dataloader.EncoderDataset(vae_net=self.vae_net, vae_net_path=self.vae_net_path,
                                          batch_size=cfg.batch_size, 
                                          base_dataset=self.img_dataset,
                                          device=cfg.device)
            self.cache_lat2lat = \
                LatentCache(net=self.net, net_path=self.net_path, 
                            dataset=lat_dataset,
                            batch_size=cfg.batch_size, device=cfg.device)
            self.cache_img2lat = \
                LatentCache(net=self.vae_net, net_path=self.vae_net_path, 
                            dataset=lat_dataset, 
                            batch_size=cfg.batch_size, device=cfg.device)

        elif getattr(exp, 'is_denoiser'):
            self.latent_dim = self.vae_net.latent_dim
            self.cache_img2lat = \
                LatentCache(net=self.vae_net, net_path=self.vae_net_path, 
                            dataset=self.img_dataset, 
                            batch_size=cfg.batch_size, device=cfg.device)
            
            if self.denoise:
                self.noise_sched = \
                    noisegen.make_noise_schedule(type='cosine', timesteps=cfg.steps, noise_type='normal')

        else:
            raise Exception(f"not implemented for {type(self.net)}")

    def to_latent(self, img_idxs: List[int]) -> List[VarEncoderOutput]:
        if isinstance(self.net, denoise.DenoiseModel):
            return self.cache_lat2lat.encouts_for_idxs(img_idxs)
        
        return self.cache_img2lat.encouts_for_idxs(img_idxs)

    def to_image_t(self, latent: VarEncoderOutput, row: int) -> Tensor:
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

            # latent to render is in sample
            if self.denoise:
                sample = sample.unsqueeze(0)
                dn_steps = cfg.steps

                if cfg.mode == 'denoise-images':
                    sample, _noise, _amount, _timestep = self.noise_sched.add_noise(sample, timestep=cfg.add_noise_steps)

                elif cfg.mode == 'denoise-steps':
                    dn_steps = cfg.denoise_steps_list[row]

                sample = self.noise_sched.gen(net=self.net, inputs=sample, steps=dn_steps)
                sample = sample[0]

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
    # exps = [exp for exp in exps if getattr(exp, 'vae_path', None) and getattr(exp, 'image_size', None)]

    _image_size = min([getattr(exp, 'image_size', None) or getattr(exp, 'net_image_size', None) for exp in exps])
    print(f"image_size = {_image_size}")
    for i, exp in enumerate(exps):
        print(f"{i + 1}. {exp.shortcode}: {exp.label}")

    # image_size = max([exp.net_image_size for exp in exps])
    # image_size = 512
    # image_size = 256
    output_image_size = cfg.output_image_size or _image_size
    nchannels = 3
    ncols = len(exps)
    padded_image_size = output_image_size + _padding

    filename = cfg.output or f"gen-{cfg.mode}.png"
    if cfg.mode == "denoise-steps":
        row_labels = [f"{s} steps" for s in cfg.denoise_steps_list]

    elif cfg.mode == "latent":
        row_labels = [f"latent {i}" for i in range(cfg.nrows)]

    elif cfg.mode == "interp":
        row_labels = ["first"]
        row_labels.extend([f"interp {i}" for i in range(cfg.nrows - 2)])
        row_labels.append("last")

    elif cfg.mode in ["roundtrip", "denoise-images"]:
        row_labels = [f"img {i}" for i in range(cfg.nrows)]

    else: # random
        row_labels = [f"rand {i}" for i in range(cfg.nrows)]

    print(f"   ncols: {ncols}")
    print(f"   nrows: {cfg.nrows}")
    print(f"    mode: {cfg.mode}")
    print(f"filename: {filename}")

    font_size = max(10, output_image_size // 20)
    _font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    # generate column headers
    max_width = output_image_size + _padding
    exp_descrs: List[str] = list()
    for exp in exps:
        descr = dn_util.exp_descr(exp)

        # descr.append(f"tloss {exp.last_train_loss:.3f}")
        exp_descrs.append(descr)

    col_titles, max_title_height = \
        image_util.fit_strings_multi(exp_descrs, max_width=max_width, font=_font)
    
    _create_image(nrows=cfg.nrows, ncols=ncols, image_size=output_image_size, title_height=max_title_height)

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
        try:
            state.setup(exp)
        except FileNotFoundError as e:
            print(f"{exp.shortcode} checkpoint disappeared, skipping..")
            print(e)
            continue

        for row in range(cfg.nrows):
            latent = state.latents[row]
            out_t = state.to_image_t(latent, row)

            out = image_util.tensor_to_pil(out_t, output_image_size)

            # draw this image
            xy = (_minx + col * padded_image_size, _miny + row * padded_image_size)
            _img.paste(out, box=xy)

            if cfg.mode == 'denoise-steps':
                text = f"denoise {cfg.denoise_steps_list[row]}"
                _left, _top, right, bot = _draw.textbbox(xy=(0, 0), text=text, font=_font)
                tleft = xy[0]
                tright = tleft + right
                tbot = xy[1] + padded_image_size
                ttop = tbot - bot
                _draw.rectangle(xy=(tleft, ttop, tright, tbot), fill='black')
                _draw.text(xy=(tleft, ttop), text=text, font=_font, fill='white')
    
    _img.save(filename)