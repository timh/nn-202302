import sys
from pathlib import Path
from typing import List, Union, Literal, Dict
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append("..")
from experiment import Experiment, ExpRun
import image_util
import dn_util
import cmdline
from models import vae, denoise, unet, ae_simple, linear
from latent_cache import LatentCache
import noisegen

MODES = ("random interp roundtrip "
         "denoise-random denoise-steps denoise-images").split()
Mode = Literal["random", "interp", "roundtrip", "denoise-random", "denoise-steps", "denoise-images"]

ModelType = Union[vae.VarEncDec, denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]

HELP = """
        random: start with random inputs.
        interp: interpolate between two random images. if using a denoiser, it will do denoising subtraction.
     roundtrip: take N images, and run them through the whole network(s)
denoise-random: denoise random latents for --steps steps
 denoise-steps: denoise random latents for steps range(1, timesteps-1, timesteps//nrows)
denoise-images: add noise to images and denoise them
"""
# python gen_samples2.py -nc VarEncDec -b 16 -s time -m rand-latent -a 'nepochs > 100'
class Config(cmdline.QueryConfig):
    mode: Mode
    output: str
    image_dir: str
    output_image_size: int
    denoise_steps: int

    steps: int
    nrows: int

    experiments: List[Experiment] = None
    all_image_idxs: List[int] = None
    dataset_for_size: Dict[int, Dataset] = dict()
    random_latents: Dict[str, Tensor] = dict()

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="interp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--rows", dest='nrows', default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="alex-many-1024")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("-n", "--denoise_steps", type=int, default=300)
    
    def parse_args(self) -> 'Config':
        res = super().parse_args()

        self.experiments = super().list_experiments()
        if not len(self.experiments):
            self.error("no experiments!")
        
        if self.sort_key == 'time':
            self.experiments = list(reversed(self.experiments))
        
        if not self.output_image_size:
            image_sizes = [dn_util.exp_image_size(exp) for exp in self.experiments]
            self.output_image_size = max(image_sizes)

        return res
    
    def get_rand_latent(self, latent_dim: List[int], row: int) -> Tensor:
        latent_dim_str = str(latent_dim)
        if latent_dim_str not in self.random_latents:
            latent_dim_batch = [self.nrows, *latent_dim]
            self.random_latents[latent_dim_str] = torch.randn(size=latent_dim_batch, device=self.device)
        return self.random_latents[latent_dim_str][row]
    
    def list_experiments(self) -> List[Experiment]:
        return self.experiments
    
    def get_col_labels(self) -> List[str]:
        return [dn_util.exp_descr(exp) for exp in self.experiments]
    
    def get_dataset(self, image_size: int) -> Dataset:
        if image_size not in self.dataset_for_size:
            dataset, _ = \
                image_util.get_datasets(image_size=image_size, image_dir=self.image_dir,
                                        train_split=1.0)
            self.dataset_for_size[image_size] = dataset

        dataset = self.dataset_for_size[image_size]

        if self.all_image_idxs is None:
            self.all_image_idxs = list(range(len(dataset)))
            random.shuffle(self.all_image_idxs)
        
        return dataset

"""
instantiated for each Experiment
"""
class State:
    cfg: Config
    column: int
    exp: Experiment
    run: ExpRun

    net: ModelType
    net_path: Path
    image_size: int

    vae_net: vae.VarEncDec = None
    vae_path: Path = None

    cache: LatentCache
    sched: noisegen.NoiseSchedule = None
    denoised_sofar: Tensor = None
    denoised_steps_sofar: int = 0

    def __init__(self, cfg: Config, column: int, exp: Experiment):
        self.cfg = cfg
        self.column = column
        self.exp = exp
        self.run = exp.get_run()

        self.net_path = self.run.checkpoint_path
        self.net = dn_util.load_model(self.net_path).to(cfg.device)
        if isinstance(self.net, unet.Unet):
            self.vae_path = exp.vae_path
            self.vae_net = dn_util.load_model(Path(exp.vae_path)).to(cfg.device)
            self.sched = noisegen.make_noise_schedule(type='cosine', timesteps=300, noise_type='normal')
            self.denoised_sofar = None
            self.denoised_steps_sofar = 0
        
        self.image_size = dn_util.exp_image_size(self.exp)
        self.dataset = self.cfg.get_dataset(self.image_size)

        cache_net = self.vae_net or self.net
        cache_path = self.vae_path or self.net_path
        self.cache = LatentCache(net=cache_net, net_path=cache_path,
                                 batch_size=cfg.batch_size,
                                 dataset=self.dataset, device=cfg.device)
        
    def gen(self, row: int) -> Tensor:
        if self.cfg.mode == 'roundtrip':
            return self.gen_roundtrip(row)
        if self.cfg.mode == 'interp':
            return self.gen_interp(row)
        if self.cfg.mode == 'random':
            return self.gen_random(row)
        if self.cfg.mode == 'denoise-steps':
            return self.gen_denoise_steps(row)

    def gen_roundtrip(self, row: int) -> Tensor:
         idx = self.cfg.all_image_idxs[row]
         samples = self.cache.samples_for_idxs([idx])

         if isinstance(self.net, unet.Unet):
             dn_in = torch.stack(samples)
             dn_out = self.net(dn_in)
             samples = dn_out

         return self.cache.decode(samples)[0]

    def gen_interp(self, row: int) -> Tensor:
        idx0, idx1 = self.cfg.all_image_idxs[0:2]
        lat_start, lat_end = self.cache.samples_for_idxs([idx0, idx1])

        amount_start = (self.cfg.nrows - row) / self.cfg.nrows
        amount_end = row / self.cfg.nrows
        latent_lerp = amount_start * lat_start + amount_end * lat_end

        if isinstance(self.net, unet.Unet):
            # denoise it??
            noised_in = latent_lerp.unsqueeze(0).to(self.cfg.device)
            noise_out = self.net(noised_in)
            latent_lerp = (noised_in - noise_out)[0]

        return self.cache.decode([latent_lerp])[0]
    
    def gen_random(self, row: int) -> Tensor:
        if isinstance(self.net, unet.Unet):
            latent = self.cfg.get_rand_latent(self.vae_net.latent_dim, row)
            dn_in = latent.unsqueeze(0)
            dn_out = self.net(dn_in)
            latent = dn_out[0]
        else:
            latent = self.cfg.get_rand_latent(self.net.latent_dim, row)

        return self.cache.decode([latent])[0]

    def _denoise_steps(self, row: int) -> int:
        steps = row / (self.cfg.nrows - 1)
        steps = int(steps * (self.sched.timesteps - 3)) + 1
        return steps

    def gen_denoise_steps(self, row: int) -> Tensor:
        if not isinstance(self.net, unet.Unet):
            return torch.zeros((3, self.image_size, self.image_size))

        if self.denoised_sofar is None:
            self.denoised_sofar = self.cfg.get_rand_latent(self.vae_net.latent_dim, 0).unsqueeze(0)

        steps = self._denoise_steps(row)
        print(f"  col {self.column}, row {row}: steps {steps}")
        while self.denoised_steps_sofar < steps:
            dn_in = self.denoised_sofar
            steps_in = self.sched.timesteps - self.denoised_steps_sofar - 2
            # print(f"    steps_in {steps_in}")
            dn_out = self.sched.gen_frame(net=self.net, inputs=dn_in, timestep=steps_in)
            self.denoised_steps_sofar += 1
            self.denoised_sofar = dn_out

        denoised = self.denoised_sofar[0]

        return self.cache.decode([denoised])[0]

    def get_row_labels(self) -> List[str]:
        if self.cfg.mode == 'roundtrip':
            return [f"rt {row}" for row in self.cfg.all_image_idxs[:self.cfg.nrows]]
        elif self.cfg.mode == 'interp':
            return [f"{self.cfg.nrows - row} / {row}" for row in range(self.cfg.nrows)]
        elif self.cfg.mode == 'random':
            return [f"rand lat {row}" for row in range(self.cfg.nrows)]
        elif self.cfg.mode == 'denoise-steps':
            steps_list = [self._denoise_steps(row) for row in range(self.cfg.nrows)]
            return [f"dn {steps}" for steps in steps_list]
        return list()

@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()
    ncols = len(exps)

    first_state = State(cfg=cfg, column=0, exp=exps[0])

    col_labels = cfg.get_col_labels()
    row_labels = first_state.get_row_labels()
    grid = image_util.ImageGrid(ncols=ncols, nrows=cfg.nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)

    for column, exp in enumerate(exps):
        print()
        print(f"{column + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")
        if column == 0:
            state = first_state
        else:
            state = State(cfg=cfg, column=column, exp=exp)

        for row in range(cfg.nrows):
            image_t = state.gen(row)
            grid.draw_tensor(col=column, row=row, image_t=image_t)

    grid._image.save(f"make_samples2.png")

if __name__ == "__main__":
    main()
    
