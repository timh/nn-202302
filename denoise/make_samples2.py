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

MODES = "rand-latent interp roundtrip denoise-random denoise-steps denoise-images".split()
Mode = Literal["rand-latent", "interp", "roundtrip", "denoise-random", "denoise-steps", "denoise-images"]

Model = Union[vae.VarEncDec, denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]

# python gen_samples2.py -nc VarEncDec -b 16 -s time -m rand-latent -a 'nepochs > 100'
class Config(cmdline.QueryConfig):
    mode: Mode
    output: str
    image_dir: str
    output_image_size: int

    steps: int
    nrows: int

    experiments: List[Experiment] = None
    all_image_idxs: List[int] = None
    dataset_for_size: Dict[int, Dataset] = None
    random_latents: Dict[str, Tensor] = None

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="interp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--rows", dest='nrows', default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="alex-many-1024")
        self.add_argument("-i", "--output_image_size", type=int, default=None)

        self.dataset_for_size = dict()
    
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

        if self.mode == 'rand-latent':
            self.random_latents = dict()

            for exp in self.experiments:
                latent_dim = exp.net_latent_dim
                latent_dim_str = str(latent_dim)
                if latent_dim_str in self.random_latents:
                    continue
                latent_dim_batch = [cfg.nrows, *latent_dim]
                self.random_latents[latent_dim_str] = torch.randn(size=latent_dim_batch)

        return res
    
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

    net: Model
    net_path: Path
    image_size: int

    vae_net: vae.VarEncDec = None
    vae_path: Path

    cache: LatentCache

    def __init__(self, cfg: Config, column: int, exp: Experiment):
        self.cfg = cfg
        self.column = column
        self.exp = exp
        self.run = exp.get_run()

        self.net_path = self.run.checkpoint_path
        self.net = dn_util.load_model(self.net_path).to(cfg.device)
        if isinstance(self.net, unet.Unet):
            self.vae_path = exp.vae_path
            self.vae_net = dn_util.load_model(Path(exp.vae_path))
        
        self.image_size = self.net.image_size
        self.dataset = self.cfg.get_dataset(self.image_size)

        self.cache = LatentCache(net=self.net, net_path=self.net_path, 
                                 batch_size=cfg.batch_size,
                                 dataset=self.dataset, device=cfg.device)

    def gen(self, row: int) -> Tensor:
        if self.cfg.mode == 'roundtrip':
            return self.gen_roundtrip(row)
        if self.cfg.mode == 'interp':
            return self.gen_interp(row)
        if self.cfg.mode == 'rand-latent':
            return self.gen_rand_latent(row)

    def gen_roundtrip(self, row: int) -> Tensor:
         idx = self.cfg.all_image_idxs[row]
         samples = self.cache.samples_for_idxs([idx])
         return self.cache.decode(samples)[0]

    def gen_interp(self, row: int) -> Tensor:
        idx0, idx1 = self.cfg.all_image_idxs[0:2]
        lat_start, lat_end = self.cache.samples_for_idxs([idx0, idx1])

        amount_start = (self.cfg.nrows - row) / self.cfg.nrows
        amount_end = row / self.cfg.nrows
        latent_lerp = amount_start * lat_start + amount_end * lat_end
        return self.cache.decode([latent_lerp])[0]
    
    def gen_rand_latent(self, row: int) -> Tensor:
        latent_dim_str = str(self.exp.net_latent_dim)
        latent = self.cfg.random_latents[latent_dim_str][row]
        latent = latent.to(self.cfg.device)
        return self.cache.decode([latent])[0]

    def get_row_labels(self) -> List[str]:
        if self.cfg.mode == 'roundtrip':
            return [f"rt {row}" for row in self.cfg.all_image_idxs[:self.cfg.nrows]]
        elif self.cfg.mode == 'interp':
            return [f"{cfg.nrows - row} / {row}" for row in range(self.cfg.nrows)]
        elif self.cfg.mode == 'rand-latent':
            return [f"rand lat {row}" for row in range(self.cfg.nrows)]
        return list()
    

if __name__ == "__main__":
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
        print(f"{column + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")
        if column == 0:
            state = first_state
        else:
            state = State(cfg=cfg, column=column, exp=exp)

        for row in range(cfg.nrows):
            image_t = state.gen(row)
            grid.draw_tensor(col=column, row=row, image_t=image_t)

    grid._image.save(f"make_samples2.png")
