import sys
from pathlib import Path
from typing import List, Literal
from functools import partial

import torch
from torch import Tensor

from nnexp.experiment import Experiment
from nnexp.images import image_util
from nnexp.denoise import dn_util, imagegen
from nnexp.utils import cmdline

MODES = "random lerp roundtrip".split()
Mode = Literal["random", "lerp", "roundtrip"]

HELP = """
                  random: start with random inputs.
                    lerp: interpolate between two random images. if using a denoiser, it will do denoising subtraction.
               roundtrip: take N images, and run them through the whole network(s)
"""
# python make_samples.py -nc Unet -b 8 -a 'nepochs > 100' 'ago < 12h' 'loss_type = l1' 
#   -m denoise-steps --steps 600 --repeat 4
class Config(cmdline.QueryConfig):
    mode: Mode
    output: str
    image_dir: str
    output_image_size: int
    seed: int

    # nrepeats: int
    nrows: int
    experiments: List[Experiment] = None

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="lerp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--nrows", "--row", default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="images.2018-2020")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--seed", type=int, default=None)

    def parse_args(self) -> 'Config':
        res = super().parse_args()

        if self.seed is None:
            self.seed = torch.randint(0, 2**31, size=(1,)).item()

        self.experiments = super().list_experiments()
        if not len(self.experiments):
            self.error("no experiments!")
        
        if self.sort_key == 'time':
            self.experiments = list(reversed(self.experiments))
        
        if not self.output_image_size:
            image_sizes = [dn_util.exp_image_size(exp) for exp in self.experiments]
            self.output_image_size = max(image_sizes)

        self.gen = imagegen.ImageGen(image_dir=self.image_dir,
                                     output_image_size=self.output_image_size, 
                                     device=self.device, batch_size=self.batch_size)

        return res
    
    def output_path(self) -> Path:
        path_parts = [
            self.mode,
        ]
        path_parts.append(f"seed_{self.seed}")
        
        return Path(image_util.DEFAULT_DIR, "make_samples-" + ",".join(path_parts) + ".png")

    def list_experiments(self) -> List[Experiment]:
        return self.experiments
    
    def get_col_labels(self) -> List[str]:
        return [dn_util.exp_descr(exp) for exp in self.experiments]
    
@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()

    col_labels = cfg.get_col_labels()
    row_labels = []

    ncols = len(exps)
    nrows = cfg.nrows

    grid = image_util.ImageGrid(ncols=ncols, nrows=nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)
    for exp_idx, exp in enumerate(exps):
        exp_run = exp.get_run(loss_type='vloss')
        gen_exp = cfg.gen.for_run(exp, exp_run)

        print()
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")

        for exp_col_idx in range(1):
            torch.manual_seed(cfg.seed)

            if cfg.mode == 'random':
                start_idx = exp_col_idx * cfg.nrows
                end_idx = start_idx + cfg.nrows
                images = gen_exp.gen_random(start_idx=start_idx, end_idx=end_idx)

            elif cfg.mode == 'lerp':
                start_idx = exp_col_idx * 2
                end_idx = start_idx + 1
                start, end = gen_exp.get_image_latents(image_idxs=[start_idx, end_idx], shuffled=True)
                images = gen_exp.gen_lerp(start=start, end=end, steps=cfg.nrows)

            elif cfg.mode == 'roundtrip':
                start_idx = exp_col_idx * cfg.nrows
                end_idx = start_idx + cfg.nrows
                images = gen_exp.gen_roundtrip(image_idxs=list(range(start_idx, end_idx)), shuffled=True)

            for row, image in enumerate(images):
                col = exp_idx * 1 + exp_col_idx
                grid.draw_image(col=col, row=row, image=image)

    out_path = cfg.output_path()
    grid._image.save(out_path)
    print(f"wrote to {out_path}")

if __name__ == "__main__":
    main()
    
