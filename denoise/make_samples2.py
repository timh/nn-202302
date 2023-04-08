import sys
from pathlib import Path
from typing import List, Literal

import torch
from torch import Tensor

sys.path.append("..")
from experiment import Experiment, ExpRun
import image_util
import dn_util
import cmdline
import imagegen

MODES = ("random interp roundtrip "
         "denoise-random denoise-steps denoise-images").split()
Mode = Literal["random", "interp", "roundtrip", 
               "denoise-random", "denoise-steps", "denoise-images"]

HELP = """
              random: start with random inputs.
              interp: interpolate between two random images. if using a denoiser, it will do denoising subtraction.
           roundtrip: take N images, and run them through the whole network(s)
      denoise-random: denoise random latents for --steps steps
       denoise-steps: denoise random latents for steps range(1, timesteps-1, timesteps//nrows)
      denoise-images: add noise to images and denoise them, each in one step
"""
# python make_samples.py -nc Unet -b 8 -a 'nepochs > 100' 'ago < 12h' 'loss_type = l1' 
#   -m denoise-steps --steps 600 --repeat 4
class Config(cmdline.QueryConfig):
    mode: Mode
    output: str
    image_dir: str
    output_image_size: int
    nrepeats: int
    denoise_steps: int
    noise_steps: int

    steps: int
    nrows: int

    experiments: List[Experiment] = None

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="interp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--rows", dest='nrows', default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="images.alex-1024")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--repeat", dest='nrepeats', type=int, default=1)
        self.add_argument("-n", "--steps", dest='steps', type=int, default=300, 
                          help="denoising steps for denoise-random, denoise-steps, denoise-images")
        self.add_argument("-N", "--noise_steps", dest='noise_steps', type=int, default=100, 
                          help="steps of noise to add for mode=denoise-images")
    
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
    
    def list_experiments(self) -> List[Experiment]:
        return self.experiments
    
    def get_col_labels(self) -> List[str]:
        return [dn_util.exp_descr(exp) for exp in self.experiments]
    
@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    print(f"{cfg.batch_size=}")

    exps = cfg.list_experiments()
    ncols = len(exps) * cfg.nrepeats

    col_labels = cfg.get_col_labels()
    # row_labels = first_state.get_row_labels()
    row_labels = []

    if cfg.nrepeats > 1:
        col_labels_new: List[str] = ["" for _ in range(len(exps) * cfg.nrepeats)]
        for exp_idx, label in enumerate(col_labels):
            col_labels_new[exp_idx * cfg.nrepeats] = label
        col_labels = col_labels_new

    grid = image_util.ImageGrid(ncols=ncols, nrows=cfg.nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)
    gen = imagegen.ImageGen(image_dir=cfg.image_dir, output_image_size=cfg.output_image_size, device=cfg.device, batch_size=cfg.batch_size)
    for exp_idx, exp in enumerate(exps):
        run = exp.get_run()
        column = exp_idx * cfg.nrepeats

        print()
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")

        for repeat_idx in range(cfg.nrepeats):
            start_idx = repeat_idx * cfg.nrows
            end_idx = start_idx + cfg.nrows
            image_idxs = list(range(start_idx, end_idx))

            if cfg.nrepeats > 1:
                print(f"  repeat {repeat_idx + 1}/{cfg.nrepeats}")
            if cfg.mode == 'interp':
                start_idx = repeat_idx * 2
                end_idx = start_idx + 1
                start, end = gen.get_image_latents(exp, run, image_idxs=[start_idx, end_idx], shuffled=True)
                images = gen.gen_lerp(exp=exp, run=run, 
                                      start=start, end=end, steps=cfg.nrows)
            elif cfg.mode == 'roundtrip':
                images = gen.gen_roundtrip(exp=exp, run=run,
                                           image_idxs=image_idxs, shuffled=True)
            elif cfg.mode == 'random':
                images = gen.gen_random(exp=exp, run=run,
                                        start_idx=start_idx, end_idx=end_idx)
            # elif cfg.mode == 'denoise-images':
            #     pass
            # elif cfg.mode == 'denoise-steps':
            #     pass
            elif cfg.mode == 'denoise-random':
                latents = gen.gen_random(exp, run, start_idx=start_idx, end_idx=end_idx)
                images = gen.gen_denoise_full(exp, run, steps=cfg.steps, latents=latents)

            else:
                raise NotImplementedError(f"{cfg.mode} not implemented")

            for row, image in enumerate(images):
                grid.draw_image(col=column + repeat_idx, row=row, image=image)
            # images = gen.
            # for row in range(cfg.nrows):
            #     grid.draw_tensor()

    grid._image.save(f"make_samples2.png")

if __name__ == "__main__":
    main()
    
