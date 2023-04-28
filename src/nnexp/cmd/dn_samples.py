import sys
from pathlib import Path
from typing import List, Literal
from functools import partial
from dataclasses import dataclass

import torch
from torch import Tensor

from nnexp.experiment import Experiment
from nnexp.images import image_util
from nnexp.denoise import dn_util, imagegen
from nnexp.utils import cmdline

class Config(cmdline.QueryConfig):
    image_dir: str
    output_image_size: int
    seed: int

    # nrepeats: int
    steps: int
    nrows: int

    # deterministic: bool
    # scale_noise: float

    _nimages: int
    embeds: List[Tensor] = None
    ds_idxs: List[int]
    text: List[str]

    _scale_str: str
    scale: float = 1.0
    scale_max: float = None
    scale_count: int = 5

    _guidance_str: str
    guidance: float = 1.0
    guidance_max: float = None

    experiments: List[Experiment] = None
    igen: imagegen.ImageGen

    def __init__(self):
        super().__init__()
        self.add_argument("--nrows", "--row", default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="images.2018-2020")
        self.add_argument("-I", "--output_image_size", type=int, default=None)
        self.add_argument("--seed", type=int, default=None)

        self.add_argument("--steps", type=int, default=60, help="denoising steps")

        # self.add_argument("-D", "--deterministic", default=False, action='store_true', 
        #                   help="use the same random values for noise generation across batches")
        # self.add_argument("--scale_noise", default=1.0, type=float,
        #                   help="to prevent unusable flicker when denoising, scale noise used by denoiser")

        self.add_argument("-n", "--nimages", dest='_nimages', default=None, type=int)
        self.add_argument("-i", "--images", dest='ds_idxs', default=list(), nargs='+', type=int)
        self.add_argument("-t", "--text", default=list(), type=str, nargs='+',
                          help="use clip embedding from given text strings to drive denoise. sets --repeat")

        self.add_argument("-S", "--scale", dest='_scale_str', default=None)
        self.add_argument("-g", "--guidance", dest='_guidance_str', default=None)
    
    def parse_args(self) -> 'Config':
        res = super().parse_args()

        if self.seed is None:
            self.seed = torch.randint(0, 2**31, size=(1,)).item()

        self.experiments = [exp for exp in super().list_experiments() if getattr(exp, 'is_denoiser', None)]
        if not len(self.experiments):
            self.error("no experiments!")
        
        if self.sort_key == 'time':
            self.experiments = list(reversed(self.experiments))
        
        if not self.output_image_size:
            image_sizes = [dn_util.exp_image_size(exp) for exp in self.experiments]
            self.output_image_size = max(image_sizes)

        self.igen = imagegen.ImageGen(image_dir=self.image_dir,
                                      output_image_size=self.output_image_size, 
                                      device=self.device, batch_size=self.batch_size)

        if self.ds_idxs or self._nimages:
            self.embeds = list()

            ds = self.igen.get_dataset(512)
            if not self.ds_idxs:
                self.ds_idxs = torch.randint(low=0, high=len(ds), size=(self._nimages,)).tolist()

            for ds_idx in self.ds_idxs:
                image_t = ds[ds_idx][0]
                image = image_util.tensor_to_pil(image_t)
                embed = self.igen._clip_cache.encode_images([image])[0]
                self.embeds.append(embed)
        elif self.text:
            self.embeds = self.igen._clip_cache.encode_text(self.text)
        else:
            self.embeds = None
        
        if (self._guidance_str or self._scale_str) and self.embeds is None:
            self.error("must have text or images to use scale or guidance")
        
        if self._guidance_str:
            if "-" in self._guidance_str:
                self.guidance, self.guidance_max = map(int, self._guidance_str.split("-"))
            else:
                self.guidance = int(self._guidance_str)
        
        if self._scale_str:
            if "-" in self._scale_str:
                self.scale, self.scale_max = map(int, self._scale_str.split("-"))
            else:
                self.scale = int(self._scale_str)
        
        return res
    
    def output_path(self) -> Path:
        path_parts: List[str] = []

        if self.text:
            path_parts.append("clip_text_" + "-".join(self.text))
        if self.ds_idxs:
            path_parts.append("images_" + "_".join(map(str, self.ds_idxs)))

        if self.scale:
            path_parts.append(f"scale_{self.scale:.1f}")
            if self.scale_max:
                path_parts[-1] += f"-{self.scale_max}"

        if self.guidance:
            path_parts.append(f"cfg_{self.guidance:.1f}")
            if self.guidance_max:
                path_parts[-1] += f"-{self.guidance_max}"
                path_parts.append(f"cfg_max_{self.guidance_max:.1f}")

        path_parts.append(f"seed_{self.seed}")
        
        return Path(image_util.DEFAULT_DIR, "dn-" + ",".join(path_parts) + ".png")

    def list_experiments(self) -> List[Experiment]:
        return self.experiments

    def get_ncol_per_exp(self) -> int:
        res = len(self.embeds) if self.embeds is not None else 1
        if self.scale_max and self.guidance_max:
            res *= self.scale_count
        return res
    
    def get_ncols(self) -> int:
        return len(self.experiments) * self.get_ncol_per_exp()

    def get_col_labels(self) -> List[str]:
        base_labels = [dn_util.exp_descr(exp) for exp in self.experiments]
        ncols = self.get_ncols()
        col_per = self.get_ncol_per_exp()

        labels: List[str] = ["" for _ in range(ncols)]
        for exp_idx, label in enumerate(base_labels):
            labels[exp_idx * col_per] = label

        return labels

@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()

    col_labels = cfg.get_col_labels()
    row_labels = []

    # print(f"denoise with clip scale {cfg.scale:.1f} to {cfg.scale_max:.1f}")
    # print(f"denoise with guidance {cfg.guidance:.1f} to {cfg.guidance_max:.1f}")

    ncols = cfg.get_ncols()
    nrows = cfg.nrows
    if cfg.ds_idxs:
        nrows = cfg.nrows + 1

    grid = image_util.ImageGrid(ncols=ncols, nrows=nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)
    if cfg.ds_idxs:
        ds = cfg.igen.get_dataset(512)
        col_per = cfg.get_ncol_per_exp()
        for exp_idx in range(len(exps)):
            for exp_col, ds_idx in enumerate(cfg.ds_idxs):
                image = image_util.tensor_to_pil(ds[ds_idx][0], image_size=cfg.output_image_size)
                col = exp_idx * col_per + exp_col
                grid.draw_image(col=col, row=nrows - 1, image=image, annotation=str(ds_idx))

    for exp_idx, exp in enumerate(exps):
        exp_run = exp.get_run(loss_type='vloss')
        gen_exp = cfg.igen.for_run(exp, exp_run)

        print()
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")

        all_latents: List[Tensor] = list()
        all_guidance: List[Tensor] = list()
        all_scale: List[Tensor] = list()
        all_annotations: List[str] = list()
        all_embeds: List[Tensor] = list()

        num_exp_rpts = len(cfg.embeds) if cfg.embeds is not None else 1
        for exp_rpt in range(num_exp_rpts):
            scale: List[Tensor] = list()
            guidance: List[Tensor] = list()
            annotations: List[Tensor] = list()

            if cfg.scale_max and cfg.guidance_max:
                latents = gen_exp.get_random_latents(0, 1) * cfg.nrows * cfg.scale_count
                for scale_val in torch.linspace(start=cfg.scale, end=cfg.scale_max, steps=cfg.scale_count).tolist():
                    scale.extend([scale_val] * cfg.nrows)
                    guide_values = torch.linspace(start=cfg.guidance, end=cfg.guidance_max, steps=cfg.nrows).tolist()
                    for guide_val in guide_values:
                        guidance.append(guide_val)
                        annotations.append(f"guidance {guide_val:.3f}, scale {scale_val:.3f}")
                    
            elif cfg.scale_max:
                latents = gen_exp.get_random_latents(0, 1) * cfg.nrows
                scale = torch.linspace(start=cfg.scale, end=cfg.scale_max, steps=cfg.nrows).tolist()
                annotations = [f"scale {scale_val:.3f}" for scale_val in scale]
                guidance = [cfg.guidance] * cfg.nrows
            elif cfg.guidance_max:
                latents = gen_exp.get_random_latents(0, 1) * cfg.nrows
                guidance = torch.linspace(start=cfg.guidance, end=cfg.guidance_max, steps=cfg.nrows).tolist()
                annotations = [f"guidance {guide_val:.3f}" for guide_val in guidance]
                scale = [cfg.scale] * cfg.nrows
            else:
                latents = gen_exp.get_random_latents(0, cfg.nrows)
                guidance = [cfg.guidance] * cfg.nrows
                scale = [cfg.scale] * cfg.nrows
                annotations = [""] * cfg.nrows

            all_latents.extend(latents)
            all_guidance.extend(guidance)
            all_scale.extend(scale)
            all_annotations.extend(annotations)
            if cfg.embeds is None:
                all_embeds = None
            else:
                embeds = [cfg.embeds[exp_rpt]] * len(latents)
                all_embeds.extend(embeds)

        torch.manual_seed(cfg.seed)

        images = gen_exp.gen_denoise_ddim(steps=cfg.steps, latents=all_latents,
                                          clip_embeds=all_embeds, clip_guidance=all_guidance, clip_scale=all_scale)

        for i, (image, annotation) in enumerate(zip(images, all_annotations)):
            row = i % cfg.nrows
            col = exp_idx * cfg.get_ncol_per_exp() + (i // cfg.nrows)
            grid.draw_image(col=col, row=row, image=image, annotation=annotation)

    out_path = cfg.output_path()
    grid._image.save(out_path)
    print(f"wrote to {out_path}")

if __name__ == "__main__":
    main()
    
