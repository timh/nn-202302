import sys
from pathlib import Path
from typing import List, Literal
from functools import partial

import torch
from torch import Tensor

sys.path.append("..")
from experiment import Experiment
import image_util
import dn_util
import cmdline
import imagegen

MODES = ("random lerp roundtrip "
         "denoise-full denoise-steps denoise-lerp-clip "
         "denoise-grid denoise-scale denoise-guidance "
         "denoise-image-full denoise-image-steps").split()
Mode = Literal["random", "lerp", "roundtrip", 
               "denoise-full", "denoise-steps", "denoise-lerp-clip",
               "denoise-grid", "denoise-scale", "denoise-guidance",
               "denoise-image-full", "denoise-image-steps"]

HELP = """
                  random: start with random inputs.
                    lerp: interpolate between two random images. if using a denoiser, it will do denoising subtraction.
               roundtrip: take N images, and run them through the whole network(s)

            denoise-full: denoise random latents with --steps steps
           denoise-steps: denoise random latents with steps range(1, --steps - 1, --steps // nrows)
            denoise-grid: denoise: gen grid with scale and guidance
           denoise-scale: denoise: gen a column of images for (--clip_scale .. --clip_scale_max)
        denoise-guidance: denoise: gen a column of images for (--clip_guidance .. --clip_guidance_max)

       denoise-lerp-clip: denoise random latents with full steps, with CLIP guidance lerp'ed between (text | image) inputs
      denoise-image-full: add --noise_steps to images then denoise them with --steps
     denoise-image-steps: add --noise_steps to images then denoise them with steps range(1, --steps - 1, --steps // nrows)
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
    denoise_steps: int
    noise_steps: int

    steps: int
    steps_ddim: int
    nrows: int

    deterministic: bool
    scale_noise: float

    img_idxs: List[int]

    nimages: int
    clip_embeds: List[Tensor] = None
    clip_text: List[str]
    clip_model_name: str = "RN50"
    clip_scale: float
    clip_scale_max: float

    clip_guidance: float
    clip_guidance_max: float
    clip_guidance_count: int

    experiments: List[Experiment] = None

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="lerp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--nrows", "--row", default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="images.2018-2020")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--seed", type=int, default=None)

        self.add_argument("-n", "--steps", dest='steps', type=int, default=300, 
                          help="denoising steps")
        self.add_argument("--ddim", dest="steps_ddim", type=int, default=None)
        self.add_argument("-N", "--noise_steps", dest='noise_steps', type=int, default=100, 
                          help="steps of noise")

        self.add_argument("-D", "--deterministic", default=False, action='store_true', 
                          help="use the same random values for noise generation across batches")
        self.add_argument("--scale_noise", default=1.0, type=float,
                          help="to prevent unusable flicker when denoising, scale noise used by denoiser")

        self.add_argument("--nimages", default=None, type=int)
        self.add_argument("--images", dest='img_idxs', default=list(), nargs='+', type=int)
        self.add_argument("--clip_text", default=list(), type=str, nargs='+',
                          help="use clip embedding from given text strings to drive denoise. sets --repeat")

        self.add_argument("--clip_scale", default=1.0, type=float)
        self.add_argument("--clip_scale_max", default=2.0, type=float)
        self.add_argument("--clip_guidance", default=1.0, type=float)
        self.add_argument("--clip_guidance_max", default=2.0, type=float)
        self.add_argument("--clip_guidance_count", default=5, type=int)
    
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
                                     clip_model_name=self.clip_model_name,
                                     device=self.device, batch_size=self.batch_size)

        if self.img_idxs or self.nimages:
            self.clip_embeds = list()

            ds = self.gen.get_dataset(512)
            if not self.img_idxs:
                self.img_idxs = torch.randint(low=0, high=len(ds), size=(self.nimages,)).tolist()

            for i, ds_idx in enumerate(self.img_idxs):
                image_t = ds[ds_idx][0]
                image = image_util.tensor_to_pil(image_t)
                embed = self.gen._clip_cache.encode_images([image])[0]
                self.clip_embeds.append(embed)
        elif self.clip_text:
            self.clip_embeds = self.gen._clip_cache.encode_text(self.clip_text)
        else:
            self.clip_embeds = [None] * self.get_ncol_per_exp()
        
        if self.mode == 'denoise-scale' or self.mode == 'denoise-grid':
            self.clip_scale == self.clip_scale or 1.0
            self.clip_scale_max = self.clip_scale_max or 10.0
        if self.mode == 'denoise-guidance' or self.mode == 'denoise-grid':
            self.clip_guidance = self.clip_guidance or 0.0
            self.clip_guidance_max = self.clip_guidance_max or 1.0

        return res
    
    def output_path(self) -> Path:
        path_parts = [
            self.mode,
        ]

        if self.clip_text:
            path_parts.append("clip_text_" + "-".join(self.clip_text))
        if self.img_idxs:
            path_parts.append("images_" + "_".join(map(str, self.img_idxs)))

        if self.clip_scale:
            path_parts.append(f"clip_scale_{self.clip_scale:.1f}")
        if self.clip_scale_max:
            path_parts.append(f"clip_scale_max_{self.clip_scale_max:.1f}")
        if self.clip_guidance:
            path_parts.append(f"cfg_{self.clip_guidance:.1f}")
        if self.clip_guidance_max:
            path_parts.append(f"cfg_max_{self.clip_guidance_max:.1f}")
        path_parts.append(f"seed_{self.seed}")
        
        return Path("runs", "make_samples-" + ",".join(path_parts) + ".png")

    def list_experiments(self) -> List[Experiment]:
        return self.experiments
    
    def get_col_labels(self) -> List[str]:
        base_labels = [dn_util.exp_descr(exp) for exp in self.experiments]
        ncols = self.get_ncols()

        labels: List[str] = ["" for _ in range(ncols)]
        for exp_idx, label in enumerate(base_labels):
            labels[exp_idx * self.get_ncol_per_exp()] = label

        return labels
    
    def make_image(self):
        pass

    def get_ncol_per_exp(self) -> int:
        res = 1
        if self.mode == 'denoise-grid':
            res *= self.clip_guidance_count

        if self.img_idxs:
            res *= len(self.img_idxs)
        elif self.clip_text:
            res *= len(self.clip_text)
        return res

    def get_ncols(self) -> int:
        return len(self.experiments) * self.get_ncol_per_exp()

    def get_nrows(self) -> int:
        return self.nrows
    

@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()

    col_labels = cfg.get_col_labels()
    row_labels = []

    print(f"denoise with clip scale {cfg.clip_scale:.1f} to {cfg.clip_scale_max:.1f}")
    print(f"denoise with guidance {cfg.clip_guidance:.1f} to {cfg.clip_guidance_max:.1f}")

    nrows = cfg.get_nrows()
    if cfg.img_idxs:
        nrows = cfg.get_nrows() + 1

    grid = image_util.ImageGrid(ncols=cfg.get_ncols(), nrows=nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)
    if cfg.img_idxs:
        ds = cfg.gen.get_dataset(512)
        col_per = cfg.get_ncol_per_exp()
        for exp_idx in range(len(exps)):
            for exp_col, ds_idx in enumerate(cfg.img_idxs):
                image = image_util.tensor_to_pil(ds[ds_idx][0], image_size=cfg.output_image_size)
                col = exp_idx * col_per + exp_col
                grid.draw_image(col=col, row=nrows - 1, image=image, annotation=str(ds_idx))


    for exp_idx, exp in enumerate(exps):
        exp_run = exp.get_run(loss_type='vloss')
        gen_exp = cfg.gen.for_run(exp, exp_run, deterministic=cfg.deterministic, scale_noise=cfg.scale_noise)

        print()
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")


        for exp_col_idx in range(cfg.get_ncol_per_exp()):
            annotations: List[str] = [""] * cfg.nrows

            torch.manual_seed(cfg.seed)

            denoise = partial(gen_exp.gen_denoise_full, steps=cfg.steps)
            if cfg.steps_ddim is not None:
                denoise = partial(gen_exp.gen_denoise_ddim, steps=cfg.steps_ddim)

            if len(cfg.clip_text):
                for row in range(cfg.nrows):
                    annotations[row] += cfg.clip_text[exp_col_idx % len(cfg.clip_text)]

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

            elif cfg.mode == 'denoise-full':
                latents = gen_exp.get_random_latents(start_idx=0, end_idx=cfg.nrows)
                print(f"embeds {cfg.clip_embeds[exp_col_idx]}")
                images = denoise(latents=list(latents),
                                 clip_guidance=cfg.clip_guidance,
                                 clip_scale=cfg.clip_scale,
                                 clip_embeds=cfg.clip_embeds[exp_col_idx])

            elif cfg.mode == 'denoise-steps':
                if cfg.clip_embeds[exp_col_idx] is not None:
                    latent = gen_exp.get_random_latents(start_idx=0, end_idx=1)[0]
                else:
                    latent = gen_exp.get_random_latents(start_idx=exp_col_idx, end_idx=exp_col_idx+1)[0]
                images = gen_exp.gen_denoise_full(steps=cfg.steps, latents=[latent], yield_count=cfg.nrows,
                                                  clip_embeds=cfg.clip_embeds[exp_col_idx], clip_scale=cfg.clip_scale,
                                                  clip_guidance=cfg.clip_guidance)

            elif cfg.mode == 'denoise-scale':
                latent = gen_exp.get_random_latents(start_idx=exp_col_idx, end_idx=exp_col_idx + 1)[0]
                clip_scale = torch.linspace(start=cfg.clip_scale, end=cfg.clip_scale_max, steps=cfg.nrows).tolist()

                for i, value in enumerate(clip_scale):
                    if annotations[i]:
                        annotations[i] += ", "
                    annotations[i] += f"scale {value:.2f}"

                images = denoise(latents=[latent] * cfg.nrows, 
                                 clip_embeds=cfg.clip_embeds[exp_col_idx],
                                 clip_guidance=cfg.clip_guidance,
                                 clip_scale=clip_scale)

            elif cfg.mode == 'denoise-guidance':
                latent = gen_exp.get_random_latents(start_idx=exp_col_idx, end_idx=exp_col_idx + 1)[0]
                guidance = torch.linspace(start=cfg.clip_guidance, end=cfg.clip_guidance_max, steps=cfg.nrows).tolist()

                for i, value in enumerate(guidance):
                    if annotations[i]:
                        annotations[i] += ", "
                    annotations[i] += f"guidance {value:.2f}"

                images = denoise(latents=[latent] * cfg.nrows,
                                 clip_embeds=cfg.clip_embeds[exp_col_idx],
                                 clip_guidance=guidance,
                                 clip_scale=cfg.clip_scale)

            elif cfg.mode == 'denoise-grid':
                image_no = exp_col_idx // cfg.clip_guidance_count
                latent = gen_exp.get_random_latents(start_idx=image_no, end_idx=image_no + 1)[0]
                guide_weight = (exp_col_idx % cfg.clip_guidance_count) / (cfg.clip_guidance_count - 1)
                guidance = torch.lerp(input=torch.tensor(cfg.clip_guidance), end=torch.tensor(cfg.clip_guidance_max), weight=guide_weight).item()
                clip_scale = torch.linspace(start=cfg.clip_scale, end=cfg.clip_scale_max, steps=cfg.nrows).tolist()

                for i, value in enumerate(clip_scale):
                    if annotations[i]:
                        annotations[i] += ", "
                    annotations[i] += f"scale {value:.2f}, guide {guidance:.2f}"

                guidance = [guidance] * cfg.nrows
                images = denoise(latents=[latent] * cfg.nrows,
                                 clip_embeds=cfg.clip_embeds[image_no], 
                                 clip_scale=clip_scale,
                                 clip_guidance=guidance)

            elif cfg.mode == 'denoise-lerp-clip':
                latent = gen_exp.get_random_latents(start_idx=0, end_idx=1)[0]
                clip_start = cfg.clip_embeds[exp_col_idx]
                clip_end = cfg.clip_embeds[exp_col_idx + 1]

                clip_lerp = gen_exp.interpolate_tensors(clip_start, clip_end, cfg.nrows)
                clip_lerp_list = [embed for embed in clip_lerp]
                images = denoise(latents=[latent] * cfg.nrows,
                                 clip_embeds=clip_lerp_list, 
                                 clip_guidance=cfg.clip_guidance,
                                 clip_scale=cfg.clip_scale)

            elif cfg.mode == 'denoise-image-full':
                latents = gen_exp.get_image_latents(image_idxs=list(range(start_idx, end_idx)), shuffled=True)
                latents = gen_exp.add_noise(latents, timestep=cfg.noise_steps)
                images = denoise(latents=latents)

            elif cfg.mode == 'denoise-image-steps':
                latent = gen_exp.get_image_latents(image_idxs=[exp_col_idx], shuffled=True)[0]
                latent = gen_exp.add_noise([latent], timestep=cfg.noise_steps)[0]
                images = gen_exp.gen_denoise_full(steps=cfg.steps, override_max=cfg.noise_steps,
                                                  latents=[latent], yield_count=cfg.nrows)

            else:
                raise NotImplementedError(f"{cfg.mode} not implemented")

            for row, (image, annotation) in enumerate(zip(images, annotations)):
                col = exp_idx * cfg.get_ncol_per_exp() + exp_col_idx
                grid.draw_image(col=col, row=row, image=image, annotation=annotation)

    out_path = cfg.output_path()
    grid._image.save(out_path)
    print(f"wrote to {out_path}")

if __name__ == "__main__":
    main()
    
