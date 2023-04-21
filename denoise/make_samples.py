import sys
from pathlib import Path
from typing import List, Literal
from PIL import Image

import torch

sys.path.append("..")
from experiment import Experiment
import image_util
import dn_util
import cmdline
import imagegen

MODES = ("random interp roundtrip "
         "denoise-random-full denoise-random-steps denoise-random-scale "
         "denoise-image-full denoise-image-steps").split()
Mode = Literal["random", "interp", "roundtrip", 
               "denoise-random-full", "denoise-random-steps", "denoise-random-scale",
               "denoise-image-full", "denoise-image-steps"]

HELP = """
               random: start with random inputs.
               interp: interpolate between two random images. if using a denoiser, it will do denoising subtraction.
            roundtrip: take N images, and run them through the whole network(s)

  denoise-random-full: denoise random latents with --steps steps
 denoise-random-steps: denoise random latents with steps range(1, --steps - 1, --steps // nrows)
 denoise-random-scale: denoise random latents with full steps, scale from --clip_scale to --clip_scale_max
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
    nrepeats: int
    denoise_steps: int
    noise_steps: int

    steps: int
    nrows: int

    deterministic: bool
    scale_noise: float

    img_idxs: List[int]

    clip_nimages: int
    clip_text: List[str]
    clip_model_name: str = "RN50"
    clip_scale: float
    clip_scale_max: float

    experiments: List[Experiment] = None

    def __init__(self):
        super().__init__()
        self.add_argument("-m", "--mode", default="interp", choices=MODES)
        self.add_argument("-o", "--output", default=None)
        self.add_argument("--nrows", "--row", default=10, type=int, help="number of rows")
        self.add_argument("-d", "--image_dir", default="images.alex-1024")
        self.add_argument("-i", "--output_image_size", type=int, default=None)
        self.add_argument("--repeat", dest='nrepeats', type=int, default=1)
        self.add_argument("-n", "--steps", dest='steps', type=int, default=300, 
                          help="denoising steps")
        self.add_argument("-N", "--noise_steps", dest='noise_steps', type=int, default=100, 
                          help="steps of noise")

        self.add_argument("-D", "--deterministic", default=False, action='store_true', 
                          help="use the same random values for noise generation across batches")
        self.add_argument("--scale_noise", default=1.0, type=float,
                          help="to prevent unusable flicker when denoising, scale noise used by denoiser")

        self.add_argument("--clip_text", default=list(), type=str, nargs='+',
                          help="use clip embedding from given text strings to drive denoise. sets --repeat")
        self.add_argument("--clip_nimages", default=None, type=int)
        self.add_argument("--images", dest='img_idxs', default=list(), nargs='+', type=int)
        self.add_argument("--clip_scale", default=1.0, type=float,
                          help="multiplier used for cross attention with CLIP; minimum when used with -m denoise-random-scale")
        self.add_argument("--clip_scale_max", default=10.0, type=float,
                          help="multiplier used for cross attention with CLIP; maximum when used with -m denoise-random-scale")
    
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

        if len(self.clip_text):
            self.nrepeats = len(self.clip_text)
        
        if self.clip_nimages:
            self.nrepeats = self.clip_nimages
        if self.img_idxs:
            self.nrepeats = len(self.img_idxs)

        return res
    
    def output_path(self) -> Path:
        path_parts = [
            self.mode,
        ]

        if self.clip_text:
            path_parts.append("clip_text_" + "-".join(self.clip_text))
        if self.clip_text or self.clip_nimages or self.mode == 'denoise-random-scale':
            path_parts.append(f"clip_scale_{self.clip_scale:.1f}")
        if self.mode == 'denoise-random-scale':
            path_parts.append(f"clip_scale_max_{self.clip_scale_max:.1f}")
        if self.clip_nimages:
            path_parts.append(f"clip_nimages_{self.clip_nimages}")
        if self.img_idxs:
            path_parts.append("images_" + "_".join(map(str, self.img_idxs)))
        
        return Path("make_samples-" + ",".join(path_parts) + ".png")

    def list_experiments(self) -> List[Experiment]:
        return self.experiments
    
    def get_col_labels(self) -> List[str]:
        return [dn_util.exp_descr(exp) for exp in self.experiments]
    
@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

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

    if cfg.mode == 'denoise-random-scale':
        print(f"denoise with clip scale {cfg.clip_scale:.1f} to {cfg.clip_scale_max:.1f}")

    nrows = cfg.nrows
    if cfg.clip_nimages or cfg.img_idxs:
        nrows += 1
    grid = image_util.ImageGrid(ncols=ncols, nrows=nrows, 
                                image_size=cfg.output_image_size,
                                col_labels=col_labels,
                                row_labels=row_labels)
    gen = imagegen.ImageGen(image_dir=cfg.image_dir, 
                            output_image_size=cfg.output_image_size, 
                            clip_model_name=cfg.clip_model_name,
                            device=cfg.device, batch_size=cfg.batch_size)

    clip_images: List[Image.Image] = [None] * cfg.nrepeats
    if cfg.clip_nimages or cfg.img_idxs:
        ds = gen.get_dataset(512)
        if cfg.img_idxs:
            ds_idxs = cfg.img_idxs
        else:
            ds_idxs = torch.randint(low=0, high=len(ds), size=(cfg.clip_nimages,)).tolist()

        clip_images = list()
        for repeat_idx in range(cfg.nrepeats):
            ds_idx = ds_idxs[repeat_idx]
            image_t = ds[ds_idx][0]
            image = image_util.tensor_to_pil(image_t)
            clip_images.append(image)
            for exp_idx in range(len(exps)):
                col = exp_idx * cfg.nrepeats + repeat_idx
                grid.draw_image(col=col, row=cfg.nrows, image=image, annotation=str(ds_idx))

    clip_text: List[str] = [None] * cfg.nrepeats
    if cfg.clip_text:
        clip_text = cfg.clip_text
    
    for exp_idx, exp in enumerate(exps):
        gen_exp = gen.for_run(exp, exp.get_run(), deterministic=cfg.deterministic, scale_noise=cfg.scale_noise)
        column = exp_idx * cfg.nrepeats

        print()
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp.net_class}: {exp.label}")

        for repeat_idx in range(cfg.nrepeats):
            annotations: List[str] = [""] * cfg.nrows
            start_idx = repeat_idx * cfg.nrows
            end_idx = start_idx + cfg.nrows
            image_idxs = list(range(start_idx, end_idx))

            torch.manual_seed(0)

            if len(cfg.clip_text):
                annotations.clear()
                annotations.extend([cfg.clip_text[repeat_idx]] * cfg.nrows)

            if cfg.nrepeats > 1:
                print(f"  repeat {repeat_idx + 1}/{cfg.nrepeats}")

            if cfg.mode == 'random':
                images = gen_exp.gen_random(start_idx=start_idx, end_idx=end_idx)

            elif cfg.mode == 'interp':
                start_idx = repeat_idx * 2
                end_idx = start_idx + 1
                start, end = gen_exp.get_image_latents(image_idxs=[start_idx, end_idx], shuffled=True)
                images = gen_exp.gen_lerp(start=start, end=end, steps=cfg.nrows)

            elif cfg.mode == 'roundtrip':
                images = gen_exp.gen_roundtrip(image_idxs=image_idxs, shuffled=True)

            elif cfg.mode == 'denoise-random-full':
                latents = gen_exp.get_random_latents(start_idx=0, end_idx=cfg.nrows)
                images = gen_exp.gen_denoise_full(steps=cfg.steps, latents=list(latents),
                                                  clip_text=clip_text[repeat_idx], clip_images=clip_images[repeat_idx],
                                                  clip_scale=cfg.clip_scale)

            elif cfg.mode == 'denoise-random-steps':
                if clip_text or clip_images:
                    latent = gen_exp.get_random_latents(start_idx=0, end_idx=1)[0]
                else:
                    latent = gen_exp.get_random_latents(start_idx=repeat_idx, end_idx=repeat_idx+1)[0]
                images = gen_exp.gen_denoise_full(steps=cfg.steps, latents=[latent], yield_count=cfg.nrows,
                                                  clip_text=clip_text[repeat_idx], clip_images=clip_images[repeat_idx],
                                                  clip_scale=cfg.clip_scale)

            elif cfg.mode == 'denoise-random-scale':
                clip_scale = torch.linspace(start=cfg.clip_scale, end=cfg.clip_scale_max, steps=cfg.nrows).tolist()
                latent = gen_exp.get_random_latents(start_idx=0, end_idx=1)[0]

                for i, one_scale in enumerate(clip_scale):
                    if annotations[i]:
                        annotations[i] += ", "
                    annotations[i] += f"scale {one_scale:.1f}"

                images = gen_exp.gen_denoise_full(steps=cfg.steps, latents=[latent] * cfg.nrows, 
                                                  clip_text=clip_text[repeat_idx], clip_images=clip_images[repeat_idx],
                                                  clip_scale=clip_scale)

            elif cfg.mode == 'denoise-image-full':
                latents = gen_exp.get_image_latents(image_idxs=list(range(start_idx, end_idx)), shuffled=True)
                latents = gen_exp.add_noise(latents, timestep=cfg.noise_steps)
                images = gen_exp.gen_denoise_full(steps=cfg.steps, latents=latents)

            elif cfg.mode == 'denoise-image-steps':
                latent = gen_exp.get_image_latents(image_idxs=[repeat_idx], shuffled=True)[0]
                latent = gen_exp.add_noise([latent], timestep=cfg.noise_steps)[0]
                images = gen_exp.gen_denoise_full(steps=cfg.steps, override_max=cfg.noise_steps,
                                                  latents=[latent], yield_count=cfg.nrows)

            else:
                raise NotImplementedError(f"{cfg.mode} not implemented")

            for row, (image, annotation) in enumerate(zip(images, annotations)):
                grid.draw_image(col=column + repeat_idx, row=row, image=image, annotation=annotation)

    out_path = cfg.output_path()
    grid._image.save(out_path)
    print(f"wrote to {out_path}")

if __name__ == "__main__":
    main()
    
