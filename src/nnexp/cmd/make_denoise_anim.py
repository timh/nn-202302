from typing import List, Tuple, Union, Dict, Callable, Generator
from pathlib import Path
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch

from nnexp.utils import cmdline
from nnexp.experiment import Experiment, ExpRun
from nnexp.images import image_util
from nnexp.denoise import dn_util, imagegen
from nnexp.denoise.clip_cache import ClipModelName

# python make_denoise_anim.py -nc Unet -sc qxseog -I 256 -b 4 -s tloss -f 
#   --steps 100 -n 90 --direction --noise_mult 5e-3 --walk_mult 0.1
#
# python make_denoise_anim.py -nc Unet -a 'ago < 2h' -I 256 -b 4 -s tloss
class Config(cmdline.QueryConfig):
    image_size: int
    image_dir: str
    steps: int
    fps: int
    ngen: int
    overwrite: bool
    repeat: int

    lerp_nlatents: int
    frames_per_pair: int

    walk_frames: int
    walk_mult: float
    walk_in_direction: bool

    deterministic: bool
    scale_noise: float

    lerp_clip: bool
    clip_model_name: ClipModelName = None
    clip_rand_count: int
    clip_image_idx: List[int]
    clip_text: List[str]

    clip_scale: float
    clip_scale_max: float

    def __init__(self):
        super().__init__()
        self.add_argument("-I", "--image_size", type=int, required=True)
        self.add_argument("-d", "--image_dir", default="images.2018-2020")
        self.add_argument("--steps", default=300, type=int, help="denoise steps")
        self.add_argument("--fps", type=int, default=30)
        self.add_argument("-f", "--overwrite", default=False, action='store_true')
        self.add_argument("--repeat", default=1, type=int)
        self.add_argument("--ngen", default=None, type=int, help="only generate N animations")

        self.add_argument("-N", "--lerp_nlatents", default=None, type=int, 
                          help="instead of denoising --steps on one noise start, denoise the latents between N noise starts")
        self.add_argument("--frames_per_pair", "--fpp", default=60, type=int, 
                          help="only applicable with --lerp_nlatents. number of frames per noise pair")

        self.add_argument("-n", "--walk_frames", default=None, type=int,
                          help="instead of denoising on one picture, or with random noise keyframes (--lerp_nlatents), walk randomly from a starting latent and denoise this many frames")
        self.add_argument("--walk_mult", default=1.0, type=float, 
                          help="with --walk_frames, use this multiplier against the random walk")
        self.add_argument("--direction", dest='walk_in_direction', default=False, action='store_true',
                          help="for --walk, use a fixed random direction, so the walks are going somewhere")

        self.add_argument("-D", "--deterministic", default=False, action='store_true', 
                          help="use the same random values for noise generation across batches")
        self.add_argument("--scale_noise", default=1.0, type=float,
                          help="to prevent unusable flicker when denoising, scale noise used by denoiser")

        self.add_argument("--lerp_clip", default=False, action='store_true')
        self.add_argument("--clip_rand_count", default=0, type=int,
                          help="use clip embedding from N random images to drive the denoise. sets --repeat")
        self.add_argument("--clip_image", dest='clip_image_idx', default=list(), type=int, nargs='+',
                          help="use clip embedding from the given image indexes to drive denoise. sets --repeat")
        self.add_argument("--clip_text", default=list(), type=str, nargs='+',
                          help="use clip embedding from given text strings to drive denoise. sets --repeat")
        self.add_argument("--clip_scale", default=None, type=float)
        self.add_argument("--clip_scale_max", default=None, type=float)
        self.add_argument("--clip_model_name", default=None, type=str)

    def parse_args(self) -> 'Config':
        res = super().parse_args()
        if self.lerp_nlatents is not None and self.lerp_nlatents < 2:
            self.error(f"--lerp_nlatents must be >= 2")
        
        if self.clip_rand_count:
            self.repeat = self.clip_rand_count
        elif len(self.clip_image_idx):
            self.repeat = len(self.clip_image_idx)
        elif len(self.clip_text):
            self.repeat = len(self.clip_text)

        if any([self.clip_rand_count, len(self.clip_image_idx), len(self.clip_text)]) and self.clip_model_name is None:
            self.clip_model_name = "RN50"
    
    def make_path(self, exp: Experiment, run: ExpRun, repeat_idx: int, clip_anno: str = None)-> Path:
        path_parts = [
            f"anim_dn_{exp.shortcode}",
            f"{exp.net_class}",
            f"nepochs_{run.checkpoint_nepochs}",
            f"loss_{exp.loss_type}",
            f"steps_{self.steps}",
        ]

        if self.lerp_nlatents:
            path_parts.append(f"nlatents_{self.lerp_nlatents}")
            path_parts.append(f"fpp_{self.frames_per_pair}")
            path_parts.append(f"noise_scale_{self.scale_noise:.1E}")

        if self.walk_frames:
            path_parts.append(f"walk_{self.walk_frames}")
            path_parts.append(f"mult_{self.walk_mult:.1E}")
            path_parts.append(f"noise_scale_{self.scale_noise:.1E}")
            if self.walk_in_direction:
                path_parts.append("directional")
        
        if self.clip_scale:
            path_parts.append(f"clip_scale_{self.clip_scale:.1f}")
        
        path_base = ",".join(path_parts)
        if self.repeat > 1:
            path_base = str(path_base) + f"--{repeat_idx:02}"
        
        if clip_anno is not None:
            path_base += f"-{clip_anno}"
        
        return Path("animations", path_base + ".mp4")

def gen_frames(cfg: Config, gen_exp: imagegen.ImageGenExp, clip_image: Image.Image, clip_text: str) -> Generator[Image.Image, None, None]:
    if not any([cfg.lerp_clip, cfg.lerp_nlatents, cfg.walk_frames, cfg.clip_scale_max]):
        noise = list(gen_exp.get_random_latents(start_idx=0, end_idx=1))[0]
        yield from gen_exp.gen_denoise_full(steps=cfg.steps, yield_count=cfg.steps, latents=[noise], 
                                            clip_text=clip_text, clip_images=clip_image, clip_scale=cfg.clip_scale)
        return

    if cfg.lerp_nlatents:
        noise_list = list(gen_exp.get_random_latents(start_idx=0, end_idx=cfg.lerp_nlatents))
        for i in range(cfg.lerp_nlatents - 1):
            start, end = noise_list[i : i + 2]
            print(f"lerp latents: {i=}")
            lerp_steps = list(gen_exp.interpolate_tensors(start=start, end=end, steps=cfg.frames_per_pair))
            print(f"{len(lerp_steps)=}")
            yield from gen_exp.gen_denoise_ddim(steps=cfg.steps, latents=lerp_steps)
        return

    if cfg.clip_scale_max:
        noise = list(gen_exp.get_random_latents(start_idx=0, end_idx=1))[0]
        start = torch.tensor(cfg.clip_scale)
        end = torch.tensor(cfg.clip_scale_max)
        lerp_steps = list(gen_exp.interpolate_tensors(start, end, steps=cfg.frames_per_pair))
        yield from gen_exp.gen_denoise_ddim(steps=cfg.steps, latents=[noise] * cfg.frames_per_pair,
                                            clip_embeds=(clip_text or clip_image), 
                                            clip_scale=lerp_steps)
        return

    # elif cfg.walk_frames:
    # noise[0] = random noise
    # noise[1] = random walk from noise[0]
    # noise[N] = random walk from noise[N - 1]
    # ...
    noise = list(gen_exp.get_random_latents(start_idx=0, end_idx=cfg.walk_frames))[0]
    if cfg.walk_in_direction:
        # guide the walk in a single direction, using [-1..1] uniform noise
        direction = (torch.rand_like(noise) * 2 - 1).to(cfg.device)
    else:
        direction = 1.0

    raise NotImplemented()


@torch.no_grad()
def main():
    cfg = Config()
    cfg.parse_args()

    exps = [exp for exp in cfg.list_experiments() if getattr(exp, 'is_denoiser', None)]
    if cfg.ngen:
        exps = exps[:cfg.ngen]
    if not len(exps):
        cfg.error("no experiments")

    print(f"{len(exps)} experiments. {cfg.steps} denoising steps.")
    
    font = ImageFont.truetype(Roboto, 10)
    gen = imagegen.ImageGen(image_dir=cfg.image_dir, output_image_size=cfg.image_size,
                            device=cfg.device, batch_size=cfg.batch_size,
                            clip_model_name=cfg.clip_model_name)
    
    clip_images: List[Image.Image] = [None] * cfg.repeat
    clip_text: List[str] = [None] * cfg.repeat
    clip_anno: List[str] = [None] * cfg.repeat

    if cfg.clip_rand_count:
        ds = gen.get_dataset(512)
        nimages = len(ds)
        for i in range(cfg.clip_rand_count):
            rand_idx = torch.randint(low=0, high=nimages, size=(1,)).item()
            image_t, _ = ds[rand_idx]
            clip_images[i] = image_util.tensor_to_pil(image_t)
            clip_anno[i] = str(rand_idx)
    
    if len(cfg.clip_image_idx):
        print(f"clip image idx")
        ds = gen.get_dataset(512)
        for i, idx in enumerate(cfg.clip_image_idx):
            image_t, _ = ds[idx]
            clip_images[i] = image_util.tensor_to_pil(image_t)
            clip_anno[i] = str(idx)

    elif cfg.clip_text is not None:
        for i, text in enumerate(cfg.clip_text):
            clip_text[i] = text
            clip_anno[i] = text
            print(f"clip_anno[{i}] = {text}")

    for i, exp in enumerate(exps):
        best_run = exp.get_run(loss_type='tloss')
        gen_exp = gen.for_run(exp=exp, run=best_run, deterministic=cfg.deterministic, scale_noise=cfg.scale_noise)
        exp_descr = dn_util.exp_descr(exp, include_label=False)

        for repeat_idx in range(cfg.repeat):
            anim_path = cfg.make_path(exp=exp, run=best_run, repeat_idx=repeat_idx, clip_anno=clip_anno[repeat_idx])
            temp_path = str(anim_path).replace(".mp4", "-temp.mp4")

            logline = f"{i + 1}/{len(exps)} {anim_path}"
            if anim_path.exists() and not cfg.overwrite:
                print(f"{logline}: skipping; already exists")
                continue

            print(f"{logline}: generating...")

            title, title_height = image_util.fit_strings(exp_descr, max_width=cfg.image_size, font=font)
            height = cfg.image_size + title_height
            width = cfg.image_size

            image = Image.new("RGB", (width, height))
            draw = ImageDraw.ImageDraw(image)

            anim_out = \
                cv2.VideoWriter(temp_path,
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                cfg.fps, 
                                (width, height))

            for frame_idx, frame_img in enumerate(gen_frames(cfg=cfg, gen_exp=gen_exp, clip_image=clip_images[repeat_idx], clip_text=clip_text[repeat_idx])):
                draw.rectangle((0, 0, width, title_height), fill='black')
                draw.text((0, 0), text=title, font=font, fill='white')
                image.paste(frame_img, box=(0, title_height))

                if clip_anno[repeat_idx] is not None:
                    image_util.annotate(image=image, draw=draw, font=font, text=clip_anno[repeat_idx], upper_left=(0, 0), within_size=width,
                                        ref='upper_left')
                if cfg.clip_scale_max:
                    # HACK
                    cur_clip = torch.lerp(torch.tensor(cfg.clip_scale), torch.tensor(cfg.clip_scale_max), frame_idx / cfg.frames_per_pair)
                    image_util.annotate(image=image, draw=draw, font=font, text=f"{cur_clip:.2f}", upper_left=(0, 0), 
                                        within_size=width, ref='lower_left')

                frame_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                anim_out.write(frame_cv)

            anim_out.release()
            Path(temp_path).rename(anim_path)
            
if __name__ == "__main__":
    main()