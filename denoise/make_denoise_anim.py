from typing import List, Tuple, Union, Dict
from pathlib import Path
import cv2
import numpy as np
import sys
import tqdm

import torch
from torch import Tensor

sys.path.append("..")
import cmdline
import image_util
import noisegen
import dn_util
from models import vae, unet, denoise, ae_simple, linear
from latent_cache import LatentCache

DenoiseNet = Union[denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]

class ScaledNoiseSchedule(noisegen.NoiseSchedule):
    noise_by_timestep: Dict[int, Tensor] = dict()
    saved_noise: Tensor = None

    def noise(self, size: Tuple, timestep: int = None) -> Tuple[Tensor, Tensor]:
        noise, amount, timestep = super().noise(size, timestep)
        return noise * cfg.noise_mult, amount, timestep

# python make_denoise_anim.py -nc Unet -sc qxseog -I 256 -b 4 -s tloss -f 
#   --steps 150 --walk 1 --direction --nlatents 2 --noise_mult 5e-3
#
# python make_denoise_anim.py -nc Unet -sc qxseog -I 256 -b 4 -s tloss -f 
#   --steps 100 --walk 2 --direction --nlatents 2 --noise_mult 5e-3
class Config(cmdline.QueryConfig):
    image_size: int
    image_dir: str
    steps: int
    fps: int
    ngen: int
    overwrite: bool
    repeat: int

    nlatents: int
    frames_per_pair: int
    walk_mult: float
    walk_in_direction: bool
    noise_mult: float

    def __init__(self):
        super().__init__()
        self.add_argument("-I", "--image_size", type=int, required=True)
        self.add_argument("-d", "--image_dir", default='alex-many-1024')
        self.add_argument("--steps", default=None, type=int, help="denoise steps")
        self.add_argument("--frames_per_pair", "--fpp", default=60, type=int, help="only applicable with --nlatents set. number of frames per noise pair")
        self.add_argument("--fps", type=int, default=30)
        self.add_argument("-f", "--overwrite", default=False, action='store_true')
        self.add_argument("--repeat", default=1, type=int)
        self.add_argument("--ngen", default=None, type=int, help="only generate N animations")

        self.add_argument("-n", "--nlatents", default=1, type=int, 
                          help="instead of denoising --steps on one noise start, denoise the latents between N noise starts")
        self.add_argument("--walk", dest='walk_mult', default=None, type=float, 
                          help="for --nlatents, generate latent noise based on a random walk with this multiplier, instead of pure random for each")
        self.add_argument("--direction", dest='walk_in_direction', default=False, action='store_true',
                          help="for --walk, use a softmaxed random direction, so the walks are going somewhere")
        self.add_argument("--noise_mult", dest='noise_mult', default=1e-3, type=float,
                          help="to prevent unusable flicker when denoising with --nlatents, scale noise used by denoiser")

    def parse_args(self) -> 'Config':
        res = super().parse_args()
        if (self.walk_mult or self.walk_in_direction) and self.nlatents <= 1:
            self.error(f"if --walk or --direction is set, --nlatents must be set >= 2")

def _load_nets(denoise_path: Path, vae_path: Path) -> Tuple[DenoiseNet, vae.VarEncDec]:
    try:
        denoise_dict = torch.load(denoise_path)
        denoise = dn_util.load_model(denoise_dict)
    except Exception as e:
        print(f"error loading from {denoise_path}", file=sys.stderr)
        raise e

    try:
        vae_dict = torch.load(vae_path)
        vae = dn_util.load_model(vae_dict)
    except Exception as e:
        print(f"error loading vae from {vae_path}", file=sys.stderr)
        raise e

    return denoise, vae

def gen_frames(cfg: Config,
               sched: noisegen.NoiseSchedule, latent_dim: List[int],
               dnet: DenoiseNet, vae: vae.VarEncDec,
               anim_out: cv2.VideoWriter):
    if cfg.nlatents >= 2:
        gen_frames_lerp(cfg, sched, latent_dim, dnet, vae, anim_out)
        return
    gen_frames_one(cfg, sched, latent_dim, dnet, vae, anim_out)
    
def gen_frames_one(cfg: Config,
                   sched: noisegen.NoiseSchedule, latent_dim: List[int],
                   dnet: DenoiseNet, vae: vae.VarEncDec,
                   anim_out: cv2.VideoWriter):

    noise, _amount = sched.noise([1, *latent_dim])
    noise = noise.to(cfg.device)
    next_input = noise

    # step_list = torch.linspace(sched.timesteps - 1, 0, steps).int()
    step_list = list(reversed(range(sched.timesteps - cfg.steps, sched.timesteps - 1)))
    for step in tqdm.tqdm(step_list):
        next_input = sched.gen_frame(net=dnet, inputs=next_input, timestep=step)
        frame_out_t = vae.decode(next_input)
        frame_out = image_util.tensor_to_pil(frame_out_t, cfg.image_size)
        frame_cv = cv2.cvtColor(np.array(frame_out), cv2.COLOR_RGB2BGR)
        anim_out.write(frame_cv)

def gen_frames_lerp(cfg: Config,
                    sched: noisegen.NoiseSchedule, latent_dim: List[int],
                    dnet: DenoiseNet, vae: vae.VarEncDec,
                    anim_out: cv2.VideoWriter):
    # generate the noise
    noise_list: List[Tensor] = list()
    if cfg.walk_mult is not None:
        # noise[0] = random noise
        # noise[1] = random walk from noise[0]
        # noise[N] = random walk from noise[N - 1]
        # ...
        if cfg.walk_in_direction:
            # guide the walk in a single direction
            direction = sched.noise_fn(latent_dim).to(cfg.device)
            # direction = torch.softmax(direction, dim=-1)
        else:
            direction = 1.0
        for _ in range(cfg.nlatents):
            noise = sched.noise_fn(latent_dim).to(cfg.device)
            rwalk = noise * cfg.walk_mult * direction
            if len(noise_list):
                new_noise = noise_list[-1] + rwalk
                noise_list.append(new_noise)
            else:
                noise_list.append(noise)
    else:
        # all noise entries are pure random noise.
        for _ in range(cfg.nlatents):
            noise = sched.noise_fn(latent_dim)
            noise_list.append(noise.to(cfg.device))

    # for i in range(len(noise_list) - 1):
    #     ncur, nnext = noise_list[i:i + 2]
    #     diff = nnext - ncur
    #     print(f"{diff.mean()=:.1E} {diff.min()=:.1E} {diff.max()=:.1E} {diff.std()=:.1E}")

    total_frames = cfg.frames_per_pair * (cfg.nlatents - 1)
    noise_lerp = torch.zeros((cfg.batch_size, *latent_dim), device=cfg.device)

    # generate the frames in batches.
    for frame_start in tqdm.tqdm(range(0, total_frames, cfg.batch_size)):
        # python iteration to generate the lerp'ed noise.
        frame_end = min(total_frames, frame_start + cfg.batch_size)
        for frame in range(frame_start, frame_end):
            lat_idx = frame // cfg.frames_per_pair
            start, end = noise_list[lat_idx : lat_idx + 2]
            frame_in_pair = frame % cfg.frames_per_pair

            noise_lerp[frame - frame_start] = \
                torch.lerp(input=start, end=end, weight=frame_in_pair / cfg.frames_per_pair)

        # for i in range(len(noise_lerp) - 1):
        #     lcur, lnext = noise_lerp[i:i + 2]
        #     diff = lnext - lcur
        #     print(f"lerp: {diff.mean()=:.1E} {diff.min()=:.1E} {diff.max()=:.1E} {diff.std()=:.1E}")

        # now generate multiple frames at once. each member of this batch is
        # for a different frame, and is a full denoise.
        denoised_batch = sched.gen(net=dnet, inputs=noise_lerp, steps=cfg.steps)
        img_t_batch = vae.decode(denoised_batch)

        for img_t in img_t_batch:
            img = image_util.tensor_to_pil(img_t, cfg.image_size)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            anim_out.write(img_cv)


if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    if cfg.nlatents > 1:
        # use a scaled scheduler for less flicker during denoising interpolation
        sched_betas = noisegen.make_betas(type='cosine', timesteps=300)
        sched_noise_fn = noisegen.noise_normal
        sched = ScaledNoiseSchedule(betas=sched_betas, timesteps=300, noise_fn=sched_noise_fn)
    else:
        sched = noisegen.make_noise_schedule(type='cosine', timesteps=300, noise_type='normal')

    cfg.steps = cfg.steps or sched.timesteps

    exps = [exp for exp in cfg.list_experiments() if getattr(exp, 'is_denoiser', None)]
    print(f"{len(exps)=}")
    if not len(exps):
        raise Exception("nada")
    
    with torch.no_grad():
        if cfg.ngen:
            exps = exps[:cfg.ngen]

        for i, exp in enumerate(exps):
            best_run = exp.get_run(loss_type='tloss')
            best_path = best_run.checkpoint_path

            path_parts = [
                f"anim_dn_{exp.created_at_short}-{exp.shortcode}",
                f"{exp.net_class}",
                f"nepochs_{best_run.checkpoint_nepochs}",
                f"steps_{cfg.steps}"
            ]
            if cfg.repeat > 1:
                path_parts.append(f"repeats_{cfg.repeat}")
            if cfg.nlatents > 1:
                path_parts.append(f"nlatents_{cfg.nlatents}")
                path_parts.append(f"fpp_{cfg.frames_per_pair}")
                if cfg.walk_mult is not None:
                    path_parts.append(f"walk_{cfg.walk_mult:.1E}")
                if cfg.walk_in_direction:
                    path_parts.append("directional")

            path_base = Path("animations", ",".join(path_parts))
            animpath = Path(str(path_base) + ".mp4")
            animpath_tmp = str(path_base) + "-tmp.mp4"

            logline = f"{i + 1}/{len(exps)} {animpath}"
            if animpath.exists() and not cfg.overwrite:
                print(f"{logline}: skipping; already exists")
                continue

            if not hasattr(exp, 'vae_path'):
                print(f"skip {exp.shortcode}: {exp.nepochs=}: vae_path = None")
                continue

            if best_path is None:
                for run in exp.runs:
                    print(f"  run.cp_path = {run.checkpoint_path}")
                print(f"skip {exp.shortcode}: {exp.nepochs=}: best_path = None")
                continue

            vae_path = Path(exp.vae_path)
            dnet, vae = _load_nets(best_path, vae_path)

            dataset, _ = \
                image_util.get_datasets(image_size=exp.image_size, image_dir=cfg.image_dir,
                                        train_split=1.0)

            print(f"{logline}: generating...")

            latent_dim = vae.latent_dim.copy()
            dnet = dnet.to(cfg.device)
            vae = vae.to(cfg.device)

            cache = LatentCache(net=vae, net_path=vae_path, batch_size=cfg.batch_size,
                                dataset=dataset, device=cfg.device)
            
            anim_out = \
                cv2.VideoWriter(animpath_tmp,
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                cfg.fps, 
                                (cfg.image_size, cfg.image_size))

            for _ in range(cfg.repeat):
                gen_frames(cfg=cfg, sched=sched, latent_dim=latent_dim,
                           dnet=dnet, vae=vae, anim_out=anim_out)

            anim_out.release()
            Path(animpath_tmp).rename(animpath)
            
