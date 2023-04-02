from typing import Tuple, Union
from pathlib import Path
import torch
import cv2
import numpy as np
import sys

sys.path.append("..")
import cmdline
import image_util
import noisegen
import dn_util
from models import vae, unet, denoise, ae_simple, linear
from latent_cache import LatentCache
import tqdm

DenoiseNet = Union[denoise.DenoiseModel, unet.Unet, ae_simple.AEDenoise, linear.DenoiseLinear]

class Config(cmdline.QueryConfig):
    image_size: int
    image_dir: str
    steps: int
    fps: int

    nlatents: int
    frames_per_pair: int
    mult: float

    ngen: int

    def __init__(self):
        super().__init__()
        self.add_argument("-I", "--image_size", type=int, required=True)
        self.add_argument("-d", "--image_dir", default='alex-many-1024')
        self.add_argument("--steps", default=None, type=int, help="denoise steps")
        self.add_argument("--frames_per_pair", "--fpp", default=60, type=int, help="only applicable with --nlatents set. number of frames per noise pair")
        self.add_argument("--fps", type=int, default=30)

        self.add_argument("-n", "--nlatents", default=1, type=int, help="instead of denoising --steps on one noise start, denoise the latents between N noise starts")
        self.add_argument("--mult", default=None, type=float, help="amount to adjust noise by when walking from original latent")
        self.add_argument("--ngen", default=None, type=int, help="only generate N animations")

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

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    sched = noisegen.make_noise_schedule(type='cosine', timesteps=300, noise_type='normal')
    steps = cfg.steps or sched.timesteps

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
                f"anim_{exp.created_at_short}-{exp.shortcode}",
                f"{exp.net_class}",
                f"nepochs_{best_run.checkpoint_nepochs}",
                f"steps_{steps}"
            ]
            if cfg.nlatents > 1:
                path_parts.append(f"nlatents_{cfg.nlatents}")
                path_parts.append(f"fpp_{cfg.frames_per_pair}")
                if cfg.mult:
                    path_parts.append(f"mult_{cfg.mult:.2f}")

            path_base = Path("animations", ",".join(path_parts))
            animpath = Path(str(path_base) + ".mp4")
            animpath_tmp = str(path_base) + "-tmp.mp4"

            logline = f"{i + 1}/{len(exps)} {animpath}"
            if animpath.exists():
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

            if cfg.nlatents < 2:
                noise, _amount = sched.noise([1, *latent_dim])
                noise = noise.to(cfg.device)
                next_input = noise

                step_list = torch.linspace(sched.timesteps - 1, 0, steps).int()
                for step in tqdm.tqdm(step_list):
                    next_input = sched.gen_frame(net=dnet, inputs=next_input, timestep=step)
                    frame_out_t = vae.decode(next_input)
                    frame_out = image_util.tensor_to_pil(frame_out_t, cfg.image_size)
                    frame_cv = cv2.cvtColor(np.array(frame_out), cv2.COLOR_RGB2BGR)
                    anim_out.write(frame_cv)
            else:
                noise_list = list()
                if cfg.mult:
                    while len(noise_list) < cfg.nlatents:
                        noise, _amount = sched.noise(latent_dim)
                        noise = noise.to(cfg.device)
                        rwalk = (noise * cfg.mult)
                        if len(noise_list):
                            noise_list.append(noise_list[-1][0] + rwalk)
                        else:
                            noise_list.append(noise)
                else:
                    for _ in range(cfg.nlatents):
                        noise, _amount = sched.noise(latent_dim)
                        noise_list.append(noise.to(cfg.device))

                total_frames = cfg.frames_per_pair * (cfg.nlatents - 1)
                noise_lerp = torch.zeros((cfg.batch_size, *latent_dim), device=cfg.device)

                for frame_start in tqdm.tqdm(range(0, total_frames, cfg.batch_size)):
                    frame_end = min(total_frames, frame_start + cfg.batch_size)
                    for frame in range(frame_start, frame_end):
                        lat_idx = frame // cfg.frames_per_pair
                        start, end = noise_list[lat_idx : lat_idx + 2]
                        frame_in_pair = frame % cfg.frames_per_pair

                        noise_lerp[frame - frame_start] = torch.lerp(input=start, end=end, weight=frame_in_pair / cfg.frames_per_pair)

                    denoised_batch = sched.gen(net=dnet, inputs=noise_lerp, steps=steps, truth_is_noise=True)
                    img_t_batch = vae.decode(denoised_batch)

                    for img_t in img_t_batch:
                        img = image_util.tensor_to_pil(img_t, cfg.image_size)
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        anim_out.write(img_cv)

            anim_out.release()
            Path(animpath_tmp).rename(animpath)
            
