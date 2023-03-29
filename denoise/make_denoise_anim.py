from typing import Tuple
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
from models import vae, unet
from latent_cache import LatentCache
import tqdm


class Config(cmdline.QueryConfig):
    image_size: int
    image_dir: str
    fps: int

    steps_per_denoise: int
    repeat_denoise: int

    def __init__(self):
        super().__init__()
        self.add_argument("-I", "--image_size", type=int, required=True)
        self.add_argument("-d", "--image_dir", default='alex-many-1024')
        self.add_argument("--steps_per_denoise", default=None, type=int, help="override total timesteps")
        self.add_argument("--repeat_denoise", type=int, default=1, help="repeat denoise process N times")
        self.add_argument("--fps", type=int, default=30)

def _load_nets(unet_path: Path, vae_path: Path) -> Tuple[unet.Unet, vae.VarEncDec]:
    try:
        unet_dict = torch.load(unet_path)
        unet = dn_util.load_model(unet_dict)
    except Exception as e:
        print(f"error loading unet from {unet_path}", file=sys.stderr)
        raise e

    try:
        vae_dict = torch.load(vae_path)
        vae = dn_util.load_model(vae_dict)
    except Exception as e:
        print(f"error loading vae from {vae_path}", file=sys.stderr)
        raise e

    return unet, vae

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    sched = noisegen.make_noise_schedule(type='cosine', timesteps=300,
                                         noise_type='normal')

    checkpoints = [(path, exp) for path, exp in cfg.list_checkpoints()
                   if exp.net_class == 'Unet']
    
    dataset, _ = \
        image_util.get_datasets(image_size=cfg.image_size, image_dir=cfg.image_dir,
                                train_split=1.0)

    with torch.no_grad():
        for i, (path, exp) in enumerate(checkpoints):

            animpath = Path("animations", f"anim_{exp.created_at_short}-{exp.shortcode}-{exp.nepochs}.mp4")
            animpath_tmp = str(animpath).replace(".mp4", "-tmp.mp4")

            logline = f"{i + 1}/{len(checkpoints)} {animpath}"
            if animpath.exists():
                print(f"{logline}: skipping; already exists")
                continue

            vae_path = Path(exp.net_vae_path)
            unet, vae = _load_nets(path, vae_path)
            if vae.image_size != cfg.image_size:
                print(f"{logline}: skipping; vae has {vae.image_size=}")
                continue

            print(f"{logline}: generating...")

            latent_dim = vae.latent_dim.copy()
            unet = unet.to(cfg.device)
            vae = vae.to(cfg.device)

            cache = LatentCache(net=vae, net_path=vae_path, batch_size=cfg.batch_size,
                                dataset=dataset, device=cfg.device)
            
            anim_out = \
                cv2.VideoWriter(animpath_tmp,
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                cfg.fps, 
                                (cfg.image_size, cfg.image_size))

            noise = sched.noise_fn([1, *latent_dim]).to(cfg.device)
            next_input = noise

            steps_per_denoise = cfg.steps_per_denoise or sched.timesteps
            step_list = torch.linspace(sched.timesteps - 1, 0, steps_per_denoise * cfg.repeat_denoise).int()

            for step in tqdm.tqdm(step_list):
                next_input = sched.gen_frame(net=unet, inputs=next_input, timestep=step)
                frame_out_t = vae.decode(next_input)
                frame_out = image_util.tensor_to_pil(frame_out_t, cfg.image_size)
                frame_cv = cv2.cvtColor(np.array(frame_out), cv2.COLOR_RGB2BGR)
                anim_out.write(frame_cv)

            anim_out.release()
            Path(animpath_tmp).rename(animpath)
            
