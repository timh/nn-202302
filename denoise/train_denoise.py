# %%
import sys
from typing import List, Dict, Set, Tuple, Callable
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
import dn_util
import cmdline_image
import checkpoint_util
import re

import loggers.image_progress as img_prog
import denoise_progress as dn_prog
import loggers.chain as chain_logger

import dataloader
from models import denoise, vae
import noisegen
import image_util

# python train_denoise.py -c conf/dn_denoise.py -vsc azutfw -n 50 -b 256 
#   --startlr 1.0e-3 --endlr 1.0e-4 --use_best tloss 
#   -d images.alex+1star-1024 -I 512 --no_compile --sched_warmup_epochs 5
class Config(cmdline_image.ImageTrainerConfig):
    truth_is_noise: bool
    attribute_matches: List[str]
    pattern: re.Pattern
    enc_batch_size: int
    gen_steps: List[int]
    resume_shortcodes: List[str]
    vae_shortcode: str

    noise_fn_str: str
    noise_steps: int
    noise_beta_type: str
    noise_schedule: noisegen.NoiseSchedule = None

    clip_emblen: int
    clip_model_name: str
    unconditional_ratio: float

    checkpoints: List[Tuple[Path, Experiment]]

    def __init__(self):
        super().__init__("denoise")
        self.add_argument("--truth", choices=["noise", "src"], default="noise")
        self.add_argument("--noise_fn", dest='noise_fn_str', default='normal', choices=['rand', 'normal'])
        self.add_argument("--noise_steps", type=int, default=300)
        self.add_argument("--noise_beta_type", type=str, default='cosine')
        self.add_argument("--gen_steps", type=int, nargs='+', default=None)
        self.add_argument("-B", "--enc_batch_size", type=int, default=4)
        self.add_argument("--resume_shortcodes", type=str, nargs='+', default=[], help="resume only these shortcodes")
        self.add_argument("-vsc", "--vae_shortcode", type=str, help="vae shortcode", required=True)
        self.add_argument("--clip_model_name", type=str, default="RN50")
        self.add_argument("--ratio", dest='unconditional_ratio', type=float, default=None)

    def parse_args(self) -> 'Config':
        super().parse_args()

        self.truth_is_noise = (self.truth == "noise")
        self.noise_schedule = \
            noisegen.make_noise_schedule(type=self.noise_beta_type,
                                         timesteps=self.noise_steps,
                                         noise_type=self.noise_fn_str)

        vae_exp = checkpoint_util.find_experiment(self.vae_shortcode)
        if vae_exp is None or vae_exp.net_class != 'VarEncDec':
            raise Exception(f"whoops, can't find VAE with shortcode {self.vae_shortcode}")

        vae_path = vae_exp.get_run().checkpoint_path
        vae_net = dn_util.load_model(vae_path).to(self.device)
        vae_net.requires_grad_(False)
        vae_net.eval()
        vae_exp.net = vae_net

        print(f"""{vae_path}:
  last_train_loss: {vae_exp.last_train_loss:.3f}
    last_val_loss: {vae_exp.last_val_loss:.3f}
          nepochs: {vae_exp.nepochs}
         saved_at: {vae_exp.saved_at}
         relative: {vae_exp.saved_at_relative()}
        shortcode: {vae_exp.shortcode}
          nparams: {vae_exp.nparams() / 1e6:.3f}M""")
    
        self.vae_net = vae_net
        self.vae_path = vae_path
        self.vae_exp = vae_exp


        return self
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset = super().get_dataset()

        clip_model_name = "RN50"

        eds_item_type: dataloader.EDSItemType = 'sample'
        enc_dataset = dataloader.EncoderDataset(
            item_type=eds_item_type, dataset=dataset, image_dir=self.image_dir,
            vae_net=self.vae_net, vae_net_path=self.vae_path,
            device=self.device, batch_size=self.enc_batch_size,
            clip_model_name=clip_model_name
        )
        noise_ds = dataloader.NoisedDataset(base_dataset=enc_dataset, noise_schedule=self.noise_schedule, unconditional_ratio=cfg.unconditional_ratio)
        train_ds, val_ds = dataloader.split_dataset(noise_ds)

        train_dl = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=True)

        # HACK. 
        self.clip_cache = enc_dataset._clip_cache
        self.clip_emblen = self.clip_cache.get_clip_emblen()
        self.train_lat_cache = enc_dataset.cache

        return train_dl, val_dl
    
    def get_loggers(self, 
                    exps: List[Experiment]) -> chain_logger.ChainLogger:
        logger = super().get_loggers()

        dn_gen = dn_prog.DenoiseProgress(truth_is_noise=self.truth_is_noise,
                                         noise_schedule=self.noise_schedule,
                                         device=self.device,
                                         gen_steps=self.gen_steps,
                                         decoder_fn=self.vae_net.decode,
                                         latent_dim=self.vae_net.latent_dim)
        img_logger = \
            img_prog.ImageProgressLogger(basename=self.basename,
                                         progress_every_nepochs=self.progress_every_nepochs,
                                         generator=dn_gen,
                                         image_size=self.image_size,
                                         exps=exps)
        logger.loggers.append(img_logger)
        return logger

    def build_experiments(self, exps: List[Experiment],
                          train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
        exps = super().build_experiments(exps, train_dl, val_dl)
        if self.resume_shortcodes:
            exps = [exp for exp in exps if exp.shortcode in self.resume_shortcodes]
        return exps

    def get_loss_fn(self, exp: Experiment):
        backing_loss_fn = train_util.get_loss_fn(exp.loss_type, device=self.device)
        twotruth_loss_fn = \
            train_util.twotruth_loss_fn(backing_loss_fn=backing_loss_fn,
                                        truth_is_noise=self.truth_is_noise, 
                                        device=self.device)
        
        return twotruth_loss_fn


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = Config()
    cfg.parse_args()

    vae_net = cfg.vae_net
    vae_path = cfg.vae_path

    # set up noising dataloaders that use vae_net as the decoder. force the image_size
    # to be what the vae was trained with.
    cfg.image_size = vae_net.image_size
    train_dl, val_dl = cfg.get_dataloaders()

    exps: List[Experiment] = list()
    vae_latent_dim = vae_net.latent_dim.copy()

    # load config file
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())

    def lazy_net_denoise(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(_exp: Experiment) -> nn.Module:
            return denoise.DenoiseModel(**kwargs)
        return fn
            
    def lazy_net_vae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(_exp: Experiment) -> nn.Module:
            return vae.VAEDenoise(**kwargs)
        return fn
        
    for exp in exps:
        exp.noise_steps = cfg.noise_steps
        exp.noise_beta_type = cfg.noise_beta_type
        exp.truth_is_noise = cfg.truth_is_noise
        exp.vae_path = str(vae_path)
        exp.vae_shortcode = cfg.vae_exp.shortcode
        exp.image_size = vae_net.image_size
        exp.is_denoiser = True
        if cfg.unconditional_ratio:
            exp.unconditional_ratio = cfg.unconditional_ratio

        exp.train_dataloader = train_dl
        exp.val_dataloader = val_dl
        exp.loss_fn = cfg.get_loss_fn(exp)

        label_parts = [
            # f"noise_{cfg.noise_beta_type}_{cfg.noise_steps}",
            "img_latdim_" + "_".join(map(str, vae_latent_dim)),
            # f"noisefn_{cfg.noise_fn_str}",
        ]

        # BUG: this doesn't work until the net is started.
        dn_latent_dim = getattr(exp, "net_latent_dim", None)
        if dn_latent_dim is not None:
            label_parts.append("dn_latdim_" + "_".join(map(str, dn_latent_dim)))
        label_parts.append(f"loss_{exp.loss_type}")
        if cfg.unconditional_ratio:
            label_parts.append(f"uncond_{cfg.unconditional_ratio:.1f}")

        if len(exp.label):
            exp.label += ","
        exp.label += ",".join(label_parts)


    exps = cfg.build_experiments(exps, train_dl=train_dl, val_dl=val_dl)
    logger = cfg.get_loggers(exps)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=60, val_limit_frequency=30)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
