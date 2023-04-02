# %%
import sys
import argparse
import datetime
from typing import List, Literal, Tuple, Dict, Callable
from pathlib import Path
import itertools

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
import conv_types
import dn_util
import cmdline_image
import checkpoint_util
import re

import loggers.image_progress as img_prog
import denoise_progress as dn_prog
import loggers.chain as chain_logger

import dataloader
from models import denoise, vae, unet
import noisegen

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

    checkpoints: List[Tuple[Path, Experiment]]

    def __init__(self):
        super().__init__("denoise")
        self.add_argument("--truth", choices=["noise", "src"], default="noise")
        self.add_argument("--noise_fn", dest='noise_fn_str', default='normal', choices=['rand', 'normal'])
        self.add_argument("--noise_steps", type=int, default=300)
        self.add_argument("--noise_beta_type", type=str, default='cosine')
        self.add_argument("--gen_steps", type=int, nargs='+', default=None)
        self.add_argument("-B", "--enc_batch_size", type=int, default=4)
        self.add_argument("--shortcodes", dest='resume_shortcodes', type=str, nargs='+', default=[], help="resume only these shortcodes")
        self.add_argument("--vae_shortcode", type=str, help="vae shortcode", required=True)

    def parse_args(self) -> 'Config':
        super().parse_args()

        self.truth_is_noise = (self.truth == "noise")
        self.noise_schedule = \
            noisegen.make_noise_schedule(type=self.noise_beta_type,
                                         timesteps=self.noise_steps,
                                         noise_type=self.noise_fn_str)
        return self
    
    def get_dataloaders(self, vae_net: vae.VarEncDec, vae_net_path: Path) -> Tuple[DataLoader, DataLoader]:
        src_train_ds, src_val_ds = super().get_datasets()

        eds_item_type: dataloader.EDSItemType = 'sample'

        dl_args = dict(vae_net=vae_net, vae_net_path=vae_net_path,
                       batch_size=self.batch_size, enc_batch_size=self.enc_batch_size,
                       noise_schedule=self.noise_schedule,
                       eds_item_type=eds_item_type, 
                       shuffle=True, device=self.device)
        train_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_train_ds, **dl_args)
        val_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_val_ds, **dl_args)
        return train_dl, val_dl
    
    def get_loggers(self, 
                    vae_net: vae.VarEncDec,
                    exps: List[Experiment]) -> chain_logger.ChainLogger:
        logger = super().get_loggers()
        # return logger

        dn_gen = dn_prog.DenoiseProgress(truth_is_noise=self.truth_is_noise,
                                         noise_schedule=self.noise_schedule,
                                         device=self.device,
                                         gen_steps=self.gen_steps,
                                         decoder_fn=vae_net.decode,
                                         latent_dim=vae_net.latent_dim)
        img_logger = \
            img_prog.ImageProgressLogger(basename=self.basename,
                                         progress_every_nepochs=self.progress_every_nepochs,
                                         generator=dn_gen,
                                         image_size=self.image_size,
                                         exps=exps)
        logger.loggers.append(img_logger)
        return logger


def parse_args() -> Config:
    cfg = Config()
    cfg.parse_args()
    return cfg

def build_experiments(cfg: Config, exps: List[Experiment],
                      train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
    exps = cfg.build_experiments(exps, train_dl, val_dl)
    if cfg.resume_shortcodes:
        exps = [exp for exp in exps if exp.shortcode in cfg.resume_shortcodes]
    return exps

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()

    # grab the first vae model that we can find..
    # "loss_type ~ edge",
    # "net_class = VarEncDec",
    # "net_do_residual != True",
    # f"net_image_size = {cfg.image_size}",

    exps = [exp for exp in checkpoint_util.list_experiments() 
            if exp.shortcode == cfg.vae_shortcode
            and exp.net_class == 'VarEncDec']
    if not len(exps):
        raise Exception(f"whoops, can't find VAE with shortcode {cfg.vae_shortcode}")
    vae_exp = exps[0]
    vae_path = vae_exp.get_run().checkpoint_path
    model_dict = torch.load(vae_path)

    vae_net = dn_util.load_model(model_dict=model_dict).to(cfg.device)
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

    # set up noising dataloaders that use vae_net as the decoder.
    train_dl, val_dl = cfg.get_dataloaders(vae_net=vae_net, vae_net_path=vae_path)

    exps: List[Experiment] = list()
    latent_dim = vae_net.latent_dim.copy()

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
        exp.image_size = vae_net.image_size
        exp.is_denoiser = True

        exp.train_dataloader = train_dl
        exp.val_dataloader = val_dl
        backing_loss = train_util.get_loss_fn(exp.loss_type, device=cfg.device)
        exp.loss_fn = \
            train_util.twotruth_loss_fn(backing_loss_fn=backing_loss,
                                        truth_is_noise=cfg.truth_is_noise, 
                                        device=cfg.device)

        label_parts = [
            f"noise_{cfg.noise_beta_type}_{cfg.noise_steps}",
            "img_latdim_" + "_".join(map(str, latent_dim)),
            f"noisefn_{cfg.noise_fn_str}",
            f"loss_{exp.loss_type}"
        ]
        if len(exp.label):
            exp.label += ","
        exp.label += ",".join(label_parts)


    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)
    logger = cfg.get_loggers(vae_net, exps)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=10)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
