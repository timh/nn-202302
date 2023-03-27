# %%
import sys
import argparse
import datetime
from typing import List, Literal, Tuple, Dict, Callable
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
import noised_data
import conv_types
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

class Config(cmdline_image.ImageTrainerConfig):
    truth_is_noise: bool
    attribute_matches: List[str]
    pattern: re.Pattern
    enc_batch_size: int
    gen_steps: List[int]

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
        self.add_argument("-p", "--pattern", type=str, default=None)
        self.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])

    def parse_args(self) -> 'Config':
        super().parse_args()
        if self.pattern:
            self.pattern = re.compile(self.pattern)

        self.checkpoints = \
            checkpoint_util.find_checkpoints(attr_matchers=self.attribute_matchers,
                                             only_paths=self.pattern)
        self.checkpoints = sorted(self.checkpoints,
                                  key=lambda tup: tup[1].lastepoch_train_loss)
        
        self.truth_is_noise = (self.truth == "noise")

        self.noise_schedule = \
            noisegen.make_noise_schedule(type=self.noise_beta_type,
                                         timesteps=self.noise_steps,
                                         noise_type=self.noise_fn_str)
        return self
    
    def get_dataloaders(self, vae_net: vae.VarEncDec, vae_net_path: Path) -> Tuple[DataLoader, DataLoader]:
        src_train_ds, src_val_ds = super().get_datasets()

        dl_args = dict(vae_net=vae_net, vae_net_path=vae_net_path,
                       batch_size=self.batch_size,
                       noise_schedule=self.noise_schedule,
                       shuffle=True, device=self.device)
        train_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_train_ds, **dl_args)
        val_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_val_ds, **dl_args)
        return train_dl, val_dl
    
    def get_loggers(self, 
                    vae_net: vae.VarEncDec,
                    exps: List[Experiment]) -> chain_logger.ChainLogger:
        logger = super().get_loggers()
        dn_gen = dn_prog.DenoiseProgress(truth_is_noise=self.truth_is_noise,
                                         noise_schedule=self.noise_schedule,
                                         device=self.device,
                                         gen_steps=self.gen_steps,
                                         decoder_fn=vae_net.decode)
        img_logger = \
            img_prog.ImageProgressLogger(dirname=self.log_dirname,
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
    # NOTE: need to update do local processing BEFORE calling super().build_experiments,
    # because it looks for resume candidates and uses image_dir as part of that..
    return cfg.build_experiments(exps, train_dl, val_dl, resume_ignore_fields={'net_vae_path'})

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()

    # grab the first vae model that we can find..
    first_path, first_exp = cfg.checkpoints[0]
    print(f"{first_path}:")
    print(f"  {first_exp.lastepoch_train_loss=:.3f}")
    print(f"  {first_exp.lastepoch_val_loss=:.3f}")
    print(f"  {first_exp.nepochs=}")
    print(f"  {first_exp.saved_at=}")
    print(f"  {first_exp.saved_at_relative()=}")
    with open(first_path, "rb") as file:
        model_dict = torch.load(file)

    vae_net = dn_util.load_model(model_dict=model_dict).to(cfg.device)
    vae_net.requires_grad_(False)
    vae_net.eval()

    # set up noising dataloaders that use vae_net as the decoder.
    train_dl, val_dl = cfg.get_dataloaders(vae_net=vae_net, vae_net_path=first_path)

    exps: List[Experiment] = list()
    # load config file
    # with open(cfg.config_file, "r") as cfile:
    #     print(f"reading {cfg.config_file}")
    #     exec(cfile.read())
    def lazy_net_fn(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(exp: Experiment) -> nn.Module:
            # return denoise.DenoiseModel(**kwargs)
            net = denoise.DenoiseModel(**kwargs)
            # print(net)
            return net
        return fn
        
    # TODO hacked up experiment
    layer_str_values = [
        # "k3-s2-64-128-256-512",
        # "k3-s1-64-s2-128-256-512",
        # "k3-s2-32-64-128-256",

        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [64, 128, 256, 512]]),
        # "k3-" + "-".join([f"s1-{chan}x1-s2-{chan}" for chan in [64, 128, 256, 512]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [32, 64, 128, 256]]),
        # "k3-" + "-".join([f"s1-{chan}x1-s2-{chan}" for chan in [32, 64, 128, 256]]),

        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [32, 16, 8]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [64, 32, 16]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [128, 64, 32, 16]]),

        # ("k3-s1-8x2-16x2-32x2"),
        ("k3-s1-32x2-16x2-8x2"),    # best

        # ("k3-s1-8x4-16x4-32x4"),
        # ("k3-s1-32x4-16x4-8x4"),

        # ("k4-s1-64x2-32x2-16x2"),
        # ("k5-s1-64x2-32x2-16x2"),
    ]
    for do_residual in [False]:
        for layer_str in layer_str_values:
            exp = Experiment()
            conv_cfg = conv_types.make_config(layer_str, final_nl_type='relu')
            exp.lazy_dataloaders_fn = lambda exp: train_dl, val_dl
            exp.startlr = cfg.startlr or 1e-4
            exp.endlr = cfg.endlr or 1e-5
            exp.sched_type = "nanogpt"
            # exp.loss_type = "l2"
            exp.loss_type = "l1_smooth"
            exp.noise_steps = cfg.noise_steps
            exp.noise_beta_type = cfg.noise_beta_type
            exp.loss_fn = \
                noised_data.twotruth_loss_fn(loss_type=exp.loss_type, 
                                             truth_is_noise=cfg.truth_is_noise, 
                                             device=cfg.device)
            exp.truth_is_noise = cfg.truth_is_noise

            lat_chan, lat_size, _ = vae_net.latent_dim
            dn_chan = conv_cfg.get_channels_down(lat_chan)[-1]
            dn_size = conv_cfg.get_sizes_down_actual(lat_size)[-1]
            dn_dim = [dn_chan, dn_size, dn_size]
            
            label_parts = [
                f"denoise-{layer_str}",
                "vaedim_" + "_".join(map(str, vae_net.latent_dim)),
                "dndim_" + "_".join(map(str, dn_dim)),
                f"noisefn_{cfg.noise_fn_str}"
            ]
            if do_residual:
                label_parts.append("residual")
            label_parts.append(f"noise_{cfg.noise_beta_type}_{cfg.noise_steps}")

            exp.label = ",".join(label_parts)

            args = dict(in_latent_dim=vae_net.latent_dim, cfg=conv_cfg)
            exp.lazy_net_fn = lazy_net_fn(args)
            exps.append(exp)

    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)

    logger = cfg.get_loggers(vae_net, exps)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=10)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
