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

import model_denoise
import model_denoise2
import model_new

DEFAULT_AMOUNT_MIN = 0.0
DEFAULT_AMOUNT_MAX = 1.0

device = "cuda"

class Config(cmdline_image.ImageTrainerConfig):
    truth_is_noise: bool
    use_timestep: bool
    attribute_matches: List[str]
    pattern: re.Pattern
    amount_min: float
    amount_max: float
    use_noise_steps: int
    noise_fn_str: str
    enc_batch_size: int
    gen_steps: List[int]
    
    noise_fn: Callable[[Tuple], Tensor] = None
    amount_fn: Callable[[], Tensor] = None

    checkpoints: List[Tuple[Path, Experiment]]

    def __init__(self):
        super().__init__("denoise")
        self.add_argument("--truth", choices=["noise", "src"], default="noise")
        self.add_argument("--use_timestep", default=False, action='store_true')
        self.add_argument("--noise_fn", dest='noise_fn_str', default='normal', choices=['rand', 'normal'])
        self.add_argument("--amount_min", type=float, default=DEFAULT_AMOUNT_MIN)
        self.add_argument("--amount_max", type=float, default=DEFAULT_AMOUNT_MAX)
        self.add_argument("--use_noise_steps", type=int, default=0)
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
        # print("\n  ".join([format(exp.lastepoch_train_loss, ".3f") for path, exp in self.checkpoints]))
        # sys.exit(0)
        
        self.truth_is_noise = (self.truth == "noise")

        if self.use_noise_steps:
            self.amount_fn = None
        else:
            self.amount_fn = noised_data.gen_amount_range(self.amount_min, self.amount_max)

        if self.noise_fn_str == "rand":
            self.noise_fn = noised_data.gen_noise_rand
        elif self.noise_fn_str == "normal":
            self.noise_fn = noised_data.gen_noise_normal
        else:
            raise ValueError(f"logic error: unknown {self.noise_fn_str=}")
        
        if self.use_noise_steps and (self.amount_min != DEFAULT_AMOUNT_MIN or self.amount_max != DEFAULT_AMOUNT_MAX):
            self.error(f"--use_noise_steps cannot be used with non-default --amount_min {self.amount_min:.3f} or --amount_max {self.amount_max:.3f}")
        
        if self.use_noise_steps and not self.use_timestep:
            self.error("must set --use_timestep when using --use_noise_steps")

        return self

    def get_dataloaders(self, vae_net: model_new.VarEncDec, vae_net_path: Path) -> Tuple[DataLoader, DataLoader]:
        src_train_dl, src_val_dl = super().get_dataloaders()
        train_dl, val_dl = \
            model_denoise.get_dataloaders(vae_net=vae_net,
                                          vae_net_path=vae_net_path,
                                          src_train_dl=src_train_dl,
                                          src_val_dl=src_val_dl,
                                          batch_size=self.enc_batch_size,
                                          amount_fn=self.amount_fn,
                                          noise_fn=self.noise_fn,
                                          use_timestep=self.use_timestep,
                                          use_noise_steps=self.use_noise_steps,
                                          device=self.device)
        return train_dl, val_dl
    
    def get_loggers(self, 
                    vae_net: model_new.VarEncDec,
                    exps: List[Experiment]) -> chain_logger.ChainLogger:
        logger = super().get_loggers()
        dn_gen = dn_prog.DenoiseProgress(truth_is_noise=self.truth_is_noise,
                                         use_timestep=self.use_timestep,
                                         noise_fn=self.noise_fn, 
                                         amount_fn=self.amount_fn,
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
    for exp in exps:
        exp.loss_fn = noised_data.twotruth_loss_fn(loss_type=exp.loss_type, truth_is_noise=cfg.truth_is_noise, device=device)

        if cfg.use_timestep:
            exp.use_timestep = True
            exp.label += ",timestep"

        exp.truth_is_noise = cfg.truth_is_noise
        exp.label += f",noisefn_{cfg.noise_fn_str}"
        if cfg.amount_min != DEFAULT_AMOUNT_MIN or cfg.amount_max != DEFAULT_AMOUNT_MAX:
            exp.label += f",amount_{cfg.amount_min:.2f}_{cfg.amount_max:.2f}"
        exp.amount_min = cfg.amount_min
        exp.amount_max = cfg.amount_max

    exps = cfg.build_experiments(exps, train_dl, val_dl)

    return exps

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()

    # grab the first vae model that we can find..
    first_path, first_exp = cfg.checkpoints[0]
    print(f"{first_exp.lastepoch_train_loss=:.3f}")
    print(f"{first_exp.lastepoch_val_loss=:.3f}")
    print(f"{first_exp.nepochs=}")
    print(f"{first_exp.saved_at=}")
    print(f"{first_exp.saved_at_relative()=}")
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
            # return model_denoise.DenoiseModel(**kwargs)
            net = model_denoise.DenoiseModel(**kwargs)
            # print(net)
            return net
        return fn
        
    # TODO hacked up experiment
    layer_str_values = [
        "k3-s2-64-128-256-512",
        "k3-s1-64-s2-128-256-512",
        # "k3-s2-32-64-128-256",

        "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [64, 128, 256, 512]]),
        # "k3-" + "-".join([f"s1-{chan}x1-s2-{chan}" for chan in [64, 128, 256, 512]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [32, 64, 128, 256]]),
        # "k3-" + "-".join([f"s1-{chan}x1-s2-{chan}" for chan in [32, 64, 128, 256]]),

        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [32, 16, 8]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [64, 32, 16]]),
        # "k3-" + "-".join([f"s1-{chan}x2-s2-{chan}" for chan in [128, 64, 32, 16]]),

        ("k3-s1-8x2-16x2-32x2"),
        ("k3-s1-32x2-16x2-8x2"),

        ("k3-s1-8x4-16x4-32x4"),
        ("k3-s1-32x4-16x4-8x4"),

        ("k4-s1-64x2-32x2-16x2"),
        ("k5-s1-64x2-32x2-16x2"),
    ]
    for do_residual in [False]:
        for layer_str in layer_str_values:
            exp = Experiment()
            conv_cfg = conv_types.make_config(layer_str, final_nl_type='relu')
            exp.startlr = cfg.startlr or 1e-4
            exp.endlr = cfg.endlr or 1e-5
            exp.sched_type = "nanogpt"
            exp.loss_type = "l2"
            exp.lazy_dataloaders_fn = lambda exp: train_dl, val_dl
            exp.use_noise_steps = cfg.use_noise_steps

            lat_chan, lat_size, _ = vae_net.latent_dim
            dn_chan = conv_cfg.get_channels_down(lat_chan)[-1]
            dn_size = conv_cfg.get_sizes_down_actual(lat_size)[-1]
            dn_dim = [dn_chan, dn_size, dn_size]
            
            label_parts = [
                f"denoise-{layer_str}",
                "vaedim_" + "_".join(map(str, vae_net.latent_dim)),
                "dndim_" + "_".join(map(str, dn_dim))
            ]
            if do_residual:
                label_parts.append("residual")
            if cfg.use_noise_steps:
                label_parts.append(f"nsteps_{cfg.use_noise_steps}")

            exp.label = ",".join(label_parts)

            args = dict(in_latent_dim=vae_net.latent_dim,
                        cfg=conv_cfg, # do_residual=do_residual,
                        use_timestep=cfg.use_timestep)
            exp.lazy_net_fn = lazy_net_fn(args)
            exps.append(exp)

    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)

    # TODO
    logger = cfg.get_loggers(vae_net, exps)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=10)
    t.train(device=device, use_amp=cfg.use_amp)
