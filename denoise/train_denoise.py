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
import denoise_progress
import model_denoise
import conv_types
import dn_util
import image_util
import image_latents
import cmdline_image
import checkpoint_util
import re

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
    noise_fn: str
    enc_batch_size: int

    checkpoints: List[Tuple[Path, Experiment]]

    def __init__(self):
        super().__init__("denoise")
        self.add_argument("--truth", choices=["noise", "src"], default="noise")
        self.add_argument("--use_timestep", default=False, action='store_true')
        self.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
        self.add_argument("--amount_min", type=float, default=DEFAULT_AMOUNT_MIN)
        self.add_argument("--amount_max", type=float, default=DEFAULT_AMOUNT_MAX)
        self.add_argument("-B", "--enc_batch_size", type=int, default=2)
        self.add_argument("-p", "--pattern", type=str, default=None)
        self.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
        self.add_argument("-s", "--sort", dest='sort_key', default='time')

    def parse_args(self) -> 'Config':
        super().parse_args()
        if self.pattern:
            self.pattern = re.compile(self.pattern)

        self.checkpoints = \
            checkpoint_util.find_checkpoints(attr_matchers=self.attribute_matchers,
                                             only_paths=self.pattern)
        
        self.truth_is_noise = (self.truth == "noise")


        return self

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
        exp.label += f",noisefn_{cfg.noise_fn}"
        exp.label += f",amount_{cfg.amount_min:.2f}_{cfg.amount_max:.2f}"
        exp.amount_min = cfg.amount_min
        exp.amount_max = cfg.amount_max

    exps = cfg.build_experiments(exps, train_dl, val_dl)

    return exps

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    exps: List[Experiment] = list()
    # with open(cfg.config_file, "r") as cfile:
    #     print(f"reading {cfg.config_file}")
    #     exec(cfile.read())
    

    amount_fn = noised_data.gen_amount_range(cfg.amount_min, cfg.amount_max)
    if cfg.noise_fn == "rand":
        noise_fn = noised_data.gen_noise_rand
    elif cfg.noise_fn == "normal":
        noise_fn = noised_data.gen_noise_normal
    else:
        raise ValueError(f"logic error: unknown {cfg.noise_fn=}")


    first_path, first_exp = cfg.checkpoints[0]
    with open(first_path, "rb") as file:
        model_dict = torch.load(file)

    vae_net = dn_util.load_model(model_dict=model_dict).to(cfg.device)
    vae_net.requires_grad_(False)
    vae_net.eval()
    
    src_train_dl, src_val_dl = cfg.get_dataloaders()

    encds_args = dict(net=vae_net, enc_batch_size=cfg.enc_batch_size, device=cfg.device)
    train_ds = image_latents.EncoderDataset(dataloader=src_train_dl, **encds_args)
    val_ds = image_latents.EncoderDataset(dataloader=src_val_dl, **encds_args)

    noiseds_args = dict(use_timestep=cfg.use_timestep, noise_fn=noise_fn, amount_fn=amount_fn)
    train_ds = noised_data.NoisedDataset(base_dataset=train_ds, **noiseds_args)
    val_ds = noised_data.NoisedDataset(base_dataset=val_ds, **noiseds_args)

    train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=cfg.batch_size)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, batch_size=cfg.batch_size)


    # TODO hacked up experiment
    layer_str = "k3-s2-32-16-8"
    label = f"denoise-{layer_str}"
    exp = Experiment(label=label)
    conv_cfg = conv_types.make_config(layer_str)
    exp.startlr = 1e-3
    exp.endlr = 1e-4
    exp.loss_type = "l2_sqrt"
    exp.lazy_net_fn = lambda exp: model_denoise.DenoiseModel(vae_net.latent_dim, cfg=conv_cfg)
    exp.lazy_dataloaders_fn = lambda exp: train_dl, val_dl
    exps = [exp]
    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)

    logger = cfg.get_loggers()

    # TODO
    # ae_gen = ae_progress.AutoencoderProgress(device=cfg.device)
    # img_logger = im_prog.ImageProgressLogger(dirname=dirname,
    #                                         progress_every_nepochs=cfg.progress_every_nepochs,
    #                                         generator=ae_gen,
    #                                         image_size=cfg.image_size,
    #                                         exps=exps)
    # logger.loggers.append(img_logger)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=device, use_amp=cfg.use_amp)
