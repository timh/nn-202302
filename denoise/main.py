# %%
import sys
import argparse
import datetime
from typing import List
from pathlib import Path

import torch
from torch import nn

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
import noised_data
import model
import denoise_progress
import dn_util

from loggers import tensorboard as tb_logger
from loggers import chain as chain_logger
from loggers import image_progress as im_prog
from loggers import checkpoint as ckpt_logger


DEFAULT_AMOUNT_MIN = 0.0
DEFAULT_AMOUNT_MAX = 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_epochs", type=int, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-I", "--image_size", default=128, type=int)
    parser.add_argument("-d", "--image_dir", default="alex-many-128")
    parser.add_argument("-k", "--save_top_k", default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=None)
    parser.add_argument("--startlr", type=float, default=1e-3)
    parser.add_argument("--endlr", type=float, default=1e-4)
    parser.add_argument("--truth", choices=["noise", "src"], default="src")
    parser.add_argument("--no_compile", default=False, action='store_true')
    parser.add_argument("--amp", dest="use_amp", default=False, action='store_true')
    parser.add_argument("--use_timestep", default=False, action='store_true')
    parser.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=10)
    parser.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)
    parser.add_argument("--noise_fn", default='rand', choices=['rand', 'normal'])
    parser.add_argument("--amount_min", type=float, default=DEFAULT_AMOUNT_MIN)
    parser.add_argument("--amount_max", type=float, default=DEFAULT_AMOUNT_MAX)
    parser.add_argument("--disable_noise", default=False, action='store_true', help="disable noise dataloader, generation, etc. use for training a VAE")
    parser.add_argument("--limit_dataset", default=None, type=int)
    parser.add_argument("--no_checkpoints", default=False, action='store_true')

    if False and denoise_logger.in_notebook():
        dev_args = "-n 10 -c conf/conv_sd.py -b 16 --use_timestep".split(" ")
        cfg = parser.parse_args(dev_args)
    else:
        cfg = parser.parse_args()

    truth_is_noise = (cfg.truth == "noise")
    if cfg.disable_noise and truth_is_noise:
        parser.error("specify only one of --disable_noise and --truth_is_noise")

    if cfg.num_progress and cfg.progress_every_nepochs:
        parser.error("specify only one of --num_progress and --progress_every_nepochs")
    
    if cfg.num_progress:
        cfg.num_progress = min(cfg.max_epochs, cfg.num_progress)
        cfg.progress_every_nepochs = cfg.max_epochs // cfg.num_progress

    device = "cuda"
    torch.set_float32_matmul_precision('high')

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    net: nn.Module = None
    batch_size: int = 32
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())
    
    # these params are often set by configs, but can be overridden here.
    if cfg.batch_size is not None:
        batch_size = cfg.batch_size

    if cfg.disable_noise:
        amount_fn = None
        noise_fn = None
    else:
        if cfg.noise_fn == "rand":
            noise_fn = noised_data.gen_noise_rand
        elif cfg.noise_fn == "normal":
            noise_fn = noised_data.gen_noise_normal
        else:
            raise ValueError(f"unknown {cfg.noise_fn=}")
        amount_fn = noised_data.gen_amount_range(cfg.amount_min, cfg.amount_max)

    train_dl, val_dl = dn_util.get_dataloaders(disable_noise=cfg.disable_noise,
                                               use_timestep=cfg.use_timestep,
                                               noise_fn=noise_fn, amount_fn=amount_fn,
                                               image_size=cfg.image_size,
                                               image_dir=cfg.image_dir,
                                               batch_size=batch_size,
                                               limit_dataset=cfg.limit_dataset)

    for exp in exps:
        exp.loss_type = getattr(exp, "loss_type", "l1")

        exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
        exp.lazy_optim_fn = train_util.lazy_optim_fn
        exp.lazy_sched_fn = train_util.lazy_sched_fn
        exp.device = device
        exp.optim_type = exp.optim_type or "adamw"
        exp.sched_type = exp.sched_type or "nanogpt"
        exp.max_epochs = exp.max_epochs or cfg.max_epochs

        if exp.loss_fn is None:
            exp.label += f",loss_{exp.loss_type}"
            if getattr(exp, 'do_variational', None):
                exp.label += "+kl"

            if exp.truth_is_noise:
                exp.loss_fn = train_util.get_loss_fn(loss_type=exp.loss_type, device=device)
            else:
                exp.loss_fn = noised_data.twotruth_loss_fn(loss_type=exp.loss_type, truth_is_noise=truth_is_noise, device=device)
        
        exp.label += f",batch_{batch_size}"
        exp.label += f",slr_{exp.startlr:.1E}"
        exp.label += f",elr_{exp.endlr:.1E}"

        if cfg.limit_dataset:
            exp.label += f",limit-ds_{cfg.limit_dataset}"

        if cfg.no_compile:
            exp.do_compile = False

        if exp.do_compile:
            exp.label += ",compile"

        if cfg.use_amp:
            exp.use_amp = True
            exp.label += ",useamp"

        if not cfg.disable_noise:
            if cfg.use_timestep:
                exp.use_timestep = True
                exp.label += ",timestep"

            exp.truth_is_noise = truth_is_noise
            if truth_is_noise:
                exp.label += ",truth_is_noise"
            exp.label += f",noisefn_{cfg.noise_fn}"
            exp.label += f",amount_{cfg.amount_min:.2f}_{cfg.amount_max:.2f}"
            exp.amount_min = cfg.amount_min
            exp.amount_max = cfg.amount_max

    for i, exp in enumerate(exps):
        print(f"#{i + 1} {exp.label}")
    print()

    basename = Path(cfg.config_file).stem

    now = datetime.datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    dirname = f"runs/denoise-{basename}_{cfg.max_epochs:03}_{timestr}"

    noiselog_gen = denoise_progress.DenoiseProgress(truth_is_noise=truth_is_noise, 
                                                    use_timestep=cfg.use_timestep, 
                                                    disable_noise=cfg.disable_noise,
                                                    noise_fn=noise_fn, amount_fn=amount_fn,
                                                    device=device)
    img_logger = im_prog.ImageProgressLogger(dirname=dirname,
                                             progress_every_nepochs=cfg.progress_every_nepochs,
                                             generator=noiselog_gen,
                                             image_size=(cfg.image_size, cfg.image_size))
    logger = chain_logger.ChainLogger()
    logger.loggers.append(tb_logger.TensorboardLogger(dirname=dirname))
    if not cfg.no_checkpoints:
        logger.loggers.append(ckpt_logger.CheckpointLogger(dirname=dirname, save_top_k=cfg.save_top_k))
    logger.loggers.append(img_logger)

    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=device, use_amp=cfg.use_amp)
