# %%
import sys
import argparse
from typing import List
from pathlib import Path

import torch
from torch import nn

sys.path.append("..")
import trainer
from experiment import Experiment
import noised_data
import model
import denoise_logger

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
    parser.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=20)
    parser.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)

    if denoise_logger.in_notebook():
        dev_args = "-n 10 -c conf/conv_sd.py -b 16 --use_timestep".split(" ")
        cfg = parser.parse_args(dev_args)
    else:
        cfg = parser.parse_args()

    truth_is_noise = (cfg.truth == "noise")

    if cfg.num_progress and cfg.progress_every_nepochs:
        parser.error(f"specify only one of --num_progress and --progress_every_nepochs")
    
    if cfg.num_progress:
        cfg.num_progress = min(cfg.max_epochs, cfg.num_progress)
        cfg.progress_every_nepochs = cfg.max_epochs // cfg.num_progress

    device = "cuda"
    torch.set_float32_matmul_precision('high')

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    net: nn.Module = None
    batch_size: int = 128
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())
    
    # these params are often set by configs, but can be overridden here.
    if cfg.batch_size is not None:
        batch_size = cfg.batch_size

    dataset = noised_data.load_dataset(image_dirname=cfg.image_dir, image_size=cfg.image_size,
                                       use_timestep=cfg.use_timestep)
    train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, 
                                                      train_all_data=True, val_all_data=True)

    for exp in exps:
        exp.loss_type = getattr(exp, "loss_type", "l1")

        exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
        exp.lazy_optim_fn = trainer.lazy_optim_fn
        exp.lazy_sched_fn = trainer.lazy_sched_fn
        exp.device = device
        exp.optim_type = exp.optim_type or "adamw"
        exp.sched_type = exp.sched_type or "nanogpt"
        exp.max_epochs = exp.max_epochs or cfg.max_epochs

        exp.label += f",loss_{exp.loss_type}"
        exp.label += f",batch_{batch_size}"
        exp.label += f",slr_{exp.startlr:.1E}"
        exp.label += f",elr_{exp.endlr:.1E}"
        if cfg.no_compile:
            exp.do_compile = False
        if exp.do_compile:
            exp.label += ",compile"
        if cfg.use_amp:
            exp.use_amp = True
            exp.label += ",useamp"
        if cfg.use_timestep:
            exp.use_timestep = True
            exp.label += ",timestep"
        exp.truth_is_noise = truth_is_noise
        if truth_is_noise:
            exp.label += ",truth_is_noise"

        exp.loss_fn = noised_data.twotruth_loss_fn(loss_type=exp.loss_type, truth_is_noise=truth_is_noise, device=device)

    for i, exp in enumerate(exps):
        print(f"#{i + 1} {exp.label}")
    print()

    basename = Path(cfg.config_file).stem

    logger = denoise_logger.DenoiseLogger(basename=basename, 
                                          truth_is_noise=truth_is_noise, use_timestep=cfg.use_timestep,
                                          save_top_k=cfg.save_top_k, max_epochs=cfg.max_epochs,
                                          progress_every_nepochs=cfg.progress_every_nepochs,
                                          device=device)
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=device, use_amp=cfg.use_amp)

# %%


