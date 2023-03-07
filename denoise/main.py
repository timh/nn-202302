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
    parser.add_argument("-n", "--epochs", type=int, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-I", "--image_size", default=128, type=int)
    parser.add_argument("-d", "--image_dir", default="alex-many-128")
    parser.add_argument("-k", "--save_top_k", default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=None)
    parser.add_argument("--minicnt", type=int, default=None)
    parser.add_argument("--startlr", type=float, default=1e-3)
    parser.add_argument("--endlr", type=float, default=1e-4)
    parser.add_argument("--truth", choices=["noise", "src"], default="src")
    parser.add_argument("--no_compile", default=False, action='store_true')

    if denoise_logger.in_notebook():
        # dev_args = "-c conf/conv_encdec2.py -n 200".split(" ")
        # dev_args = "-c conf/conv_encdec2.py -n 100".split(" ")
        dev_args = "-c conf/conv_encdec2.py -n 200 -b 64 --minicnt 2".split(" ")
        cfg = parser.parse_args(dev_args)
    else:
        cfg = parser.parse_args()

    truth_is_noise = (cfg.truth == "noise")

    device = "cuda"
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    loss_fn = noised_data.twotruth_loss_fn(truth_is_noise=truth_is_noise)
    torch.set_float32_matmul_precision('high')

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    net: nn.Module = None
    batch_size: int = 128
    minicnt: int = 10
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())
    
    # these params are often set by configs, but can be overridden here.
    if cfg.batch_size is not None:
        batch_size = cfg.batch_size
    if cfg.minicnt is not None:
        minicnt = cfg.minicnt

    dataset = noised_data.load_dataset(image_dirname=cfg.image_dir, image_size=cfg.image_size)
    train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, minicnt=minicnt, val_all_data=True)

    for exp in exps:
        exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
        exp.lazy_optim_fn = trainer.lazy_optim_fn
        exp.lazy_sched_fn = trainer.lazy_sched_fn
        exp.device = device
        if exp.loss_fn is None:
            exp.loss_fn = loss_fn
        if exp.startlr is None:
            exp.startlr = cfg.startlr
        if exp.endlr is None:
            exp.endlr = cfg.endlr
        if not exp.epochs:
            exp.epochs = cfg.epochs
        exp.label += f",slr_{exp.startlr:.1E}"
        if exp.sched_type != "constant":
            exp.label += f",elr_{exp.endlr:.1E}"
        exp.label += f",batch_{batch_size}"
        exp.label += f",cnt_{minicnt}"
        if cfg.no_compile:
            exp.do_compile = False

    for i, exp in enumerate(exps):
        print(f"#{i + 1} {exp.label}")
    print()

    basename = Path(cfg.config_file).stem

    logger = denoise_logger.DenoiseLogger(basename=basename, truth_is_noise=truth_is_noise, save_top_k=cfg.save_top_k, epochs=cfg.epochs, device=device)
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), 
                        logger=logger, update_frequency=30, desired_val_count=5)
    t.train(device=device)

# %%


