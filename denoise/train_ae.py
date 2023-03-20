# %%
import sys
import datetime
from typing import List, Literal, Dict, Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
# import denoise_progress
import ae_progress
import model_util
import checkpoint_util
import cmdline
import image_util

from loggers import tensorboard as tb_logger
from loggers import chain as chain_logger
from loggers import image_progress as im_prog
from loggers import checkpoint as ckpt_logger
from loggers import csv as csv_logger

DEFAULT_AMOUNT_MIN = 0.0
DEFAULT_AMOUNT_MAX = 1.0

device = "cuda"

class TrainerConfig(cmdline.TrainerConfig):
    config_file: str
    image_size: int
    image_dir: str
    save_top_k: int
    num_progress: int
    progress_every_nepochs: int
    limit_dataset: int
    no_checkpoints: bool

    def __init__(self):
        super().__init__(basename="denoise")
        self.add_argument("-c", "--config_file", type=str, required=True)
        self.add_argument("-I", "--image_size", default=128, type=int)
        self.add_argument("-d", "--image_dir", default="alex-many-128")
        self.add_argument("-k", "--save_top_k", default=1)
        self.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=10)
        self.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)
        self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("--no_checkpoints", default=False, action='store_true', help="debugging: disable writing of checkpoints")

def parse_args() -> TrainerConfig:
    cfg = TrainerConfig()
    cfg.parse_args()

    if cfg.num_progress and cfg.progress_every_nepochs:
        cfg.error("specify only one of --num_progress and --progress_every_nepochs")
    
    if cfg.num_progress:
        cfg.num_progress = min(cfg.max_epochs, cfg.num_progress)
        cfg.progress_every_nepochs = cfg.max_epochs // cfg.num_progress
    
    return cfg

def build_experiments(cfg: TrainerConfig, exps: List[Experiment],
                      train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
    # NOTE: need to update do local processing BEFORE calling super().build_experiments,
    # because it looks for resume candidates and uses image_dir as part of that..
    for exp in exps:
        exp.image_dir = cfg.image_dir
        if cfg.limit_dataset:
            exp.label += f",limit-ds_{cfg.limit_dataset}"

    exps = cfg.build_experiments(exps, train_dl, val_dl)

    return exps

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()
    cfg.basename = f"denoise_{Path(cfg.config_file).stem}"
    dirname = cfg.log_dirname

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())
    
    # these params are often set by configs, but can be overridden here.
    train_dl, val_dl = \
        image_util.get_dataloaders(image_size=cfg.image_size,
                                   image_dir=cfg.image_dir,
                                   batch_size=cfg.batch_size,
                                   limit_dataset=cfg.limit_dataset)

    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)
    
    # build loggers
    ae_gen = ae_progress.AutoencoderProgress(device=device)
    img_logger = im_prog.ImageProgressLogger(dirname=dirname,
                                             progress_every_nepochs=cfg.progress_every_nepochs,
                                             generator=ae_gen,
                                             image_size=cfg.image_size,
                                             exps=exps)
    logger = chain_logger.ChainLogger()
    # logger.loggers.append(csv_logger.CsvLogger(Path("runs/experiments.csv"), runpath=Path(dirname)))
    logger.loggers.append(tb_logger.TensorboardLogger(dirname=dirname))
    if not cfg.no_checkpoints:
        skip_similar = True
        if cfg.do_resume:
            skip_similar = False
        cplogger = ckpt_logger.CheckpointLogger(dirname=dirname, save_top_k=cfg.save_top_k, 
                                                skip_similar=skip_similar)
        logger.loggers.append(cplogger)
    logger.loggers.append(img_logger)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=device, use_amp=cfg.use_amp)
