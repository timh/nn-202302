from typing import List
from torch.utils.data import Dataset, DataLoader

import cmdline
import image_util
from experiment import Experiment

from loggers import tensorboard as tb_logger
from loggers import wandb_logger
from loggers import chain as chain_logger
from loggers import checkpoint as ckpt_logger
# from loggers import csv as csv_logger

class ImageTrainerConfig(cmdline.TrainerConfig):
    image_size: int
    image_dir: str
    save_top_k: int
    num_progress: int
    progress_every_nepochs: int
    # limit_dataset: int

    do_checkpoints: bool
    do_tensorboard: bool
    do_wandb: bool
    no_log: bool

    def __init__(self, basename: str = ""):
        super().__init__(basename=basename)
        self.add_argument("-I", "--image_size", default=512, type=int)
        self.add_argument("-d", "--image_dir", default="images.2018-2020")
        self.add_argument("-k", "--save_top_k", default=1, type=int)
        self.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=10)
        self.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)
        # self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("--no_checkpoints", dest='do_checkpoints', default=True, action='store_false', help="debugging: disable writing of checkpoints")
        self.add_argument("--tensorboard", dest='do_tensorboard', default=False, action='store_true', help="enable wandb")
        self.add_argument("--no_wandb", dest='do_wandb', default=True, action='store_false', help="disable wandb")
        self.add_argument("--no_log", default=False, action='store_true', help="debugging: disable tensorboard & checkpoints")
    
    def parse_args(self) -> 'ImageTrainerConfig':
        super().parse_args()
        if self.num_progress and self.progress_every_nepochs:
            self.error("specify only one of --num_progress and --progress_every_nepochs")
        
        if self.num_progress:
            self.num_progress = min(self.max_epochs, self.num_progress)
            self.progress_every_nepochs = self.max_epochs // self.num_progress
        
        return self

    def get_loggers(self) -> chain_logger.ChainLogger:
        logger = chain_logger.ChainLogger()

        if self.no_log:
            self.do_tensorboard = False
            self.do_checkpoints = False
            self.do_wandb = False

        if self.do_tensorboard:
            logger.loggers.append(tb_logger.TensorboardLogger(basename=self.basename, started_at=self.started_at))

        if self.do_wandb:
            logger.loggers.append(wandb_logger.WandbLogger(basename=self.basename, started_at=self.started_at))

        if self.do_checkpoints:
            cp_logger = \
                ckpt_logger.CheckpointLogger(basename=self.basename, started_at=self.started_at, 
                                             save_top_k=self.save_top_k)
            logger.loggers.append(cp_logger)
        
        return logger

    def get_dataset(self) -> Dataset:
        return image_util.get_dataset(image_size=self.image_size, image_dir=self.image_dir)

