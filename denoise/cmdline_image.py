from typing import Tuple

from torch.utils.data import Dataset, DataLoader

import cmdline
import image_util

from loggers import tensorboard as tb_logger
from loggers import chain as chain_logger
from loggers import checkpoint as ckpt_logger
from loggers import csv as csv_logger


class ImageTrainerConfig(cmdline.TrainerConfig):
    config_file: str
    image_size: int
    image_dir: str
    save_top_k: int
    num_progress: int
    progress_every_nepochs: int
    limit_dataset: int

    no_checkpoints: bool
    no_tensorboard: bool
    no_log: bool

    def __init__(self, basename: str = ""):
        super().__init__(basename=basename)
        self.add_argument("-c", "--config_file", default=None)
        self.add_argument("-I", "--image_size", default=128, type=int)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-k", "--save_top_k", default=1, type=int)
        self.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=10)
        self.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)
        self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("--no_checkpoints", default=False, action='store_true', help="debugging: disable writing of checkpoints")
        self.add_argument("--no_tensorboard", default=False, action='store_true', help="debugging: disable tensorboard")
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
            self.no_tensorboard = True
            self.no_checkpoints = True

        if not self.no_tensorboard:
            logger.loggers.append(tb_logger.TensorboardLogger(basename=self.basename, started_at=self.started_at))

        if not self.no_checkpoints:
            cp_logger = \
                ckpt_logger.CheckpointLogger(basename=self.basename, started_at=self.started_at, 
                                             save_top_k=self.save_top_k)
            logger.loggers.append(cp_logger)
        
        return logger

    def get_datasets(self, train_split = 0.9) -> Tuple[Dataset, Dataset]:
        return image_util.get_datasets(image_size=self.image_size,
                                       image_dir=self.image_dir,
                                       train_split=train_split,
                                       limit_dataset=self.limit_dataset)

    def get_dataloaders(self, train_split = 0.9, shuffle = True) -> Tuple[DataLoader, DataLoader]:
        return image_util.get_dataloaders(image_size=self.image_size,
                                          image_dir=self.image_dir,
                                          train_split=train_split,
                                          shuffle=shuffle,
                                          batch_size=self.batch_size,
                                          limit_dataset=self.limit_dataset)
