import sys

import cmdline
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

    def __init__(self):
        super().__init__(basename="")
        self.add_argument("-c", "--config_file", type=str, required=True)
        self.add_argument("-I", "--image_size", default=128, type=int)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-k", "--save_top_k", default=1, type=int)
        self.add_argument("--progress", "--num_progress", dest='num_progress', type=int, default=10)
        self.add_argument("--progress_every_nepochs", dest='progress_every_nepochs', type=int, default=None)
        self.add_argument("--limit_dataset", default=None, type=int, help="debugging: limit the size of the dataset")
        self.add_argument("--no_checkpoints", default=False, action='store_true', help="debugging: disable writing of checkpoints")
    
    def parse_args(self) -> 'ImageTrainerConfig':
        super().parse_args()
        if self.num_progress and self.progress_every_nepochs:
            self.error("specify only one of --num_progress and --progress_every_nepochs")
        
        if self.num_progress:
            self.num_progress = min(self.max_epochs, self.num_progress)
            self.progress_every_nepochs = self.max_epochs // self.num_progress
        
        return self

    def get_loggers(self) -> chain_logger.ChainLogger:
        dirname = self.log_dirname

        logger = chain_logger.ChainLogger()
        # logger.loggers.append(csv_logger.CsvLogger(Path("runs/experiments.csv"), runpath=Path(dirname)))
        logger.loggers.append(tb_logger.TensorboardLogger(dirname=dirname))

        if not self.no_checkpoints:
            skip_similar = True
            if self.do_resume:
                skip_similar = False
            cp_logger = \
                ckpt_logger.CheckpointLogger(dirname=dirname, 
                                             save_top_k=self.save_top_k, 
                                             skip_similar=skip_similar)
            logger.loggers.append(cp_logger)
        
        return logger



