import sys
from typing import List

sys.path.append("..")
import trainer
from experiment import Experiment

class ChainLogger(trainer.TrainerLogger):
    def __init__(self, *loggers: List[trainer.TrainerLogger]):
        self.loggers = list(loggers)
    
    def on_exp_start(self, exp: Experiment):
        for logger in self.loggers:
            logger.on_exp_start(exp)

    def on_exp_end(self, exp: Experiment):
        for logger in self.loggers:
            logger.on_exp_end(exp)

    def on_batch(self, exp: Experiment, batch: int, batch_size: int, train_loss_batch: float):
        super().on_batch(exp, batch, batch_size, train_loss_batch)
        for logger in self.loggers:
            logger.on_batch(exp, batch, batch_size, train_loss_batch)

    def on_epoch_end(self, exp: Experiment):
        for logger in self.loggers:
            logger.on_epoch_end(exp)

    def print_status(self, exp: Experiment, batch: int, batch_size: int, train_loss_epoch: float):
        for logger in self.loggers:
            logger.print_status(exp, batch, batch_size, train_loss_epoch)

    def update_val_loss(self, exp: Experiment):
        for logger in self.loggers:
            logger.update_val_loss(exp)


