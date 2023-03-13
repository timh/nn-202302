import sys
from typing import List

sys.path.append("..")
import trainer
from experiment import Experiment

class ChainLogger(trainer.TrainerLogger):
    def __init__(self, *loggers: List[trainer.TrainerLogger]):
        self.loggers = loggers
    
    def on_exp_start(self, exp: Experiment):
        for logger in self.loggers:
            logger.on_exp_start(exp)

    def on_exp_end(self, exp: Experiment):
        for logger in self.loggers:
            logger.on_exp_end(exp)

    def on_batch(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_batch: float):
        super().on_batch(exp, epoch, batch, exp_batch, train_loss_batch)
        for logger in self.loggers:
            logger.on_batch(exp, epoch, batch, exp_batch, train_loss_batch)

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        for logger in self.loggers:
            logger.on_epoch_end(exp, epoch, train_loss_epoch)

    def print_status(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_epoch: float):
        for logger in self.loggers:
            logger.print_status(exp, epoch, batch, exp_batch, train_loss_epoch)

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        for logger in self.loggers:
            logger.update_val_loss(exp, epoch, val_loss)


