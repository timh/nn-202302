import datetime
import sys
import torch.utils.tensorboard as tboard

sys.path.append("..")
from trainer import TrainerLogger
from experiment import Experiment

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter = None
    basename: str
    dirname: str

    def __init__(self, dirname: str):
        super().__init__(dirname)

    def _ensure_writer(self):
        if self.writer is None:
            self.writer = tboard.SummaryWriter(log_dir=self.dirname)

    def on_batch(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_batch: float):
        self._ensure_writer()
        self.writer.add_scalars("batch/tloss", {exp.label: train_loss_batch}, global_step=exp_batch)
        self.writer.add_scalars("batch/lr", {exp.label: exp.cur_lr}, global_step=exp_batch)

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        self._ensure_writer()
        self.writer.add_scalars("epoch/tloss", {exp.label: train_loss_epoch}, global_step=epoch)
        self.writer.add_scalars("epoch/lr", {exp.label: exp.cur_lr}, global_step=epoch)
    
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        self._ensure_writer()
        self.writer.add_scalars("epoch/vloss", {exp.label: val_loss}, global_step=epoch)


