import datetime
import sys
import torch.utils.tensorboard as tboard

sys.path.append("..")
from trainer import TrainerLogger
from experiment import Experiment

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter = None
    dirname: str = None

    def on_batch(self, exp: Experiment, epoch: int, batch: int, exp_batch: int, train_loss_batch: float):
        # self.writer.add_scalars("batch/tloss", {self.get_exp_base(exp): train_loss_batch}, global_step=exp_batch)
        # self.writer.add_scalars("batch/lr", {self.get_exp_base(exp): exp.cur_lr}, global_step=exp_batch)
        self.writer.add_scalar("batch/tloss", train_loss_batch, global_step=exp_batch)
        self.writer.add_scalar("batch/lr", exp.cur_lr, global_step=exp_batch)

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        self.dirname = super().get_exp_path(subdir="tensorboard", exp=exp, mkdir=True)
        self.writer = tboard.SummaryWriter(log_dir=self.dirname)

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        # self.writer.add_scalars("epoch/tloss", {self.get_exp_base(exp): train_loss_epoch}, global_step=epoch)
        # self.writer.add_scalars("epoch/lr", {self.get_exp_base(exp): exp.cur_lr}, global_step=epoch)
        self.writer.add_scalar("epoch/tloss", train_loss_epoch, global_step=epoch)
        self.writer.add_scalar("epoch/lr", exp.cur_lr, global_step=epoch)
    
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        # self.writer.add_scalars("epoch/vloss", {self.get_exp_base(exp): val_loss}, global_step=epoch)
        self.writer.add_scalar("epoch/vloss", val_loss, global_step=epoch)


