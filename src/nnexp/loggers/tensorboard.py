import datetime
import sys
import torch.utils.tensorboard as tboard

sys.path.append("..")
from nnexp.training.trainer import TrainerLogger
from nnexp.experiment import Experiment

class TensorboardLogger(TrainerLogger):
    writer: tboard.SummaryWriter = None
    dirname: str = None

    def on_batch(self, exp: Experiment, batch: int, batch_size: int, train_loss_batch: float):
        self.writer.add_scalar("batch/tloss", train_loss_batch, global_step=exp.nbatches)
        self.writer.add_scalar("batch/lr", exp.cur_lr, global_step=exp.nbatches)

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        dirname = super().get_exp_path(subdir="tensorboard", exp=exp, mkdir=True)
        self.writer = tboard.SummaryWriter(log_dir=dirname)

    def on_epoch_end(self, exp: Experiment):
        self.writer.add_scalar("epoch/tloss", exp.last_train_loss, global_step=exp.nepochs)
        self.writer.add_scalar("epoch/lr", exp.cur_lr, global_step=exp.nepochs)

        # look for other .*loss.*_hist lists on the experiment, and plot them too
        for field in dir(exp):
            if field in {'val_loss_hist', 'train_loss_hist'}:
                continue

            if field.endswith("_hist") and "loss" in field:
                # print(f"tb {field=}")
                val = getattr(exp, field, None)
                if isinstance(val, list) and len(val):
                    name = field[:-5]
                    self.writer.add_scalar(f"epoch/{name}", val[-1], global_step=exp.nepochs)
    
    def update_val_loss(self, exp: Experiment):
        self.writer.add_scalar("epoch/vloss", exp.last_val_loss, global_step=exp.nepochs)
