import sys
import wandb

sys.path.append("..")
from trainer import TrainerLogger
from experiment import Experiment

class WandbLogger(TrainerLogger):
    def on_batch(self, exp: Experiment, batch: int, batch_size: int, train_loss_batch: float):
        wandb.log({"batch/tloss": train_loss_batch}, step=exp.nbatches)

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        # NOTE: bug in WandB? my misunderstanding? if I use resume='allow' and this code writes data for a
        # step (??) that's already been written, the data in WandB for this ID stops updating. 
        # use resume=None to overwrite the old data.
        # wandb.init(project=self.basename, id=f"{exp.created_at_short}-{exp.shortcode}", resume='allow', config=exp.id_values(), reinit=True)
        # wandb.init(project=self.basename, id=f"{exp.created_at_short}-{exp.shortcode}", resume=None, config=exp.id_values(), reinit=True)

        # NOTE: that didn't work either. just using name and no id.
        wandb.init(project=self.basename, name=f"{exp.created_at_short}-{exp.shortcode}", resume=None, config=exp.id_values(), reinit=True)
        wandb.watch(exp.net, log='all')

    def on_epoch_end(self, exp: Experiment):
        wandb.log({"epoch": exp.nepochs, "epoch/tloss": exp.last_train_loss, "epoch/lr": exp.cur_lr}, step=exp.nbatches)

        # # look for other .*loss.*_hist lists on the experiment, and plot them too
        # for field in dir(exp):
        #     if field in {'val_loss_hist', 'train_loss_hist'}:
        #         continue

        #     if field.endswith("_hist") and "loss" in field:
        #         # print(f"tb {field=}")
        #         val = getattr(exp, field, None)
        #         if isinstance(val, list) and len(val):
        #             name = field[:-5]
        #             self.writer.add_scalar(f"epoch/{name}", val[-1], global_step=exp.nepochs)
    
    def update_val_loss(self, exp: Experiment):
        # self.writer.add_scalar("epoch/vloss", exp.last_val_loss, global_step=exp.nepochs)
        wandb.log({"epoch": exp.nepochs, "epoch/vloss": exp.last_val_loss}, step=exp.nbatches)
