import datetime
import sys
from collections import deque
from pathlib import Path
from typing import List, Set

sys.path.append("..")
import checkpoint_util
import trainer
from experiment import Experiment

class CheckpointLogger(trainer.TrainerLogger):
    save_top_k: int
    skip_similar: bool

    update_metadata_freq: datetime.timedelta
    update_metadata_at: datetime.datetime

    saved_epochs: Set[int]

    def __init__(self, *,
                 basename: str, runs_dir: Path = None,
                 started_at: datetime.datetime = None,
                 save_top_k: int, update_metadata_freq: int = 60):
        super().__init__(basename=basename, started_at=started_at, runs_dir=runs_dir)
        self.save_top_k = save_top_k
        self.update_metadata_freq = datetime.timedelta(seconds=update_metadata_freq)
        self.update_metadata_at = datetime.datetime.now() + self.update_metadata_freq

    def get_checkpoint_path(self, exp: Experiment, epoch: int) -> Path:
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        filename = f"epoch_{epoch:04}--{self.started_at_str}.ckpt"
        return Path(path, filename)
    
    def get_json_path(self, exp: Experiment):
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        return Path(path, "metadata.json")

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)
        self.saved_epochs = set()

    def on_exp_end(self, exp: Experiment):
        super().on_exp_end(exp)
        checkpoint_util.save_metadata(exp, self.get_json_path(exp))

    def update_val_loss(self, exp: Experiment):
        super().update_val_loss(exp)

        json_path = self.get_json_path(exp)
        ckpt_path = self.get_checkpoint_path(exp, exp.nepochs)

        topk_val = sorted(exp.val_loss_hist, key=lambda loss_tup: loss_tup[1])[:self.save_top_k]
        topk_val_epochs, topk_val_losses = zip(*topk_val)

        train_hist = zip(range(len(exp.train_loss_hist)), exp.train_loss_hist)
        topk_train = sorted(train_hist, key=lambda loss_tup: loss_tup[1])[:self.save_top_k]
        topk_train_epochs, topk_train_losses = zip(*topk_train)

        if topk_train_epochs[0] == exp.nepochs or topk_val_epochs[0] == exp.nepochs:
            # this epoch was better for train or val loss.
            start = datetime.datetime.now()
            checkpoint_util.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            self.saved_epochs.add(exp.nepochs)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()

            self.update_metadata_at = end + self.update_metadata_freq

            loss_strs: List[str] = list()
            if topk_train_epochs[0] == exp.nepochs:
                loss_strs.append(f"tloss {topk_train_losses[0]:.5f}")
            if topk_val_epochs[0] == exp.nepochs:
                loss_strs.append(f"vloss {topk_val_losses[0]:.5f}")
            print(f"    saved checkpoint {exp.nepochs + 1}: {', '.join(loss_strs)} in {elapsed:.2f}s")

            rm_epochs = self.saved_epochs - set(topk_train_epochs) - set(topk_val_epochs)
            for rm_epoch in sorted(rm_epochs):
                path = self.get_checkpoint_path(exp, rm_epoch)
                if path.exists():
                    path.unlink()
                    print(f"  removed checkpoint {rm_epoch + 1}")
            
        else:
            start = datetime.datetime.now()
            if start >= self.update_metadata_at:
                checkpoint_util.save_metadata(exp, json_path)
                end = datetime.datetime.now()
                elapsed = (end - start).total_seconds()
                print(f"  updated metadata in {elapsed:.2f}s")

                self.update_metadata_at = end + self.update_metadata_freq
