import datetime
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List

sys.path.append("..")
import checkpoint_util
import trainer
from experiment import Experiment

class CheckpointLogger(trainer.TrainerLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]
    top_k_epochs: Deque[int]
    top_k_vloss: Deque[float]
    skip_similar: bool

    def __init__(self, *,
                 basename: str, started_at: datetime.datetime = None,
                 save_top_k: int, skip_similar: bool = True):
        super().__init__(basename=basename, started_at=started_at)
        self.save_top_k = save_top_k
        self.skip_simiilar = skip_similar

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_epochs = deque()
        self.top_k_vloss = deque()

        if not self.skip_simiilar:
            return

        similar_exps = [cp_exp
                        for _cp_path, cp_exp in checkpoint_util.find_checkpoints()
                        if exp.shortcode == cp_exp.shortcode]
        for exp in similar_exps:
            if exp.cur_run().finished:
                exp.skip = True

    def get_checkpoint_path(self, exp: Experiment, epoch: int) -> Path:
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        filename = f"epoch_{epoch:04}--{self.started_at_str}.ckpt"
        return Path(path, filename)
    
    def get_json_path(self, exp: Experiment):
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        filename = f"metadata.json"
        return Path(path, filename)

    def on_exp_end(self, exp: Experiment):
        super().on_exp_end(exp)

        checkpoint_util.save_metadata(exp, self.get_json_path(exp))

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)

        json_path = self.get_json_path(exp)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss

            start = datetime.datetime.now()
            ckpt_path = self.get_checkpoint_path(exp, epoch)
            checkpoint_util.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            print(f"    saved checkpoint {epoch + 1}: vloss {val_loss:.5f} in {elapsed:.2f}s: {ckpt_path}")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(ckpt_path)
                self.top_k_epochs.append(epoch)
                self.top_k_vloss.append(val_loss)
                if len(self.top_k_checkpoints) > self.save_top_k:
                    to_remove_ckpt = self.top_k_checkpoints.popleft()
                    removed_epoch = self.top_k_epochs.popleft()
                    removed_vloss = self.top_k_vloss.popleft()
                    to_remove_ckpt.unlink()
                    print(f"  removed checkpoint {removed_epoch + 1}: vloss {removed_vloss:.5f}")
        else:
            checkpoint_util.save_metadata(exp, json_path)

