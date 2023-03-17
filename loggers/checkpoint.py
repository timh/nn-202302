# %%
import datetime
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List

sys.path.append("..")
import model_util
import trainer
from experiment import Experiment

class CheckpointLogger(trainer.TrainerLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]
    top_k_jsons: Deque[Path]
    top_k_epochs: Deque[int]
    top_k_vloss: Deque[float]

    def __init__(self, dirname: str, save_top_k: int):
        super().__init__(dirname)
        self.save_top_k = save_top_k

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_jsons = deque()
        self.top_k_epochs = deque()
        self.top_k_vloss = deque()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        similar_checkpoints = [(cp_path, cp_exp) for cp_path, cp_exp in model_util.find_checkpoints()
                               if exp.is_same(cp_exp)]
        for ckpt_path, _exp in similar_checkpoints:
            # ckpt_path               = candidate .ckpt file
            # ckpt_path.parent        = "checkpoints" dir
            # ckpt_path.parent.parent = timestamped dir for that run
            status_path = Path(ckpt_path.parent.parent, exp.label + ".status")
            if status_path.exists():
                exp.skip = True
                return
        
    def on_exp_end(self, exp: Experiment):
        super().on_exp_end(exp)
        if not exp.skip:
            path = Path(self.dirname, f"{exp.label}.status")
            with open(path, "w") as file:
                file.write(str(exp.nepochs))

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss

            ckpt_path = self._status_path(exp, "checkpoints", epoch + 1, ".ckpt")
            json_path = self._status_path(exp, "checkpoints", epoch + 1, ".json")

            start = datetime.datetime.now()
            model_util.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            print(f"    saved checkpoint {epoch + 1}: vloss {val_loss:.5f} in {elapsed:.2f}s: {ckpt_path}")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(ckpt_path)
                self.top_k_jsons.append(json_path)
                self.top_k_epochs.append(epoch)
                self.top_k_vloss.append(val_loss)
                if len(self.top_k_checkpoints) > self.save_top_k:
                    to_remove_ckpt = self.top_k_checkpoints.popleft()
                    to_remove_json = self.top_k_jsons.popleft()
                    removed_epoch = self.top_k_epochs.popleft()
                    removed_vloss = self.top_k_vloss.popleft()
                    to_remove_ckpt.unlink()
                    to_remove_json.unlink()
                    print(f"  removed checkpoint {removed_epoch + 1}: vloss {removed_vloss:.5f}")
