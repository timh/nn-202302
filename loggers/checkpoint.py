import datetime
import sys
from collections import deque
from pathlib import Path
from typing import List, Set, Tuple

sys.path.append("..")
import checkpoint_util
import trainer
from experiment import Experiment

LossTuple = Tuple[int, float]
class CheckpointLogger(trainer.TrainerLogger):
    save_top_k: int
    skip_similar: bool

    update_metadata_freq: datetime.timedelta
    update_metadata_at: datetime.datetime

    saved_train: List[LossTuple]
    saved_val: List[LossTuple]

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
        self.saved_train = list()
        self.saved_val = list()

    def on_exp_end(self, exp: Experiment):
        super().on_exp_end(exp)
        self.update(exp)

    def update_val_loss(self, exp: Experiment):
        super().update_val_loss(exp)
        self.update(exp)

    def update(self, exp: Experiment):
        json_path = self.get_json_path(exp)
        ckpt_path = self.get_checkpoint_path(exp, exp.nepochs)

        def should_save(update_epoch: int, update_loss: float, 
                        saved_list: List[LossTuple]) -> bool:
            if len(saved_list) < self.save_top_k:
                return True
            
            worst_epoch, worst_loss = saved_list[-1]
            if worst_epoch != update_epoch and update_loss < worst_loss:
                return True
            return False
        
        def sort_saves(saved_list: List[LossTuple]):
            return sorted(saved_list, key=lambda losstup: losstup[1])
        
        def split_saves(saved_list: List[LossTuple]) -> Tuple[List[LossTuple], List[LossTuple]]:
            saved_list = sort_saves(saved_list)
            return saved_list[:self.save_top_k], saved_list[self.save_top_k:]
        
        train_epoch, train_loss = exp.nepochs, exp.train_loss_hist[-1]
        save_train = should_save(train_epoch, train_loss, self.saved_train)

        val_epoch, val_loss = exp.val_loss_hist[-1]
        save_val = should_save(val_epoch, val_loss, self.saved_val)
        
        if save_train or save_val:
            # this epoch was better for train or val loss.
            start = datetime.datetime.now()
            checkpoint_util.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()

            self.update_metadata_at = end + self.update_metadata_freq

            loss_strs: List[str] = list()
            if save_train:
                loss_strs.append(f"tloss {train_loss:.5f}")
                self.saved_train.append((train_epoch, train_loss))
            if save_val:
                loss_strs.append(f"vloss {val_loss:.5f}")
                self.saved_val.append((val_epoch, val_loss))
            print(f"    saved checkpoint {exp.nepochs + 1}: {', '.join(loss_strs)} in {elapsed:.2f}s")

            self.saved_train, rm_train = split_saves(self.saved_train)
            self.saved_val, rm_val = split_saves(self.saved_val)
            keep_train_epochs = set([losstup[0] for losstup in self.saved_train])
            keep_val_epochs = set([losstup[0] for losstup in self.saved_val])

            # print(f"{keep_train_epochs=}")
            # print(f"{keep_val_epochs=}")

            for rm_epoch, rm_loss in rm_train:
                path = self.get_checkpoint_path(exp, rm_epoch)
                if rm_epoch in keep_val_epochs or not path.exists():
                    continue
                path.unlink()
                print(f"  removed checkpoint {rm_epoch + 1}: tloss {rm_loss:.5f}")
            
            for rm_epoch, rm_loss in rm_val:
                path = self.get_checkpoint_path(exp, rm_epoch)
                if rm_epoch in keep_train_epochs or not path.exists():
                    continue
                path.unlink()
                print(f"  removed checkpoint {rm_epoch + 1}: vloss {rm_loss:.5f}")
            
        else:
            start = datetime.datetime.now()
            if start >= self.update_metadata_at:
                checkpoint_util.save_metadata(exp, json_path)
                end = datetime.datetime.now()
                elapsed = (end - start).total_seconds()
                print(f"  updated metadata in {elapsed:.2f}s")

                self.update_metadata_at = end + self.update_metadata_freq
