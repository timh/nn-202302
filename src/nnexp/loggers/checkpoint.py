import datetime
import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from nnexp import checkpoint_util as cputil
from nnexp.training import trainer
from nnexp.experiment import Experiment

@dataclass
class SavedLoss:
    epoch: int
    loss: float
    cp_path: Path

    def __lt__(self, other: 'SavedLoss') -> bool:
        return self.loss < other.loss

class CheckpointLogger(trainer.TrainerLogger):
    save_top_k: int
    skip_similar: bool

    update_metadata_freq: datetime.timedelta
    update_metadata_at: datetime.datetime

    update_checkpoint_freq: datetime.timedelta
    update_train_at: datetime.datetime
    update_val_at: datetime.datetime

    saved_train: List[SavedLoss]
    saved_val: List[SavedLoss]

    def __init__(self, *,
                 basename: str, runs_dir: Path = None,
                 started_at: datetime.datetime = None,
                 save_top_k: int, 
                 update_metadata_freq: int = 60,
                 update_checkpoint_freq: int = 60):
        super().__init__(basename=basename, started_at=started_at, runs_dir=runs_dir)
        now = datetime.datetime.now()

        self.save_top_k = save_top_k
        self.update_metadata_freq = datetime.timedelta(seconds=update_metadata_freq)
        self.update_metadata_at = now + self.update_metadata_freq

        self.update_checkpoint_freq = datetime.timedelta(seconds=update_checkpoint_freq)
        self.update_train_at = now + self.update_checkpoint_freq
        self.update_val_at = now + self.update_checkpoint_freq

        if save_top_k != 1:
            raise NotImplemented(f"K other than 1 is not implemented: {save_top_k=}")

    def get_checkpoint_path(self, exp: Experiment, epoch: int, loss_type: str) -> Path:
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        filename = f"epoch_{epoch:04}--{self.started_at_str}--{loss_type}.ckpt"
        return Path(path, filename)
    
    def get_json_path(self, exp: Experiment):
        path = super().get_exp_path("checkpoints", exp, mkdir=True)
        return Path(path, "metadata.json")

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.saved_train = list()
        self.saved_val = list()

    def on_epoch_end(self, exp: Experiment):
        super().on_epoch_end(exp)

        now = datetime.datetime.now()
        is_last_epoch = (exp.nepochs == exp.max_epochs - 1)

        md_path = self.get_json_path(exp)
        new_cp_path_train = self.get_checkpoint_path(exp, exp.nepochs, "tloss")
        new_cp_path_val = self.get_checkpoint_path(exp, exp.nepochs, "vloss")

        new_tloss = SavedLoss(epoch=exp.nepochs, loss=exp.train_loss_hist[-1], cp_path=new_cp_path_train)
        if len(exp.val_loss_hist):
            vepochs, vloss = exp.val_loss_hist[-1]
            new_vloss = SavedLoss(epoch=vepochs, loss=vloss, cp_path=new_cp_path_val)
        else:
            new_vloss = None

        def should_save(loss_type: str) -> Tuple[bool, SavedLoss]:
            result = False
            to_remove: SavedLoss = None
            if loss_type == 'tloss':
                if now < self.update_train_at and not is_last_epoch:
                    return False, None
                saved_list = self.saved_train
                new_loss = new_tloss
            else:
                if new_vloss is None:
                    return False, None
                if now < self.update_val_at and not is_last_epoch:
                    return False, None
                saved_list = self.saved_val
                new_loss = new_vloss

            if len(saved_list) < self.save_top_k or new_loss.loss < saved_list[-1].loss:
                result = True
                saved_list.append(new_loss)
                saved_list = sorted(saved_list)
                if loss_type == 'tloss':
                    self.saved_train = saved_list
                else:
                    self.saved_val = saved_list
            
            if len(saved_list) > self.save_top_k:
                to_remove = saved_list.pop()
            return result, to_remove
        
        do_save_train, to_remove_train = should_save('tloss')
        do_save_val, to_remove_val = should_save('vloss')

        if not any([do_save_train, do_save_val]):
            if now >= self.update_metadata_at or is_last_epoch:
                print(f"  saved metadata")
                cputil.save_metadata(exp, md_path)
                self.update_metadata_at = now + self.update_metadata_freq
            return

        # build up strings and paths for below.
        save_strs: List[str] = list()
        old_cp_path_train: Path = None
        if to_remove_train:
            save_strs.append(f"replaced epoch {to_remove_train.epoch} -> {exp.nepochs}: tloss {to_remove_train.loss:.5f} -> {new_tloss.loss:.5f}")
            old_cp_path_train = to_remove_train.cp_path
        elif do_save_train:
            save_strs.append(f"saved epoch {exp.nepochs}: tloss {new_tloss.loss:.5f}")

        old_cp_path_val: Path = None
        if to_remove_val:
            save_strs.append(f"replaced epoch {to_remove_val.epoch} -> {exp.nepochs}: vloss {to_remove_val.loss:.5f} -> {new_vloss.loss:.5f}")
            old_cp_path_val = to_remove_val.cp_path
        elif do_save_val:
            save_strs.append(f"saved epoch {exp.nepochs}: vloss {new_vloss.loss:.5f}")

        self.update_metadata_at = datetime.datetime.now() + self.update_metadata_freq

        # now do the work.
        if do_save_train:
            cputil.save_checkpoint(exp=exp, new_cp_path=new_cp_path_train, md_path=md_path, old_cp_path=old_cp_path_train)
            self.update_train_at = now + self.update_checkpoint_freq

        if do_save_val:
            cputil.save_checkpoint(exp=exp, new_cp_path=new_cp_path_val, md_path=md_path, old_cp_path=old_cp_path_val)
            self.update_val_at = now + self.update_checkpoint_freq

        print("  " + "\n  ".join(save_strs))
