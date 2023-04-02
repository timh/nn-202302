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

        if save_top_k != 1:
            raise NotImplemented(f"K other than 1 is not implemented: {save_top_k=}")

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

    def on_epoch_end(self, exp: Experiment, train_loss_epoch: float):
        super().on_epoch_end(exp, train_loss_epoch=train_loss_epoch)
        self.update(exp)

    def update_val_loss(self, exp: Experiment):
        super().update_val_loss(exp)
        self.update(exp)

    def update(self, exp: Experiment):
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
        
        new_train_epoch, new_train_loss = exp.nepochs, exp.train_loss_hist[-1]
        do_save_train = should_save(new_train_epoch, new_train_loss, self.saved_train)

        new_val_epoch, new_val_loss = exp.val_loss_hist[-1]
        do_save_val = should_save(new_val_epoch, new_val_loss, self.saved_val)

        md_path = self.get_json_path(exp)
        new_cp_path = self.get_checkpoint_path(exp, exp.nepochs)

        if do_save_train or do_save_val:
            # this epoch was better for train or val loss.
            self.update_metadata_at = datetime.datetime.now() + self.update_metadata_freq

            if do_save_train:
                self.saved_train.append((new_train_epoch, new_train_loss))
            if do_save_val:
                self.saved_val.append((new_val_epoch, new_val_loss))
            self.saved_train, replace_train = split_saves(self.saved_train)
            self.saved_val, replace_val = split_saves(self.saved_val)

            saved_tepochs = set([losstup[0] for losstup in self.saved_train])
            saved_vepochs = set([losstup[0] for losstup in self.saved_val])

            # filter out the removals for any checkpoints that are still to be
            # kept for the other loss type.
            toreplace_train = [(epoch, loss) for epoch, loss in replace_train if epoch not in saved_vepochs]
            toreplace_val = [(epoch, loss) for epoch, loss in replace_val if epoch not in saved_tepochs]

            if toreplace_train or toreplace_val:
                for toreplace, new_loss, loss_type in [(toreplace_train, new_train_loss, "tloss"), (toreplace_val, new_val_loss, "vloss")]:
                    for replace_epoch, replace_loss in toreplace:
                        old_cp_path = self.get_checkpoint_path(exp, replace_epoch)
                        if not old_cp_path.exists():
                            continue

                        checkpoint_util.save_checkpoint(exp=exp, old_cp_path=old_cp_path, new_cp_path=new_cp_path, md_path=md_path)
                        print(f"  replaced checkpoint {replace_epoch + 1} -> {exp.nepochs + 1}: {loss_type} {replace_loss:.5f} -> {new_loss:.5f}")
            else:
                checkpoint_util.save_checkpoint(exp=exp, new_cp_path=new_cp_path, md_path=md_path)
                print(f"     saved checkpoint {exp.nepochs + 1}: tloss {new_train_loss:.5f}, vloss {new_val_loss:.5f}")

        else:
            start = datetime.datetime.now()
            if start >= self.update_metadata_at:
                checkpoint_util.save_metadata(exp, md_path)
                end = datetime.datetime.now()
                elapsed = (end - start).total_seconds()
                print(f"  updated metadata in {elapsed:.2f}s")

                self.update_metadata_at = end + self.update_metadata_freq
