# %%
import datetime
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Callable

import denoise_progress
import model
from denoise_exp import DNExperiment
from torch import Tensor

sys.path.append("..")
import loadsave
import trainer

class DenoiseLogger(trainer.TensorboardLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]
    top_k_jsons: Deque[Path]
    top_k_epochs: Deque[int]
    top_k_vloss: Deque[float]
    noise_in: Tensor = None

    progress_every_nepochs: int = 0
    _progress: denoise_progress.DenoiseProgress = None

    def __init__(self, basename: str, max_epochs: int, save_top_k: int, 
                 progress_every_nepochs: int, 
                 truth_is_noise: bool, use_timestep: bool, disable_noise: bool,
                 noise_fn: Callable[[Tuple], Tensor], amount_fn: Callable[[], Tensor],
                 device: str):
        super().__init__(f"denoise_{basename}_{max_epochs:04}")

        self.save_top_k = save_top_k
        self.truth_is_noise = truth_is_noise
        self.use_timestep = use_timestep
        self.disable_noise = disable_noise
        self.device = device

        self.progress_every_nepochs = progress_every_nepochs
        if progress_every_nepochs:
            self._progress = \
                denoise_progress.DenoiseProgress(truth_is_noise=self.truth_is_noise, 
                                                 use_timestep=self.use_timestep, 
                                                 disable_noise=self.disable_noise,
                                                 noise_fn=noise_fn, amount_fn=amount_fn,
                                                 device=self.device)

    def _status_path(self, exp: DNExperiment, subdir: str, epoch: int = 0, suffix = "") -> str:
        filename: List[str] = [exp.label]
        if epoch:
            filename.append(f"epoch_{epoch:04}")
        filename = ",".join(filename)
        path = Path(self.dirname, subdir or "", filename + suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def on_exp_start(self, exp: DNExperiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_jsons = deque()
        self.top_k_epochs = deque()
        self.top_k_vloss = deque()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        similar_checkpoints = [(path, exp) for path, exp in loadsave.find_checkpoints()
                               if exp.label == exp.label]
        for ckpt_path, _exp in similar_checkpoints:
            # ckpt_path               = candidate .ckpt file
            # ckpt_path.parent        = "checkpoints" dir
            # ckpt_path.parent.parent = timestamped dir for that run
            status_path = Path(ckpt_path.parent.parent, exp.label + ".status")
            if status_path.exists():
                exp.skip = True
                return
        
        if self.progress_every_nepochs:
            ncols = exp.max_epochs // self.progress_every_nepochs
            self._progress_path = Path(self._status_path(exp, "images", suffix="-progress.png"))
            self._progress.on_exp_start(exp=exp, ncols=ncols, path=self._progress_path)

    def on_exp_end(self, exp: DNExperiment):
        super().on_exp_end(exp)

        if not exp.skip:
            path = Path(self.dirname, f"{exp.label}.status")
            with open(path, "w") as file:
                file.write(str(exp.nepochs))

    def on_epoch_end(self, exp: DNExperiment, epoch: int, train_loss_epoch: float):
        super().on_epoch_end(exp, epoch, train_loss_epoch)

        if self.progress_every_nepochs and (epoch + 1) % self.progress_every_nepochs == 0:
            col = epoch // self.progress_every_nepochs
            # NOTE: this uses random images and noise each column.
            self._progress.add_column(exp=exp, epoch=epoch, col=col)

            symlink_path = Path("runs", "last-progress.png")
            symlink_path.unlink(missing_ok=True)
            symlink_path.symlink_to(self._progress_path.absolute())

    def update_val_loss(self, exp: DNExperiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss

            ckpt_path = self._status_path(exp, "checkpoints", epoch + 1, ".ckpt")
            json_path = self._status_path(exp, "checkpoints", epoch + 1, ".json")

            start = datetime.datetime.now()
            loadsave.save_ckpt_and_metadata(exp, ckpt_path, json_path)
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

def in_notebook():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

