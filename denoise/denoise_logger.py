# %%
import sys
import re
from typing import Deque, List, Dict, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
from IPython import display

import torch
from torch import Tensor

sys.path.append("..")
import trainer
import experiment
from experiment import Experiment
import model

class DenoiseLogger(trainer.TensorboardLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]
    top_k_metadatas: Deque[Path]
    noise_in: Tensor = None

    def __init__(self, basename: str, truth_is_noise: bool, save_top_k: int, max_epochs: int, device: str):
        super().__init__(f"denoise_{basename}_{max_epochs:04}")

        nrows = 3
        ncols = 5
        base_dim = 6
        plt.gcf().set_figwidth(base_dim * ncols)
        plt.gcf().set_figheight(base_dim * nrows)

        out_title = "output (noise)" if truth_is_noise else "output (denoised src)"
        noise_title = "truth (noise)" if truth_is_noise else "added noise"

        self.axes_in_noised = plt.subplot(nrows, ncols, 1, title="input (src + noise)")
        self.axes_out = plt.subplot(nrows, ncols, 2, title=out_title)
        self.axes_truth_noise = plt.subplot(nrows, ncols, 3, title=noise_title)
        if truth_is_noise:
            self.axes_in_sub_out = plt.subplot(nrows, ncols, 4, title="in - out (input w/o noise)")
        self.axes_src = plt.subplot(nrows, ncols, 5, title="truth (src)")

        self.axes_gen = {val: plt.subplot(nrows, ncols, 6 + i, title=f"{val} steps") 
                         for i, val in enumerate([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])}
        
        self.save_top_k = save_top_k
        self.device = device
        self.truth_is_noise = truth_is_noise

    def _status_path(self, exp: Experiment, subdir: str, epoch: int, suffix = "") -> str:
        path = Path(self.dirname, subdir or "", f"{exp.label},epoch_{epoch + 1:04}{suffix}")
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_metadatas = deque()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        similar_checkpoints = [(path, exp) for path, exp in find_all_checkpoints(Path("runs"))
                               if exp.label == exp.label]
        for ckpt_path, _exp in similar_checkpoints:
            # ckpt_path               = candidate .ckpt file
            # ckpt_path.parent        = "checkpoints" dir
            # ckpt_path.parent.parent = timestamped dir for that run
            status_path = Path(ckpt_path.parent.parent, exp.label + ".status")
            if status_path.exists():
                exp.skip = True
                break

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

            ckpt_path = self._status_path(exp, "checkpoints", epoch, ".ckpt")
            json_path = self._status_path(exp, "checkpoints", epoch, ".json")

            experiment.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            print(f"    saved {ckpt_path}/.json")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(ckpt_path)
                self.top_k_metadatas.append(json_path)
                if len(self.top_k_checkpoints) > self.save_top_k:
                    to_remove_ckpt = self.top_k_checkpoints.popleft()
                    to_remove_json = self.top_k_metadatas.popleft()
                    to_remove_ckpt.unlink()
                    to_remove_json.unlink()
                    print(f"  removed {to_remove_ckpt}/.json")

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        super().print_status(exp, epoch, batch, batches, train_loss)

        in_noised = exp.last_train_in[-1]
        out_noise = exp.last_train_out[-1]
        truth_noise = exp.last_train_truth[-1][0]
        src = exp.last_train_truth[-1][1]

        chan, width, height = in_noised.shape

        def transpose(img: Tensor) -> Tensor:
            img = img.clamp(min=0, max=1)
            img = img.detach().cpu()
            return torch.permute(img, (1, 2, 0))

        self.axes_in_noised.imshow(transpose(in_noised))
        self.axes_out.imshow(transpose(out_noise))
        self.axes_truth_noise.imshow(transpose(truth_noise))
        if self.truth_is_noise:
            self.axes_in_sub_out.imshow(transpose(in_noised - out_noise))
        self.axes_src.imshow(transpose(src))

        if self.noise_in is None:
            # use the same noise for all experiments & all epochs.
            self.noise_in = model.gen_noise((1, 3, width, width)).to(self.device) + 0.5

        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, truth_is_noise=self.truth_is_noise, input=self.noise_in, device=self.device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        if in_notebook():
            display.display(plt.gcf())
        img_path = self._status_path(exp, "images", epoch, ".png")
        plt.savefig(img_path)
        print(f"  saved PNG to {img_path}")


# conv_encdec2_k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_1,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.860M,epoch_0739,vloss_0.10699.ckpt
def find_all_checkpoints(runsdir: Path) -> List[Tuple[Path, Experiment]]:
    res: List[Tuple[Path, Experiment]] = list()
    for run_path in runsdir.iterdir():
        if not run_path.is_dir():
            continue
        checkpoints = Path(run_path, "checkpoints")
        if not checkpoints.exists():
            continue

        for ckpt_path in checkpoints.iterdir():
            if not ckpt_path.name.endswith(".ckpt"):
                continue
            meta_path = Path(str(ckpt_path)[:-5] + ".json")
            exp = experiment.load_experiment_metadata(meta_path)
            res.append((ckpt_path, exp))

    return res

if __name__ == "__main__":
    all_cp = find_all_checkpoints()
    print("all:")
    print("\n".join(map(str, all_cp)))
    print()

    label = "k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_3,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.828M"
    filter_cp = [cp for cp in all_cp if cp.label == label]
    print("matching:")
    print("\n".join(map(str, filter_cp)))
    print()
            
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

