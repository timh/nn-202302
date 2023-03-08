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
from experiment import Experiment
import model

class DenoiseLogger(trainer.TensorboardLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]

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
        self.top_k_checkpoints = deque()
        self.device = device
        self.truth_is_noise = truth_is_noise

    def _status_path(self, exp: Experiment, subdir: str, epoch: int, suffix = "") -> str:
        path = Path(self.dirname, subdir or "", f"{exp.label},epoch_{epoch + 1:04}{suffix}")
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints.clear()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        for ckpt_path in find_similar_checkpoints(exp.label):
            # ckpt_path               = candidate .ckpt file
            # ckpt_path.parent        = "checkpoints" dir
            # ckpt_path.parent.parent = timestamped dir for that run
            status_file = Path(ckpt_path.path.parent.parent, f"{exp.label}.status")
            if status_file.exists():
                with open(status_file, "r") as file:
                    epochs_done = int(file.read().strip())
                if epochs_done == exp.max_epochs:
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
            state_dict = exp.state_dict()
            ckpt_path = self._status_path(exp, "checkpoints", epoch, ".ckpt")
            with open(ckpt_path, "wb") as torchfile:
                torch.save(state_dict, torchfile)
            print(f"    saved {ckpt_path}")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(ckpt_path)
                if len(self.top_k_checkpoints) > self.save_top_k:
                    to_remove = self.top_k_checkpoints.popleft()
                    print(f"  removed {to_remove}")
                    to_remove.unlink()

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

        noisein = model.gen_noise((1, 3, width, width)).to(self.device) + 0.5
        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, truth_is_noise=self.truth_is_noise, input=noisein, device=self.device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        if in_notebook():
            display.display(plt.gcf())
        img_path = self._status_path(exp, "images", epoch, ".png")
        plt.savefig(img_path)
        print(f"  saved PNG to {img_path}")


@dataclass
class CheckpointResult:
    path: Path
    label: str
    conv_descs: str
    fields: Dict[str, str]
    status: Dict[str, str]
    emblen: int = None
    nlin: int = None
    hidlen: int = None
    sched_type: str = None
    slr: float = None
    elr: float = None
    nparams: int = None
    epoch: int = None
    do_batch_norm: bool = None
    do_layer_norm: bool = None
    do_flatconv2d: bool = None

RE_CHECKPOINT = re.compile(r"(.*),(emblen.+),(epoch_.+)\.ckpt")

# conv_encdec2_k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_1,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.860M,epoch_0739,vloss_0.10699.ckpt
def find_all_checkpoints() -> List[CheckpointResult]:
    res: List[Path] = list()
    for run_path in Path("runs").iterdir():
        if not run_path.is_dir():
            continue
        checkpoints = Path(run_path, "checkpoints")
        if not checkpoints.exists():
            continue
        for ckpt_path in checkpoints.iterdir():
            match = RE_CHECKPOINT.match(ckpt_path.name)
            if not match:
                continue
            conv_descs, fields_str, status_str = match.groups()
            label = f"{conv_descs},{fields_str}"

            def str_to_dict(fullstr: str):
                pairs = list()
                for pairstr in fullstr.split(","):
                    if "_" in pairstr:
                        pairs.append(pairstr.split("_"))
                    else:
                        # TODO doesn't handle constant / nanogpt scheduler types.
                        # for bnorm, lnorm
                        pairs.append([pairstr, "True"])
                return {k: v for k, v in pairs}

            fields = str_to_dict(fields_str)
            status = str_to_dict(status_str)

            cpres = CheckpointResult(path=ckpt_path, label=label, conv_descs=conv_descs,
                                     fields=fields, status=status)
            for field in "emblen nlin hidlen slr elr nparams epoch constant nanogpt lnorm bnorm flatconv2d".split(" "):
                dict_to_look = status if field in ["epoch"] else fields
                if field in ["elr", "constant", "nanogpt", "lnorm", "bnorm", "flatconv2d"] and field not in dict_to_look:
                    continue
                if field in ["slr", "elr"]:
                    val = float(dict_to_look[field])
                elif field == "nparams":
                    val = float(dict_to_look[field][:-1]) * 1e6
                elif field in ["constant", "nanogpt"]:
                    val = field
                    field = "sched_type"
                elif field == "lnorm":
                    val = True
                    field = "do_layer_norm"
                elif field == "bnorm":
                    val = True
                    field = "do_batch_norm"
                elif field == "flatconv2d":
                    val = True
                    field = "do_flatconv2d"
                else:
                    val = int(dict_to_look[field])
                setattr(cpres, field, val)

            res.append(cpres)

    return res

def find_similar_checkpoints(label: str) -> List[CheckpointResult]:
    all_checkpoints = find_all_checkpoints()
    return [ckpt for ckpt in all_checkpoints if label in ckpt.label]

if __name__ == "__main__":
    res = find_all_checkpoints()
    print("all:")
    print("\n".join(map(str, res)))
    print()

    label = "k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_3,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.828M"
    res = find_similar_checkpoints(label)
    print("matching:")
    print("\n".join(map(str, res)))
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

