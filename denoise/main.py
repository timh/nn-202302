# %%
import sys
import importlib
import argparse
from typing import List, Dict, Deque
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from IPython import display

import torch
from torch import nn, Tensor

sys.path.append("..")
import trainer
from experiment import Experiment
import noised_data
import model

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

for m in [trainer, noised_data, model]:
    importlib.reload(m)

class Logger(trainer.TensorboardLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]

    def __init__(self, basename: str, save_top_k: int, epochs: int):
        super().__init__(f"denoise_{basename}_{epochs:04}")

        nrows = 3
        ncols = 5
        base_dim = 6
        plt.gcf().set_figwidth(base_dim * ncols)
        plt.gcf().set_figheight(base_dim * nrows)

        # input (noised src)
        # output (noise)
        # truth (noise)
        # in - out (derived denoised src)
        # src

        self.axes_in_noised = plt.subplot(nrows, ncols, 1, title="input (src + noise)")
        self.axes_out_noise = plt.subplot(nrows, ncols, 2, title="output (noise)")
        self.axes_truth_noise = plt.subplot(nrows, ncols, 3, title="truth (noise)")
        self.axes_in_sub_out = plt.subplot(nrows, ncols, 4, title="in - out (input w/o noise)")
        self.axes_src = plt.subplot(nrows, ncols, 5, title="truth (src)")

        self.axes_gen = {val: plt.subplot(nrows, ncols, 6 + i, title=f"{val} steps") 
                         for i, val in enumerate([1, 2, 5, 10, 20, 40, 60, 100])}
        
        self.save_top_k = save_top_k
        self.top_k_checkpoints = deque()

    def _filename_base(self, exp: Experiment, subdir: str, epoch: int) -> str:
        filename = f"{self.dirname}/{subdir}/{exp.label},epoch_{epoch:04}"
        Path(filename).parent.mkdir(exist_ok=True)
        return filename
    
    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints.clear()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        for path in Path("runs").iterdir():
            if not path.name.endswith(".ckpt"):
                continue
            if self.basename in path.name and exp.label in path.name:
                exp.skip = True

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss
            state_dict = exp.net.state_dict()
            filename = self._filename_base(exp, "checkpoints", epoch) + f",vloss_{self.last_val_loss:.5f}.ckpt"
            with open(filename, "wb") as torchfile:
                torch.save(state_dict, torchfile)
            print(f"    saved {filename}")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(Path(filename))
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
        self.axes_out_noise.imshow(transpose(out_noise))
        self.axes_truth_noise.imshow(transpose(truth_noise))
        self.axes_in_sub_out.imshow(transpose(in_noised - out_noise))
        self.axes_src.imshow(transpose(src))

        noisein = model.gen_noise((1, 3, width, width)).to(device) + 0.5
        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, input=noisein, device=device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        if in_notebook():
            display.display(plt.gcf())
        filename = self._filename_base(exp, "images", epoch) + ".png"
        plt.savefig(filename)
        print(f"  saved PNG to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--epochs", type=int, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-I", "--image_size", default=128, type=int)
    parser.add_argument("-d", "--image_dir", default="alex-many-128")
    parser.add_argument("-k", "--save_top_k", default=1)
    parser.add_argument("--startlr", type=float, default=1e-3)
    parser.add_argument("--endlr", type=float, default=1e-4)

    if in_notebook():
        dev_args = "-c conf/conv_denoise1.py -n 100".split(" ")
        cfg = parser.parse_args(dev_args)
    else:
        cfg = parser.parse_args()

    device = "cuda"
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    loss_fn = noised_data.twotruth_loss_fn
    torch.set_float32_matmul_precision('high')

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    net: nn.Module = None
    batch_size: int = 128
    minicnt: int = 10
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())

    dataset = noised_data.load_dataset(image_dirname=cfg.image_dir, image_size=cfg.image_size)
    train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, minicnt=minicnt)

    for exp in exps:
        exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
        exp.lazy_optim_fn = trainer.lazy_optim_fn
        exp.lazy_sched_fn = trainer.lazy_sched_fn
        exp.device = device
        if exp.loss_fn is None:
            exp.loss_fn = loss_fn
        if exp.startlr is None:
            exp.startlr = cfg.startlr
        if exp.endlr is None:
            exp.endlr = cfg.endlr
        if not exp.epochs:
            exp.epochs = cfg.epochs
        exp.label += f",slr_{exp.startlr:.1E}"
        exp.label += f",elr_{exp.endlr:.1E}"

    for i, exp in enumerate(exps):
        print(f"#{i + 1} {exp.label}")
    print()

    basename = Path(cfg.config_file).stem

    logger = Logger(basename=basename, save_top_k=cfg.save_top_k, epochs=cfg.epochs)
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, update_frequency=30)
    t.train(device=device)

# %%


