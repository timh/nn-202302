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
import numpy as np

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

    def __init__(self, save_top_k: int):
        super().__init__("denoise")

        nrows = 3
        ncols = 4
        base_dim = 6
        plt.gcf().set_figwidth(base_dim * ncols)
        plt.gcf().set_figheight(base_dim * nrows)

        self.axes_input = plt.subplot(nrows, ncols, 1, title="input (src + noise)")
        self.axes_output = plt.subplot(nrows, ncols, 2, title="output (src - noise)")
        self.axes_src = plt.subplot(nrows, ncols, 3, title="truth (src)")

        self.axes_gen = {val: plt.subplot(nrows, ncols, 5 + i, title=f"{val} steps") 
                         for i, val in enumerate([1, 5, 10, 20, 40, 60, 100])}
        
        self.save_top_k = save_top_k
        self.top_k_checkpoints = deque()

    def _filename_base(self, exp: Experiment, epoch: int) -> str:
        filename = f"{self.dirname}--{exp.label},epoch_{epoch:04}"
        return filename
    
    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints.clear()

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
            filename = self._filename_base(exp, epoch) + f",vloss_{self.last_val_loss:.5f}.ckpt"
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

        input = exp.last_train_in[-1]
        src = exp.last_train_truth[-1]
        out = exp.last_train_out[-1]
        chan, width, height = input.shape

        def transpose(img: Tensor) -> Tensor:
            img = img.clamp(min=0, max=1)
            img = img.detach().cpu()
            return np.transpose(img, (1, 2, 0))

        self.axes_input.imshow(transpose(input))
        self.axes_output.imshow(transpose(out))
        self.axes_src.imshow(transpose(src))

        noisein = model.gen_noise((1, 3, width, width)).to(device) + 0.5
        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, input=noisein, device=device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        if in_notebook():
            display.display(plt.gcf())
        filename = self._filename_base(exp, epoch) + ".png"
        plt.savefig(filename)
        print(f"  saved PNG to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--epochs", type=int, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-I", "--image_size", default=128, type=int)
    parser.add_argument("-d", "--image_dir", default="alex-many-128")
    parser.add_argument("-k", "--save_top_k", default=1)

    if in_notebook():
        dev_args = "-c conf/conv_fancy1.py -n 20".split(" ")
        cfg = parser.parse_args(dev_args)
    else:
        cfg = parser.parse_args()

    device = "cuda"
    loss_fn = nn.MSELoss()
    torch.set_float32_matmul_precision('high')

    # eval the config file. the blank variables are what's assumed as "output"
    # from evaluating it.
    net: nn.Module = None
    batch_size: int = 128
    minicnt: int = 10
    exp_descs: List[Dict[str, any]] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())

    dataset = noised_data.load_dataset(image_dirname=cfg.image_dir, image_size=cfg.image_size)
    train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, minicnt=minicnt)

    exps: List[Experiment] = list()
    for ed in exp_descs:
        net = ed.get("net", None)
        if net is not None:
            net = net.to(device)
        net_fn = ed.get("net_fn", None)
        exp = Experiment(net=net, net_fn=net_fn,
                         loss_fn=loss_fn, epochs=cfg.epochs,
                         train_dataloader=train_dl, val_dataloader=val_dl, 
                         label=ed["label"] + f"--batch_{batch_size},minicnt_{minicnt}")
        exps.append(exp)
        
    if hasattr(torch, "compile"):
        print(f"compiling models...")
        for exp in exps:
            if exp.net is not None:
                exp.net = torch.compile(exp.net)

    for i, exp in enumerate(exps):
        print(f"#{i + 1} {exp.label}")
    print()

    tcfg = trainer.TrainerConfig(exps, len(exps), model.get_optim_fn)
    logger = Logger(save_top_k=cfg.save_top_k)
    t = trainer.Trainer(logger=logger, update_frequency=30)
    t.train(tcfg, device=device)

# %%


