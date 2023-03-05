# %%
import sys
import importlib
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
from IPython import display

import torch
from torch import nn
import numpy as np

sys.path.append("..")
import trainer
from experiment import Experiment
import noised_data
import model

for m in [trainer, noised_data, model]:
    importlib.reload(m)

class Logger(trainer.TensorboardLogger):
    def __init__(self):
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
        self.last_val_loss = None

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            filename = f"{self.dirname}-{exp.label}-epoch_{epoch:04}-vloss_{val_loss:.5f}.torch"
            state_dict = exp.net.state_dict()
            with open(filename, "wb") as torchfile:
                torch.save(state_dict, torchfile)
            print(f"  saved to {filename}")
            self.last_val_loss = val_loss

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        super().print_status(exp, epoch, batch, batches, train_loss)

        input = exp.last_train_in[-1]
        src = exp.last_train_truth[-1]
        out = exp.last_train_out[-1]
        chan, width, height = input.shape

        transpose = lambda t: np.transpose(t.detach().cpu(), (1, 2, 0))

        self.axes_input.imshow(transpose(input))
        self.axes_output.imshow(transpose(out))
        self.axes_src.imshow(transpose(src))

        noisein = model.gen_noise((1, 3, width, width)).to(device) + 0.5
        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, input=noisein, device=device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        display.display(plt.gcf())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--epochs", type=int, required=True)
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-I", "--image_size", default=128, type=int)
    parser.add_argument("-d", "--image_dir", default="alex-many-128")
    cfg = parser.parse_args()

    device = "cuda"
    loss_fn = nn.MSELoss()

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

    exps = [
        Experiment(net=ed["net"].to(device), 
                   loss_fn=loss_fn, epochs=cfg.epochs,
                   train_dataloader=train_dl, val_dataloader=val_dl, 
                   label=ed["label"] + f"--batch_{batch_size}-minicnt_{minicnt}")
        for ed in exp_descs
    ]
    for i, exp in enumerate(exps):
        print(f"#{i} {exp.label} =\n", exp.net)

    tcfg = trainer.TrainerConfig(exps, len(exps), model.get_optim_fn)
    t = trainer.Trainer(logger=Logger())
    t.train(tcfg, device=device)

# %%


