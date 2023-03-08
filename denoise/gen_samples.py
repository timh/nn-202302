# %%
import sys
from pathlib import Path
from typing import List, Union, Literal
import re
import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from PIL import Image
from IPython import display
import tqdm

import torch
from torch import Tensor

sys.path.append("..")
import model
from experiment import Experiment
import denoise_logger

device = "cuda"

if __name__ == "__main__":
    checkpoints = denoise_logger.find_all_checkpoints()
    checkpoints = [cpres for cpres in checkpoints if "c32,c64,c64" in cpres.conv_descs]
    nrows = len(checkpoints)

    mode: Literal["latent", "steps"] = "latent"

    if mode == "steps":
        steps_list = [2, 5, 10, 20, 30, 50, 70, 100]
        col_labels = [f"{s} steps" for s in steps_list]
        ncols = len(steps_list) + 1
        filename = "gen-steps.png"
    else:
        num_steps = 20
        num_latents = 10
        col_labels = [f"latent {i}" for i in range(num_latents)]
        ncols = num_latents + 1
        filename = "gen-latent.png"


    base = 2.5
    fig = plt.figure(1, figsize=(base * ncols, base * nrows))
    # plt.axis('off')
    axes_list = fig.subplots(nrows, ncols)

    for axes, label in zip(axes_list[0][1:], col_labels):
        axes.set_title(label)

    checkpoints = sorted(checkpoints, key=lambda c: c.label)

    inputs: Tensor = None
    for cidx, cp in tqdm.tqdm(list(enumerate(checkpoints))):
        # print(f"{cidx + 1}/{len(checkpoints)}")
        with open(cp.path, "rb") as checkpoint_file:
            state_dict = torch.load(checkpoint_file)
            net_class = state_dict["net_class"]

        tloss = state_dict["last_train_loss"]
        vloss = state_dict["last_val_loss"]
        elr_str = f"elr {cp.elr:.1E}\n" if cp.elr else ""
        title = (f"{cp.conv_descs}\n"
                 f"emblen {cp.emblen}\n"
                 f"nlin {cp.nlin}\n"
                 f"hidlen {cp.hidlen}\n"
                 f"slr {cp.slr:.1E}\n"
                 f"{elr_str}"
                 f"epoch {cp.epoch}\n"
                 f"tloss {tloss:.3f}\n"
                 f"vloss {vloss:.3f}")

        axes: plt.Axes = axes_list[cidx][0]
        axes.text(0.5, 0.5, title, horizontalalignment='center', verticalalignment='center')

        axes.axis('off')

        exp = Experiment.new_from_state_dict(state_dict)
        exp.net: model.ConvEncDec = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to(device)
        image_size = exp.net.image_size
        nchannels = exp.net.nchannels

        if inputs is None:
            if mode == "latent":
                # gaussian distribution for latent space.
                inputs = torch.normal(0.0, 0.5, (num_latents, 1, cp.emblen), device=device)
            else:
                # uniform distribution for pixel space.
                inputs = torch.rand((1, nchannels, image_size, image_size), device=device)

        for col in range(ncols - 1):
            if mode == "latent":
                steps = num_steps
                gen_inputs = inputs[col]
                out = exp.net.decoder(inputs[col])
            else:
                steps = steps_list[col]
                gen_inputs = inputs
                out = model.generate(exp=exp, num_steps=steps, size=image_size, truth_is_noise=False, input=inputs, device=device)

            out = out[0]
            out = torch.permute(out, (1, 2, 0))   # (chan, width, height) -> (width, height, chan)
            out = out.detach().cpu()

            axes: plt.Axes = axes_list[cidx][col + 1]
            axes.axis('off')
            # axes.xaxis.set_major_locator(NullLocator())
            # axes.yaxis.set_major_locator(NullLocator())
            axes.imshow(out)
    
    fig.tight_layout()
    display.display(fig)
    fig.savefig(filename)
    fig.clear();
