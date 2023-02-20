# %%
import io
import sys
import os
import datetime
from typing import List, Tuple, Literal
import importlib

import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython import display
from PIL import Image

import data
import model
from model import Config, gen_examples, train
sys.path.insert(0, "..")
import notebook

importlib.reload(model)
importlib.reload(notebook)

# %%
cfg = Config(data.compounds_str)
cfg.setup(num_hidden=3, hidden_size=len(cfg.all_mol_names), lr=1e-4)

timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_filename = ["outputs", timestr]
base_filename = "/".join(base_filename)

# hp_num_batches = 10
# hp_batch_size = 1000
hp_num_batches = 5
hp_batch_size = 1000
# hp_learning_rates = {1: 100, 1e-4: 1000, 5e-5: 1000, 3e-5: 5000, 1e-5: 5000, 3e-6: 5000, 2e-6: 5000, 1e-6: 5000}
hp_learning_rates = {1e-4: 1000, 5e-5: 3000, 2e-5: 10000, 1e-5: 10000, 5e-6: 10000, 2e-6: 10000, 1e-6: 10000}
# hp_num_hidden = [6, 8, 10]
# hp_hidden_size = [72, 81, 90, 99]
hp_num_hidden = [6, 7]
hp_hidden_size = [85, 90, 95]

# TODO - only for fast debugging
# hp_learning_rates = {k: v//100 for k, v in hp_learning_rates.items()}
# hp_learning_rates = {k: v//10 for k, v in hp_learning_rates.items()}

data_train = gen_examples(cfg, hp_num_batches, hp_batch_size)
data_val = gen_examples(cfg, hp_num_batches, hp_batch_size)

total_epochs = sum(hp_learning_rates.values())
total_configs = len(hp_num_hidden) * len(hp_hidden_size)

labels_loss = []
labels_dist = []
for num_hidden in hp_num_hidden:
    for hidden_size in hp_hidden_size:
        labels_loss.append(f"val loss (num {num_hidden}, size {hidden_size})")
        labels_dist.append(f"val dist (num {num_hidden}, size {hidden_size})")
labels_loss.append("learning rate")
labels_dist.append("learning rate")

fig = plt.figure(0, (30, 16))
plot_loss = notebook.Plot(total_epochs, labels_loss, fig, 2, 1, 1)
plot_dist = notebook.Plot(total_epochs, labels_dist, fig, 2, 1, 2)
int_width = 20
intfig: Figure = None

config_idx = 0
for num_hidden in hp_num_hidden:
    for hidden_size in hp_hidden_size:
        cfg.setup(num_hidden=num_hidden, hidden_size=hidden_size, lr=1e-4)

        print()
        print(f"num_hidden {num_hidden}, hidden_size {hidden_size}")

        epochs_at_cfg = 0
        cfg_epochs = sum(hp_learning_rates.values())

        for lridx, (lr, epochs) in enumerate(hp_learning_rates.items()):
            print(f"lr = {lr} (num_hidden {num_hidden}, hidden_size {hidden_size})")
            cfg.optim = torch.optim.AdamW(cfg.net.parameters(), lr=lr)

            tlosses, vlosses, vdists = train(cfg, epochs, data_train, data_val)
            if len(tlosses) == 0:
                print(f"got {len(tlosses)=}, continuing")
                continue

            plot_loss.add_data(config_idx, vlosses)
            plot_dist.add_data(config_idx, vdists)
            if config_idx == 0:
                lr_tensor = torch.ones((len(vlosses),)) * lr
                plot_loss.add_data(len(plot_loss.labels) - 1, lr_tensor)
                plot_dist.add_data(len(plot_dist.labels) - 1, lr_tensor)

            smooth_steps = 100
            show_quantile = 0.6
            fig.tight_layout()
            plot_loss.render(show_quantile, smooth_steps, True)
            plot_dist.render(show_quantile, smooth_steps, True)

            fig.tight_layout()
            # display.clear_output(True)
            display.display(fig)

            epochs_at_cfg += epochs

            # save image & net
            torch_filename = f"{base_filename}-num_hidden_{num_hidden}-hidden_size_{hidden_size}.torch"
            torch.save(cfg.net, torch_filename)
            print(f"saved {torch_filename}")

            img_filename = f"{base_filename}_{config_idx}.png"
            fig.savefig(img_filename)
            print(f"saved {img_filename}")

            # save weights & gradients
            nrows = (total_configs * 4)
            fn_figure = lambda rows, cols: plt.figure(1, (int_width, ((rows * int_width / cols) * 4 + 1) * total_configs))
            titlebase = f"num hidden {num_hidden}, hidden size {hidden_size}, {epochs}@lr {lr:E}"

            # low gradients
            grad_threshold = 0.0001
            grad_title = f"grad < {grad_threshold} for {titlebase}"
            grad_fn_tensor = lambda lmod: lmod.weight.grad < grad_threshold
            intfig, intaxes1 = notebook.imshow(cfg.net, grad_fn_tensor, 
                                                fig=intfig, fn_fig=fn_figure, nrows=nrows, row=(config_idx*4) + 1,
                                                title=grad_title)

            # low weights
            weight_threshold = 0.1
            weight_title = f"weights < {weight_threshold} for {titlebase}"
            weight_fn_tensor = lambda lmod: lmod.weight < weight_threshold
            intfig, intaxes2 = notebook.imshow(cfg.net, weight_fn_tensor,
                                                fig=intfig, fn_fig=fn_figure, nrows=nrows, row=(config_idx*4) + 2,
                                                title=weight_title)

            # high weights
            weight_threshold = 1.0
            weight_title = f"weights > {weight_threshold} for {titlebase}"
            weight_fn_tensor = lambda lmod: lmod.weight > weight_threshold
            intfig, intaxes3 = notebook.imshow(cfg.net, weight_fn_tensor,
                                                fig=intfig, fn_fig=fn_figure, nrows=nrows, row=(config_idx*4) + 3,
                                                title=weight_title)

            display.display(intfig)
            mat_filename = f"{base_filename}_{config_idx}_num_hidden_{num_hidden}-hidden_size_{hidden_size}--{lridx:02}-{epochs}@lr{lr:.1E}.png"
            intfig.tight_layout()
            intfig.savefig(mat_filename)
            print(f"saved {mat_filename}")

        config_idx += 1


# %%
