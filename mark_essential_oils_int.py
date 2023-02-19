# %%
compounds_str = """
part1-CO2_50-FeO_50:
CO2: 50%
FeO: 50%

part2-CO_20-Fe_50-AgO_20-AlO_10:
CO: 20%
Fe: 50%
AgO: 20%
AlO: 10%

part3-HCO2_15-FeO2_60-Ag_25:
HCO2: 15%
FeO2: 60%
Ag: 25%

part4-HCO2_20-FeO2_60-Ag_20:
HCO2: 20%
FeO2: 60%
Ag: 20%
"""
import io
import datetime
from typing import List, Tuple, Literal
import importlib

import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython import display
from PIL import Image

import mark_essential_oils
from mark_essential_oils import Config, gen_examples, train
import notebook

importlib.reload(mark_essential_oils)
importlib.reload(notebook)

#%matplotlib widget

# %%
cfg = Config(compounds_str)
cfg.setup(num_hidden=3, hidden_size=len(cfg.all_mol_names), lr=1e-4)

num_mol = len(cfg.all_mol_names)
hp_num_batches = 10
hp_batch_size = 1000
hp_learning_rates = {1e-4: 1000, 5e-5: 1000, 3e-5: 5000, 1e-5: 5000, 3e-6: 5000, 2e-6: 5000, 1e-6: 5000}
hp_num_hidden = [4, 6, 8, 10]
# hp_hidden_size = [num_mol * 6, num_mol * 8, num_mol * 10]
# hp_hidden_size = [num_mol * 6]
hp_hidden_size = [i * num_mol for i in range(4, 12, 2)]

# TODO - only for fast debugging
# hp_learning_rates = {k: v//100 for k, v in hp_learning_rates.items()}
# hp_learning_rates = {k: v//10 for k, v in hp_learning_rates.items()}

data_train = gen_examples(cfg, hp_num_batches, hp_batch_size)
data_val = gen_examples(cfg, hp_num_batches, hp_batch_size)

timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_filename = ["outputs", "mark_essential_oils2", "results-" + timestr]
base_filename = "/".join(base_filename)

total_configs = len(hp_num_hidden) * len(hp_hidden_size)
total_epochs = sum(hp_learning_rates.values())

train_loss_hist = torch.zeros((total_epochs * total_configs, ))
val_loss_hist = torch.zeros_like(train_loss_hist)
val_dist_hist = torch.zeros_like(train_loss_hist)
lr_hist = torch.ones_like(train_loss_hist)

fig = plt.figure(0, (30, 16))
labels_loss = []
labels_dist = []
for num_hidden in hp_num_hidden:
    for hidden_size in hp_hidden_size:
        labels_loss.append(f"val loss (num {num_hidden}, size {hidden_size})")
        labels_dist.append(f"val dist (num {num_hidden}, size {hidden_size})")
labels_loss.append("learning rate")
labels_dist.append("learning rate")

plot_loss = notebook.Plot(total_epochs, labels_loss, fig, 2, 1, 1)
plot_dist = notebook.Plot(total_epochs, labels_dist, fig, 2, 1, 2)
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
            plot_loss.add_data(config_idx, vlosses)
            plot_dist.add_data(config_idx, vdists)
            if config_idx == 0:
                lr_tensor = torch.ones((epochs,)) * lr
                plot_loss.add_data(len(plot_loss.labels) - 1, lr_tensor)
                plot_dist.add_data(len(plot_dist.labels) - 1, lr_tensor)

            smooth_steps = 100
            chop_top_quantile = 0.8
            plot_loss.render(chop_top_quantile, smooth_steps, True)
            plot_dist.render(chop_top_quantile, smooth_steps, True)

            display.clear_output(True)
            display.display(fig)

            epochs_at_cfg += epochs

            # save image & net
            torch_filename = f"{base_filename}-num_hidden_{num_hidden}-hidden_size_{hidden_size}.torch"
            torch.save(cfg.net, torch_filename)
            print(f"saved {torch_filename}")

            img_filename = base_filename + ".png"
            fig.savefig(img_filename)
            print(f"saved {img_filename}")

        config_idx += 1


# %%
