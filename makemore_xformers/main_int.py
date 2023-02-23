# %%
import sys
import importlib
from typing import List

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# from IPython import display
# from PIL import Image

sys.path.insert(0, "..")
import notebook
import trainer
import experiment
from experiment import Experiment

import model

for m in notebook, trainer, model, experiment:
    importlib.reload(m)

# %%
device = "cuda"
numchar_values = [3, 4, 5]
embedding_dim_values = [10, 50, 100]
num_hidden_values = [2, 3, 4]
hidden_size_values = [10, 50, 100]
batch_size = 4096

learning_rates = [
    (3e-4,  200),
    (1e-4, 1000),
    (5e-5, 1000),
    (1e-5, 1000),
    (5e-5, 1000),
    (1e-6, 1000)
]
# for debug only TODO
# learning_rates = [(lrpair[0], lrpair[1]//100) for lrpair in learning_rates]

exp_epochs = sum([lrpair[1] for lrpair in learning_rates])

def get_optimizer_fn(exp: Experiment, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(exp.net.parameters(), lr)

class MakemoreLogger(trainer.GraphLogger):
    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch(exp, exp_epoch, lr_epoch)

        num_pred = 5
        res = model.inference(exp.numchar, exp.embedding_dim, num_pred, exp.net, device=device)
        print(f"  inference({num_pred}): {res}")

        self.fig_loss.savefig("outputs/loss.png")

# %%
loss_fn = nn.CrossEntropyLoss()

def experiments():
    print("make experiments")
    for numchar in numchar_values:
        print(f"make_data({numchar})")
        all_data = model.make_data(numchar, device)
        num_train = int(len(all_data) * 0.8)
        train_data = all_data[:num_train]
        train_dataloader = DataLoader(train_data, batch_size)
        val_data = all_data[num_train:]
        val_dataloader = DataLoader(val_data, batch_size)
        print(f"  {len(train_data)=}, {len(train_dataloader)=}")
        print(f"  {len(val_data)=}, {len(val_dataloader)=}")

        for embedding_dim in embedding_dim_values:
            for num_hidden in num_hidden_values:
                for hidden_size in hidden_size_values:
                    label = f"numchar {numchar}, embdim {embedding_dim} | numhid {num_hidden}, hidsiz {hidden_size}"
                    net = model.make_net(numchar, embedding_dim, num_hidden, hidden_size, device=device)
                    exp = Experiment(label, net, loss_fn, train_dataloader, val_dataloader)
                    exp.numchar = numchar
                    exp.embedding_dim = embedding_dim
                    yield exp

num_experiments = len(numchar_values) * len(embedding_dim_values) * len(num_hidden_values) * len(hidden_size_values)

# %%
if False:
    optim = torch.optim.AdamW(net.parameters(), 1e-4)
    for i, (inputs, truth) in enumerate(train_dataloader):
        out = net(inputs)
        l = loss_fn(out, truth)
        l.backward()
        optim.step()
        print(f"{l.item()}")

        if i >= 10:
            break

    model.inference(numchar, embedding_dim, 10, net, device)

# %%
print("train")

fig_loss = plt.figure(0, (20, 10))
tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, 1, experiments())
tr = trainer.Trainer(logger=MakemoreLogger(exp_epochs, num_experiments, fig_loss, False))
tr.train(tcfg)

# %%
