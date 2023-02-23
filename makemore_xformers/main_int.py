# %%
import sys
import importlib

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
numchar = 4
embedding_dim = 10
num_hidden = 2
hidden_size = 20
batch_size = 4096

learning_rates = [
    (1e-4, 1000),
    (5e-5, 1000),
    (1e-5, 1000),
    (5e-5, 1000),
    (1e-6, 1000)
]
exp_epochs = sum([lrpair[1] for lrpair in learning_rates])

def get_optimizer_fn(exp: Experiment, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(exp.net.parameters(), lr)

class MakemoreLogger(trainer.GraphLogger):
    def on_epoch(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch(exp, exp_epoch, lr_epoch)

        res = model.inference(numchar, embedding_dim, 5, exp.net, device=device)
        print(f"inference: {res}")

        self.fig_loss.savefig("outputs/loss.png")

# %%
print("make_net")
net = model.make_net(numchar, embedding_dim, num_hidden, hidden_size, device=device)
loss_fn = nn.CrossEntropyLoss()
# def loss_fn(outputs: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
#     res = (outputs.argmax(1) != truth).float()
#     res = res.mean()
#     res.requires_grad_(True)
#     return res

print("make_data")
all_data = model.make_data(numchar, device)
num_train = int(len(all_data) * 0.8)
train_data = all_data[:num_train]
train_dataloader = DataLoader(train_data, batch_size)
val_data = all_data[num_train:]
val_dataloader = DataLoader(val_data, batch_size)
print(f"  {len(train_data)=}, {len(train_dataloader)=}")
print(f"  {len(val_data)=}, {len(val_dataloader)=}")

# %%
if True:
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
ex = Experiment(f"numchar {numchar}, embdim {embedding_dim} - num {num_hidden}/size {hidden_size}", net, loss_fn, train_dataloader, val_dataloader)
tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, 1, [ex])
tr = trainer.Trainer(logger=MakemoreLogger(exp_epochs, 1, fig_loss, False))
tr.train(tcfg)

# %%
