# %%
import sys
import importlib
from typing import List
import datetime

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
import model_xformers

for m in notebook, trainer, model, model_xformers, experiment:
    importlib.reload(m)

# %%
device = "cuda"

def get_optimizer_fn(exp: Experiment, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(exp.net.parameters(), lr)

loss_fn = nn.CrossEntropyLoss()

class MakemoreLogger(trainer.TensorboardLogger):
    def __init__(self, num_pred = 5):
        self.now = datetime.datetime.now()
        self.num_pred = num_pred
        pass
    
    def on_exp_start(self, exp: Experiment):
        super().__init__(f"mm-xformer-kqv_mh-{exp.numchar}", now=self.now)
        return super().on_exp_start(exp)

    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch_end_infrequent(exp, exp_epoch, lr_epoch)

        res = model.inference(exp.numchar, self.num_pred, exp.net, device=device)
        print(f"  inference({self.num_pred}): {res}")

def experiments(filename = "names.txt"):
    print("make experiments")
    for numchar in numchar_values:
        print(f"make_data({numchar})")
        all_data = model.make_data(numchar, filename=filename, device=device, dtype=torch.long)
        num_train = int(len(all_data) * 0.8)
        train_data = all_data[:num_train]
        train_dataloader = DataLoader(train_data, batch_size)
        val_data = all_data[num_train:]
        val_dataloader = DataLoader(val_data, batch_size)
        print(f"  {len(train_data)=}, {len(train_dataloader)=}")
        print(f"  {len(val_data)=}, {len(val_dataloader)=}")

        for nhead in nhead_values:
            for emb_len in emb_len_values:
                for kqv_len in kqv_len_values:
                    label = f"nhead {nhead}, numchar {numchar}, emb_len {emb_len:3}, kqv_len {kqv_len:3}"
                    net = model_xformers.make_net_xformers(nhead=nhead, numchar=numchar, emb_len=emb_len, kqv_len=kqv_len, device=device)
                    exp = Experiment(label, net, None, train_dataloader, val_dataloader)
                    exp.numchar = numchar
                    yield exp

batch_size = 2048
learning_rates = [
    (3e-4,   50),
    (1e-4,   50),
    (5e-5,  100),
    (3e-5,  100),
    (1e-5,  100),
    (5e-6,  100),
    (3e-6,  100),
    (1e-6,  100)
]
# for debug only TODO

nhead_values = [4, 8]
numchar_values = [5, 10]
emb_len_values = [16, 64]
kqv_len_values = [16, 64]

# %%
print("train")

# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

# filename = "names-1000.txt"
filename = "names.txt"
tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, experiments(filename))
tr = trainer.Trainer(logger=MakemoreLogger(num_pred=10))
tr.train(tcfg)

# %%
