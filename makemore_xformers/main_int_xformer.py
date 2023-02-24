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
        super().__init__(f"mm-ss", now=self.now)
        return super().on_exp_start(exp)

    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch_end_infrequent(exp, exp_epoch, lr_epoch)

        res = model.predict(exp.net, exp.numchar, self.num_pred, device=device)
        print(f"  predict({self.num_pred}): {res}")

def experiments(filename = "shakespeare.txt"):
    print("make experiments")
    for numchar in numchar_values:
        print(f"make_data({numchar})")
        ted = model_xformers.TextEncDec(numchar, filename=filename, device=device, dtype=torch.long)
        all_examples = ted.as_pairs(batch_size)
        num_train = int(len(all_examples) * 0.8)
        train_dl = all_examples[num_train:]
        val_dl = all_examples[:num_train]
        print(f"  {len(train_dl)=}, {len(val_dl)=}")

        for nhead in nhead_values:
            for emb_len in emb_len_values:
                for head_size in head_size_values:
                    label = f"nhead {nhead}, numchar {numchar}, emb_len {emb_len:3}, head_size {head_size:3}"
                    net = model_xformers.make_net_xformers_big(ted=ted, nhead=nhead, numchar=numchar, emb_len=emb_len, head_size=head_size, device=device)
                    exp = Experiment(label, net, None, train_dl, val_dl)
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

nhead_values = [6]
numchar_values = [128, 256]
emb_len_values = [16, 64]
head_size_values = [96, 384]

# %%
print("train")

# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

filename = "shakespeare.txt"
if (len(sys.argv) > 1 and sys.argv[1] == "-d"):
    filename = "shakespeare-1000.txt"

tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, experiments(filename))
tr = trainer.Trainer(logger=MakemoreLogger(num_pred=256))
tr.train(tcfg)

# %%
