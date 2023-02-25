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
    def __init__(self, num_pred: int):
        self.now = datetime.datetime.now()
        self.num_pred = num_pred
        pass
    
    def on_exp_start(self, exp: Experiment):
        super().__init__(f"mm-ss2", now=self.now)
        return super().on_exp_start(exp)

    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch_end_infrequent(exp, exp_epoch, lr_epoch)

        res = exp.net.predict(self.num_pred, device=device)
        res = "\n" + res
        res = res.replace("\n", "\n  ")
        print(f"predict({self.num_pred}): \033[1;32m{res}\033[0m")

def experiments(filename = "shakespeare.txt"):
    print("make experiments")
    for numchar in numchar_values:
        print(f"make_data({numchar})")
        encdec = model_xformers.TextEncDec(numchar, filename=filename, device=device, dtype=torch.long)
        all_examples = encdec.as_pairs(batch_size)
        num_train = int(len(all_examples) * 0.8)
        train_dl = all_examples[num_train:]
        val_dl = all_examples[:num_train]
        print(f"  {len(train_dl)=}, {len(val_dl)=}")

        for nblock in nblock_values:
            for nhead in nhead_values:
                for emb_len in emb_len_values:
                    label = f"numchar {numchar} - nblock {nblock}, nhead {nhead} - emb_len {emb_len:3}"
                    model = model_xformers.LangModel(encdec=encdec, nblock=nblock, 
                                                        do_layernorm=do_layernorm, do_residual=do_residual,
                                                        nhead=nhead, emb_len=emb_len,
                                                        device=device)
                    exp = Experiment(label, model, None, train_dl, val_dl)
                    exp.numchar = numchar
                    yield exp

batch_size = 2048
learning_rates = [
    (3e-4, 5000), # karpathy
    # (5e-5,   100),
    # (1e-5,   100),
]
# for debug only TODO

nblock_values = [6]
do_layernorm = True
do_residual = True

nhead_values = [6]
numchar_values = [64]
emb_len_values = [384]

# %%
print("train")

# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

filename = "shakespeare.txt"
if (len(sys.argv) > 1 and sys.argv[1] == "-d"):
    filename = "shakespeare-1000.txt"

tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, experiments(filename))
tr = trainer.Trainer(logger=MakemoreLogger(num_pred=50))
tr.train(tcfg)

# %%
