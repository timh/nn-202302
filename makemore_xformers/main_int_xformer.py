# %%
import sys
import importlib
from typing import List
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
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
        super().__init__("mm-ss3b")
        self.num_pred = num_pred
    
    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch_end_infrequent(exp, exp_epoch, lr_epoch)

        res = exp.net.predict(self.num_pred, device=device)
        res = res.replace("\n", "\n  ")
        print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.1E}")
        print(f"\033[1;32m  {res}\033[0m")
        print()

def experiments(filename = "shakespeare.txt"):
    print("make experiments")
    for numchar in numchar_values:
        print(f"make_data({numchar})")
        encdec = model_xformers.TextEncDec(numchar, filename=filename, device=device, dtype=torch.long)
        all_examples = encdec.as_pairs()
        num_train = int(len(all_examples) * 0.8)

        # NOTE: karpathy uses a single mini-batch per epoch, of size (numchar)
        train_data = all_examples[:num_train]
        train_sampler = RandomSampler(train_data, num_samples=batches_per_epoch * batch_size)
        train_dl = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

        val_data = all_examples[num_train:]
        val_sampler = RandomSampler(val_data, num_samples=batches_per_epoch * batch_size)
        val_dl = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)
        print(f"  {len(train_data)=}, {len(val_data)=}")
        print(f"  {len(train_dl)=}, {len(val_dl)=}")

        first_inputs, first_truth = next(iter(val_dl))
        print(f"{len(first_inputs)=}")
        print(f"{len(first_truth)=}")

        for nblock in nblock_values:
            for nhead in nhead_values:
                for emb_len in emb_len_values:
                    fields = dict(
                        batch_size=batch_size,
                        batches_per_epoch=batches_per_epoch,
                        dropout=dropout,
                        numchar=numchar,
                        nblock=nblock,
                        nhead=nhead,
                        emb_len=emb_len
                    )
                    label = ", ".join([(f"{key} " + format(val, ".1f" if key == "dropout" else "3")) for key, val in fields.items()])
                    ptr_path = Path("runs", label)
                    if ptr_path.exists():
                        print(f"\033[1;32m{ptr_path} exists, skipping\033[0m")
                        continue

                    with open(ptr_path, "w") as file:
                        log_filename = str(Path(logger.dirname, label))
                        print(f"write {ptr_path}")
                        print(log_filename, file=file)

                    model = model_xformers.LangModel(encdec=encdec, nblock=nblock, dropout=dropout,
                                                     do_layernorm=do_layernorm, do_residual=do_residual,
                                                     nhead=nhead, emb_len=emb_len, device=device)
                    
                    # first_inputs, _first_truth = next(iter(val_dl))
                    # first_inputs: Tensor = first_inputs[:1]
                    # logger.writer.add_graph(model, first_inputs, use_strict_trace=False)

                    # exp = Experiment(label, model, loss_fn, train_dl, val_dl)
                    exp = Experiment(label, model, None, train_dl, val_dl)
                    exp.numchar = numchar
                    yield exp

learning_rates = [
    # (3e-4,  5000), # karpathy
    (3e-4,  2000),
    (5e-5,  4000),
    # (3e-4,  1000),
    # (1e-4,  1000),
    # (1e-5,  1000),
    # (5e-6,  1000),
    # (1e-6,  1000),
]
# for debug only TODO

do_layernorm = True
do_residual = True

# nc 64, nb 2, nh 4, el 24
nblock_values = [2, 4]
nhead_values = [2, 4, 6]
numchar_values = [32, 64, 96]
emb_len_values = [12, 24]
dropout = 0.2
batch_size = 2048
batches_per_epoch = 4

# %%
print("train")

# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

filename = "shakespeare.txt"
if (len(sys.argv) > 1 and sys.argv[1] == "-d"):
    filename = "shakespeare-1000.txt"

tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, experiments(filename))
logger = MakemoreLogger(num_pred=50)
tr = trainer.Trainer(logger=logger)
tr.train(tcfg)

# %%
