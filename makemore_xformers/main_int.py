# %%
import sys
import importlib
from typing import List, Callable, Tuple
import datetime
from pathlib import Path
import random
import math
import gc

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator

sys.path.insert(0, "..")
import notebook
import trainer
import experiment
from experiment import Experiment
import model_utils
from model_utils import TextExperiment
import model
import tokens

for m in [notebook, trainer, experiment, model]:
    importlib.reload(m)

# %%
device = "cuda"

# accel = Accelerator()
accel = None

# seqlen_values = [256, 512]
seqlen_values = [256]
wordlen_values = [1]
nhead_values = [2, 4, 6]
nlayers_values = [1, 2, 4, 6]
emblen_values = [384]
scheduler_values = ["StepLR", "nanogpt-cosine"]
dropout = 0.2

nepochs = 1000
batch_mini_epochs_values = [
    # (64, 1, nepochs),
    (128, 2, nepochs),
    # (256, 1, nepochs),
    # (256, 2, nepochs),
    # (256, 4, nepochs),
]

lrparams_values = [
    # ("sgd", 1e-3, 1e-4),
    ("adamw", 1e-3, 1e-4),
    # ("sgd", 1e-3, 5e-4),
    # ("adamw", 1e-3, 5e-4),
]

all_exp = [
    TextExperiment(seqlen=seqlen, wordlen=wordlen, nhead=nhead, nlayers=nlayers,
                   emblen=emblen, hidlen=emblen * 4,
                   optim_type=lrparams[0], sched_type=sched, startlr=lrparams[1], endlr=lrparams[2], 
                   batch=bme[0], minicnt=bme[1], epochs=bme[2],
                   dropout=dropout)

    # most quickly changing should be at top:
    for lrparams in lrparams_values
    for sched in scheduler_values
    for emblen in emblen_values
    for nlayers in nlayers_values
    for nhead in nhead_values
    for wordlen in wordlen_values
    for seqlen in seqlen_values
    for bme in batch_mini_epochs_values
]
random.shuffle(all_exp)

# basename = "mm-ss4tut-sgd-fast2"
basename = f"python_{nepochs}"
if accel is not None:
    basename = basename + "-accel"

# %%
print("train")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')

# for debug only TODO
# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

# filename = "shakespeare.txt"
# filename = "all_python_100000.txt"
filename = "shakespeare.txt"
# print(f"{sys.argv[1]=}")
# if len(sys.argv) > 1 and sys.argv[1] == "-d":
#     filename = "shakespeare-1000.txt"

experiments = model_utils.gen_experiments(basename=basename, text_filename=filename, all_exp=all_exp, device=device)
tcfg = trainer.TrainerConfig(experiments=experiments, 
                             get_optimizer_fn=model_utils.get_optimizer_fn,
                             accel=accel)
logger = model_utils.MakemoreLogger(num_pred=100, basename=basename, device=device, start_text="\n")
tr = trainer.Trainer(logger=logger)
tr.train(tcfg)
