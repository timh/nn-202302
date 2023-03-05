# %%
import sys
import importlib
from typing import List, Callable, Tuple
import datetime
from pathlib import Path
import random
import math
import gc
import argparse

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

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

default_nepochs = 1000
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", required=True)
parser.add_argument("-N", "--name", required=False)
parser.add_argument("-n", "--nepochs", type=int, default=default_nepochs)
parser.add_argument("-c", "--config_file", required=True)
parser.add_argument("--no_compile", default=False, action='store_true')
parser.add_argument("--flash", default="pytorch", choices=["pytorch", "xformers", "none"])
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_amp", default=False, action='store_true')

cfg = parser.parse_args()
if cfg.name is None:
    cfg.name = Path(cfg.filename).stem

has_compile = hasattr(torch, "compile")
has_pytorch_flash_attention = hasattr(F, "scaled_dot_product_attention")
compile = has_compile and not cfg.no_compile

if cfg.flash == "pytorch" and not has_pytorch_flash_attention:
    raise ValueError("pytorch nightly is required to use torch.nn.functional.scaled_dot_product_attention; pass --no_flash to avoid this error")

all_exp: List[TextExperiment]     # instantiated in config file.
with open(cfg.config_file, "r") as cfile:
    ctext = cfile.read()
    exec(ctext)

basename = f"{cfg.name}_{cfg.nepochs}"
if cfg.use_amp:
    basename = basename + "+amp"

# %%
print("train")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')

# for debug only TODO
# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

for exp in all_exp:
    exp.seed = cfg.seed
    exp.compile = compile
    exp.flash = cfg.flash

experiments = model_utils.gen_experiments(basename=basename, text_filename=cfg.filename, all_exp=all_exp, device=device)
tcfg = trainer.TrainerConfig(experiments=experiments, nexperiments=len(all_exp),
                             get_optimizer_fn=model_utils.get_optimizer_fn)
logger = model_utils.MakemoreLogger(num_pred=100, basename=basename, device=device, start_text="\n")
tr = trainer.Trainer(logger=logger)
tr.train(tcfg, device=device, use_amp=cfg.use_amp)
