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
import model_xformers
import model_xformers_tutorial as mxt

for m in notebook, trainer, model_xformers, experiment:
    importlib.reload(m)

# %%
device = "cuda"

# accel = Accelerator()
accel = None
        
class MakemoreLogger(trainer.TensorboardLogger):
    def __init__(self, num_pred: int, basename: str):
        super().__init__(basename)
        self.num_pred = num_pred
    
    def on_epoch_end_infrequent(self, exp: Experiment, epoch: int):
        # res = exp.net.predict(self.num_pred, device=device)
        res = model_utils.predict(exp.net, textmap=exp.textmap, num_preds=self.num_pred, seq_len=exp.seq_len, device=device)
        res = res.replace("\n", "\n  ")
        # BUG here: cur_lr is already advanced. why?
        print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.2E}")
        print(f"\033[1;32m  {res}\033[0m")
        print()

        super().on_epoch_end_infrequent(exp, epoch)

def make_textmapper(seq_len: int, wordmaxlen: int, batch_size: int, minibatch_count: int, filename: str) -> Tuple[mxt.TextMapper, DataLoader, DataLoader]:
    print(f"make_data({seq_len=}, {wordmaxlen=})")
    textmap = model_xformers.TextMapper(seq_len, filename=filename, device=device, dtype=torch.long, wordmaxlen=wordmaxlen)
    all_examples = textmap.as_pairs()
    num_train = int(len(all_examples) * 0.8)

    # NOTE: karpathy uses a single mini-batch per epoch, of size (seq_len)
    train_data = all_examples[:num_train]
    if minibatch_count:
        train_sampler = RandomSampler(train_data, num_samples=batch_size * minibatch_count)
        train_dl = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    else:
        train_dl = DataLoader(train_data, batch_size=batch_size)

    val_data = all_examples[num_train:]
    if minibatch_count:
        val_sampler = RandomSampler(val_data, num_samples=batch_size * minibatch_count)
        val_dl = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)
    else:
        val_dl = DataLoader(val_data, batch_size=batch_size)
    print(f"  {len(train_data)=}, {len(val_data)=}")
    print(f"  {len(train_dl)=}, {len(val_dl)=}")

    if accel is not None:
        train_dl, val_dl = accel.prepare(train_dl, val_dl)

    first_inputs, first_truth = next(iter(val_dl))
    print(f"  {len(first_inputs)=}")
    print(f"  {len(first_truth)=}")

    return textmap, train_dl, val_dl

def experiments(filename = "shakespeare.txt"):
    print("make experiments")

    for exp_idx, exp_params in enumerate(all_exp_params):
        start_lr = exp_params["startlr"]
        end_lr = exp_params["endlr"]
        if exp_idx > 0:
            print(f"before gc: {torch.cuda.memory_allocated()/1024/1024}")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"after gc: {torch.cuda.memory_allocated()/1024/1024}")

        seq_len, wordmaxlen = exp_params["seqlen"], exp_params["wordlen"]
        nhead, nlayers = exp_params["nhead"], exp_params["nlayers"]
        hidden_len, emb_len = exp_params["hidlen"], exp_params["emblen"]
        do_layernorm = exp_params["norm"]
        compile = exp_params["compile"]

        batch_size = exp_params["batch"]
        minibatch_count = exp_params["minicnt"]
        epochs = exp_params["epochs"]

        fields = exp_params.copy()
        fields["dropout"] = format(dropout, ".1f")
        fields["startlr"] = format(start_lr, ".1E")
        fields["endlr"] = format(end_lr, ".1E")

        if minibatch_count == 0:
            minibatch_count = None
            del fields["minicnt"]

        textmap, train_dl, val_dl = make_textmapper(seq_len=seq_len, wordmaxlen=wordmaxlen, batch_size=batch_size, minibatch_count=minibatch_count, filename=filename)
        fields["vocablen"] = textmap.vocab_len

        label = ", ".join([f"{key} {val}" for key, val in fields.items()])
        ptr_path = Path("runs", basename + "-" + label)
        if ptr_path.exists():
            print(f"\033[1;32m{ptr_path} exists, skipping\033[0m")
            continue

        model = mxt.TransformerModel(vocab_len=textmap.vocab_len, emb_len=emb_len, nhead=nhead, 
                                        nlayers=nlayers, hidden_len=hidden_len, 
                                        dropout=dropout, do_layernorm=do_layernorm,
                                        device=device)

        if accel is not None:
            model = accel.prepare(model)
        
        if compile:
            print("compiling...")
            start = datetime.datetime.now()
            model = torch.compile(model)
            end = datetime.datetime.now()
            print(f"  took {end - start}")

        loss_fn = mxt.loss_fn(seq_len=seq_len, vocab_len=textmap.vocab_len)
        exp = Experiment(label, model, loss_fn, train_dl, val_dl)
        exp.seq_len = seq_len
        exp.textmap = textmap
        exp.start_lr, exp.end_lr = start_lr, end_lr
        exp.optim_type = fields["optim"]
        exp.epochs = epochs
        print(f"\033[1mstart experiment {exp_idx + 1} / {len(all_exp_params)}: {exp.label}\033[0m")
        yield exp

        if torch.isnan(exp.train_loss_hist[-1]) or torch.isnan(exp.val_loss_hist[-1]):
            print(f"nan. skipping.")
            continue

        # fields['last_train_loss'] = exp.train_loss_hist[-1]
        # fields['last_val_loss'] = exp.val_loss_hist[-1]
        # fields['elapsed_sec'] = (exp.ended_at - exp.started_at).total_seconds()
        # model_utils.update_csv

        torch_path = str(ptr_path) + ".torch"
        with open(torch_path, "wb") as torch_file:
            if accel is not None:
                model = accel.unwrap_model(model, False)
            torch.save(model, torch_file)
            print(f"saved {torch_path}")

        with open(ptr_path, "w") as file:
            log_filename = str(Path(logger.dirname, label))
            print(f"write {ptr_path}")
            print(log_filename, file=file)


def get_optimizer_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    gamma = (exp.end_lr / exp.start_lr) ** (1 / exp.epochs)
    if exp.optim_type == "sgd":
        optimizer = torch.optim.SGD(exp.net.parameters(), lr=exp.start_lr)
    elif exp.optim_type == "adamw":
        optimizer = torch.optim.SGD(exp.net.parameters(), lr=exp.start_lr)
    else:
        raise ValueError(f"unknown {exp.optim_type=}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
    return optimizer, scheduler

seq_len_values = [256]
wordmaxlen_values = [1]
nhead_values = [6]
nlayers_values = [6]
hidden_len_values = [256*4]
emb_len_values = [384]
do_layernorm_values = [True]
compile_values = [False]

batch_mini_epochs_values = [
    # (64, 1, 1000),
    # (64, 1, 2000),
    # (64, 1, 5000),
    (64, 2, 5000),
    # (128, 1, 5000),
    (128, 2, 5000),
]

lrparams_values = [
    ("sgd", 1e-3, 1e-4),
    ("adamw", 1e-3, 1e-4),
    ("sgd", 1e-3, 5e-4),
    ("adamw", 1e-3, 5e-4),
]

dropout = 0.0

all_exp_params = [
    dict(seqlen=seq_len, wordlen=wordmaxlen,
         nhead=nhead, nlayers=nlayers, hidlen=hidden_len,
         emblen=emb_len,
         norm=do_layernorm,
         optim=lrparams[0], startlr=lrparams[1], endlr=lrparams[2],
         compile=compile,
         batch=bme[0], minicnt=bme[1], epochs=bme[2])

    # most quickly changing should be at top:
    for lrparams in lrparams_values
    for do_layernorm in do_layernorm_values
    for emb_len in emb_len_values
    for hidden_len in hidden_len_values
    for nlayers in nlayers_values
    for nhead in nhead_values
    for wordmaxlen in wordmaxlen_values
    for seq_len in seq_len_values
    for compile in compile_values
    for bme in batch_mini_epochs_values
]
random.shuffle(all_exp_params)

# basename = "mm-ss4tut-sgd-fast2"
basename = "mm-ss4tut-karpathy-lnorm_100"
if accel is not None:
    basename = basename + "-accel"

# %%
print("train")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')

# for debug only TODO
# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

filename = "shakespeare.txt"
if (len(sys.argv) > 1 and sys.argv[1] == "-d"):
    filename = "shakespeare-1000.txt"

tcfg = trainer.TrainerConfig(experiments=experiments(filename), get_optimizer_fn=get_optimizer_fn, accel=accel)
logger = MakemoreLogger(num_pred=100, basename=basename)
tr = trainer.Trainer(logger=logger)
tr.train(tcfg)
