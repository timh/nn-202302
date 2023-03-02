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
import model
import tokens

for m in [notebook, trainer, experiment, model]:
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
        print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.2E}")
        res = model_utils.predict(exp.net, seq_len=exp.seqlen, num_preds=self.num_pred,
                                  tokenizer=exp.tokenizer, dictionary=exp.dictionary,
                                  start_text="", device=device)
        res = res.replace("\n", "\n  ")
        # BUG here: cur_lr is already advanced. why?
        print(f"\033[1;32m  {res}\033[0m")
        print()

        super().on_epoch_end_infrequent(exp, epoch)

def make_textreader(seqlen: int, wordlen: int, batch_size: int, minibatch_count: int, filename: str) -> Tuple[tokens.TextReader, DataLoader, DataLoader]:
    print(f"make_data({seqlen=}, {wordlen=})")
    treader = tokens.WordTextReader(seq_len=seqlen, wordlen=wordlen, filename=filename, include_special=True, device=device)
    all_examples = treader.as_pairs()
    num_train = int(len(all_examples) * 0.8)

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

    return treader, train_dl, val_dl

def experiments(filename = "shakespeare.txt"):
    print("make experiments")

    for exp_idx, exp_params in enumerate(all_exp_params):
        start_lr = exp_params["startlr"]
        end_lr = exp_params["endlr"]
        if exp_idx > 0:
            gc.collect()
            torch.cuda.empty_cache()

        seqlen, wordlen = exp_params["seqlen"], exp_params["wordlen"]
        nhead, nlayers = exp_params["nhead"], exp_params["nlayers"]
        hidlen, emblen = exp_params["hidlen"], exp_params["emblen"]
        # compile = exp_params["compile"]

        batch_size = exp_params["batch"]
        minibatch_count = exp_params["minicnt"]
        epochs = exp_params["epochs"]

        fields = exp_params.copy()
        # fields["dropout"] = format(dropout, ".1f")
        fields["startlr"] = format(start_lr, ".1E")
        fields["endlr"] = format(end_lr, ".1E")

        if minibatch_count == 0:
            minibatch_count = None
            del fields["minicnt"]

        treader, train_dl, val_dl = \
            make_textreader(seqlen=seqlen, wordlen=wordlen, 
                            batch_size=batch_size, minibatch_count=minibatch_count, 
                            filename=filename)
        vocab_len = treader.dictionary.vocab_len
        fields["vocablen"] = vocab_len

        label = ", ".join([f"{key} {val}" for key, val in fields.items()])
        ptr_path = Path("runs", basename + "-" + label)
        if ptr_path.exists():
            print(f"\033[1;32m{ptr_path} exists, skipping\033[0m")
            continue

        # model = mxt.TransformerModel(vocab_len=textmap.vocab_len, emblen=emblen, nhead=nhead, 
        #                                 nlayers=nlayers, hidlen=hidlen, 
        #                                 dropout=dropout, do_layernorm=do_layernorm,
        #                                 device=device)
        net = model.TransformerModel2(vocab_len=vocab_len, emblen=emblen, nhead=nhead, 
                                      nlayers=nlayers, hidlen=hidlen, 
                                      dropout=dropout, device=device)

        if accel is not None:
            net = accel.prepare(net)
        
        if False and compile:
            print("compiling...")
            start = datetime.datetime.now()
            net = torch.compile(net)
            end = datetime.datetime.now()
            print(f"  took {end - start}")

        loss_fn = model_utils.loss_fn(seq_len=seqlen, vocab_len=vocab_len)

        exp = Experiment(label, net, loss_fn, train_dl, val_dl)
        exp.seqlen = seqlen
        exp.dictionary = treader.dictionary
        exp.tokenizer = treader.tokenizer
        exp.start_lr, exp.end_lr = start_lr, end_lr
        exp.optim_type = fields["optim"]
        exp.scheduler_type = fields["sched"]
        exp.epochs = epochs

        print(f"\033[1mstart experiment {exp_idx + 1} / {len(all_exp_params)}: {exp.label}\033[0m")
        time_start = datetime.datetime.now()
        yield exp
        time_end = datetime.datetime.now()
        time_fmt = "%Y%m%d-%H%M%S"
        elapsed = (time_end - time_start).total_seconds()
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed) % 60

        if torch.isnan(exp.train_loss_hist[-1]) or torch.isnan(exp.val_loss_hist[-1]):
            print(f"nan. skipping.")
            continue

        # fields['last_train_loss'] = exp.train_loss_hist[-1]
        # fields['last_val_loss'] = exp.val_loss_hist[-1]
        # fields['elapsed_sec'] = (exp.ended_at - exp.started_at).total_seconds()
        # model_utils.update_csv

        torch_path = str(ptr_path) + f", elapsed {elapsed:.2f}s.torch"
        with open(torch_path, "wb") as torch_file:
            if accel is not None:
                net = accel.unwrap_model(net, False)
            torch.save(net, torch_file)
            print(f"saved {torch_path}")

        with open(ptr_path, "w") as file:
            log_filename = str(Path(logger.dirname, label))
            print(f"write {ptr_path}")
            print(log_filename, file=file)

class NanoGPTCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, start_lr: float, min_lr: float, warmup_epochs: int, lr_decay_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.start_lr = start_lr
        self.min_lr = min_lr
        self._step_count = 0

    def get_lr(self) -> float:
        if self._step_count < self.warmup_epochs:
            return [self.start_lr * self._step_count / self.warmup_epochs]
        if self._step_count > self.lr_decay_epochs:
            return [self.min_lr]
        decay_ratio = (self._step_count - self.warmup_epochs) / (self.lr_decay_epochs - self.warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.start_lr - self.min_lr)]
    
    def step(self):
        self._step_count += 1
    
def get_optimizer_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if exp.optim_type == "sgd":
        optimizer = torch.optim.SGD(exp.net.parameters(), lr=exp.start_lr)
    elif exp.optim_type == "adamw":
        optimizer = torch.optim.AdamW(exp.net.parameters(), lr=exp.start_lr, betas=(0.9, 0.99))
    else:
        raise ValueError(f"unknown {exp.optim_type=}")

    if exp.scheduler_type == "StepLR":
        gamma = (exp.end_lr / exp.start_lr) ** (1 / exp.epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
    elif exp.scheduler_type == "nanogpt-cosine":
        scheduler = NanoGPTCosineScheduler(optimizer, exp.start_lr, exp.end_lr, warmup_epochs=100, lr_decay_epochs=exp.epochs)
    else:
        raise ValueError(f"unknown {exp.scheduler_type=}")
    return optimizer, scheduler

seqlen_values = [64, 128, 256]
wordlen_values = [1]
nhead_values = [1, 2, 4, 6]
nlayers_values = [1, 2, 4, 6]
emblen_values = [96, 192, 384]
# hidlen_values = [emblen * 4 for emblen in emblen_values]
scheduler_values = ["StepLR", "nanogpt-cosine"]
# scheduler_values = ["nanogpt-cosine"]
# scheduler_values = ["StepLR"]
dropout = 0.2

nepochs = 2000
batch_mini_epochs_values = [
    # (64, 1, nepochs),
    (128, 2, nepochs),
    # (256, 1, nepochs),
    (256, 2, nepochs),
]

lrparams_values = [
    ("sgd", 1e-3, 1e-4),
    ("adamw", 1e-3, 1e-4),
    # ("sgd", 1e-3, 5e-4),
    # ("adamw", 1e-3, 5e-4),
]

all_exp_params = [
    dict(seqlen=seqlen, wordlen=wordlen,
         nhead=nhead, nlayers=nlayers,
         emblen=emblen, hidlen=emblen * 4,
         optim=lrparams[0], startlr=lrparams[1], endlr=lrparams[2], sched=sched,
         batch=bme[0], minicnt=bme[1], epochs=bme[2])

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
random.shuffle(all_exp_params)

# basename = "mm-ss4tut-sgd-fast2"
basename = f"fixed2_{nepochs}"
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
