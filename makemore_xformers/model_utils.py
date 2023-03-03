from typing import List, Dict, Tuple, Literal, Set, Union, Callable
import re
import math
from pathlib import Path
import csv
import sys
from dataclasses import dataclass
import datetime

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, "..")

import tokens
import trainer
from experiment import Experiment
import model
from model import TextExperiment

class MakemoreLogger(trainer.TensorboardLogger):
    def __init__(self, num_pred: int, basename: str, start_text = "", device = "cpu"):
        super().__init__(basename)
        self.num_pred = num_pred
        self.device = device
        self.start_text = start_text
    
    def on_epoch_end_infrequent(self, exp: Experiment, epoch: int):
        print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.2E}")
        res = predict(exp.net, seq_len=exp.seqlen, num_preds=self.num_pred, 
                      tokenizer=exp.tokenizer, dictionary=exp.dictionary,
                      start_text=self.start_text, device=self.device)
        res = res.replace("\n", "\n  ")
        # BUG here: cur_lr is already advanced. why?
        print(f"\033[1;32m  {res}\033[0m")
        print()

        super().on_epoch_end_infrequent(exp, epoch)

def predict(net: nn.Module, 
             seq_len: int, num_preds: int, 
             tokenizer: tokens.Tokenizer, dictionary: tokens.Dictionary, 
             start_text = "", device = "cpu", dtype = torch.int32):
    net.eval()

    inputs = torch.zeros((1, seq_len), device=device, dtype=dtype)
    nextinputs = torch.zeros_like(inputs)

    if start_text:
        words = tokenizer.tokenize(start_text)
        start_tensors = dictionary.words_to_tensors(words, device=device, dtype=dtype)
        if start_tensors[-1].item() == dictionary.token_end.token:
            start_tensors = start_tensors[:-1]
        start_len = len(start_tensors)
        if len(start_tensors) > seq_len:
            print(f"{start_tensors.shape} longer than {seq_len=}, clamping from front")
            inputs[0] = start_tensors[len(start_text) - seq_len:]
        else:
            inputs[0, seq_len - len(start_tensors):] = start_tensors
    else:
        start_len = 0

    res = start_text
    for i in range(start_len, num_preds + start_len):
        if i < seq_len - 1:
            ones = torch.ones((seq_len, seq_len), device=device)
            mask = torch.zeros_like(ones)
            # triu = can't see into the future
            # tril = can't see further back how many tokens in output so far
            mask[torch.triu(ones, diagonal=1) == 1] = float('-inf')
            mask[torch.tril(ones, diagonal=-i) == 1] = float('-inf')
        else:
            mask = None

        outputs = net(inputs, mask)
        outputs = F.softmax(outputs, -1)
        word_idx = torch.multinomial(outputs[0, -1], 1).item()

        res += dictionary.tokens_to_str([word_idx])

        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = word_idx
        inputs, nextinputs = nextinputs, inputs

    return res

def get_optimizer_fn(exp: Experiment) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if exp.optim_type == "sgd":
        optimizer = torch.optim.SGD(exp.net.parameters(), lr=exp.startlr)
    elif exp.optim_type == "adamw":
        optimizer = torch.optim.AdamW(exp.net.parameters(), lr=exp.startlr, betas=(0.9, 0.99))
    else:
        raise ValueError(f"unknown {exp.optim_type=}")

    if exp.sched_type == "StepLR":
        gamma = (exp.endlr / exp.startlr) ** (1 / exp.epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
    elif exp.sched_type == "nanogpt-cosine":
        scheduler = trainer.NanoGPTCosineScheduler(optimizer, exp.startlr, exp.endlr, warmup_epochs=100, lr_decay_epochs=exp.epochs)
    else:
        raise ValueError(f"unknown {exp.sched_type=}")
    return optimizer, scheduler


"""
takes partially initialized experiments (with hyperparams) then fills out
their textreader, net, optim, sched. yields each in turn.
"""
def gen_experiments(basename: str, text_filename: str, all_exp: List[TextExperiment], train_split: float = 0.8, compile = False, device = "cpu"):
    print("gen_experiments")

    for exp_idx, exp in enumerate(all_exp):
        treader = tokens.WordTextReader(seq_len=exp.seqlen, wordlen=exp.wordlen,
                                        filename=text_filename,
                                        include_special=True, device=device)
        
        ntrain = int(len(treader) * train_split)
        train_data, val_data = treader.train_val_split(ntrain)

        if exp.minicnt:
            train_sampler = RandomSampler(train_data, replacement=True, num_samples=exp.batch * exp.minicnt)
            train_dl = DataLoader(train_data, batch_size=exp.batch, sampler=train_sampler, drop_last=True)

            val_sampler = RandomSampler(val_data, replacement=True, num_samples=exp.batch * exp.minicnt)
            val_dl = DataLoader(val_data, batch_size=exp.batch, sampler=val_sampler, drop_last=True)
        else:
            train_dl = DataLoader(train_data, batch_size=exp.batch, drop_last=True)
            val_dl = DataLoader(val_data, batch_size=exp.batch, drop_last=True)
        
        print(f"{len(train_data)=}, {len(val_data)=}")
        print(f"{len(train_dl)=}, {len(val_dl)=}")
        print(f"{len(next(iter(train_dl))[0])=}")

        fields = exp.to_dict()
        fields["optim"] = fields["optim_type"]
        fields["sched"] = fields["sched_type"]
        del fields["optim"]
        del fields["sched"]

        label = ", ".join([f"{key} {val}" for key, val in fields.items()])
        dir_path = Path("runs", basename + "-" + label)
        if dir_path.exists():
            print(f"\033[1;32m{dir_path} exists, skipping\033[0m")
            continue

        # if accel is not None:
        #     net = accel.prepare(net)
        
        exp.tokenizer = treader.tokenizer
        exp.dictionary = treader.dictionary
        exp.vocablen = treader.dictionary.vocab_len
        exp.loss_fn = model.loss_fn(seqlen=exp.seqlen, vocablen=exp.vocablen)
        exp.net = model.from_experiment(exp, device=device)
        exp.train_dataloader = train_dl
        exp.val_dataloader = val_dl
        exp.label = label

        if compile:
            print("compiling...")
            start = datetime.datetime.now()
            exp.net = torch.compile(exp.net)
            end = datetime.datetime.now()
            print(f"  compile took {end - start}")

        print(f"\033[1mstart experiment {exp_idx + 1} / {len(all_exp)}: {exp.label}\033[0m")
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

        extras = f", elapsed {elapsed:.2f}s, vloss {exp.val_loss_hist[-1]:.3f}.ckpt"
        torch_path = str(dir_path) + extras
        with open(torch_path, "wb") as torch_file:
            checkpoint = exp.state_dict()

            # if accel is not None:
            #     net = accel.unwrap_model(net, False)
            print(f"saving {torch_path}...")
            start = datetime.datetime.now()
            torch.save(checkpoint, torch_file)
            end = datetime.datetime.now()
            print(f"  save took {end - start}")
