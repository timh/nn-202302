# %%
from typing import List, Dict, Tuple, Literal, Set, Union, Callable, Optional
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
import text_experiment
from text_experiment import TextExperiment

class MakemoreLogger(trainer.TensorboardLogger):
    def __init__(self, num_pred: int, basename: str, start_text = "", device = "cpu"):
        super().__init__(basename)
        self.num_pred = num_pred
        self.device = device
        self.start_text = start_text
    
    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        # print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.2E}")
        print(f"predict({self.num_pred})")
        res = predict(exp.net, seq_len=exp.seqlen, num_preds=self.num_pred, 
                      tokenizer=exp.tokenizer, dictionary=exp.dictionary,
                      start_text=self.start_text, device=self.device)
        res = res.replace("\n", "\n  ")
        # BUG here: cur_lr is already advanced. why?
        print(f"\033[1;32m  {res}\033[0m")
        print()

def predict(net: nn.Module, 
             seq_len: int, num_preds: int, 
             tokenizer: tokens.Tokenizer, dictionary: tokens.Dictionary, 
             top_k: Optional[int] = None, temperature = 1.0,
             start_text = "", include_start_in_output = True, device = "cpu", dtype = torch.int32):
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
        # must start below with start_len > 0, or the mask will be only float(-inf)
        start_len = 1

    res = start_text if include_start_in_output else ""
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
        outputs = outputs[0, -1]            # only need first batch, last answer
        outputs = outputs / temperature     # < 1 = smooth out probabilities (i.e., more deterministic).
                                            # > 1 = make them more spikey
        if top_k is not None:
            values, _indices = torch.topk(outputs, k=top_k, dim=-1)
            outputs[outputs < values[-1]] = float('-inf')

        outputs = F.softmax(outputs, -1)
        word_idx = torch.multinomial(outputs, 1).item()

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
def gen_experiments(basename: str, text_filename: str, all_exp: List[TextExperiment], train_split: float = 0.8, device = "cpu"):
    print("gen_experiments")

    for exp_idx, exp in enumerate(all_exp):
        treader = tokens.WordTextReader(seq_len=exp.seqlen, wordlen=exp.wordlen,
                                        filename=text_filename,
                                        include_special=True, device=device)
        ntrain = int(len(treader) * train_split)
        train_data, val_data = treader.train_val_split(ntrain)

        if exp.seed is not None:
            torch.manual_seed(exp.seed)

        if exp.minicnt:
            train_sampler = RandomSampler(train_data, replacement=True, num_samples=exp.batch * exp.minicnt)
            train_dl = DataLoader(train_data, batch_size=exp.batch, sampler=train_sampler, drop_last=True)

            val_sampler = RandomSampler(val_data, replacement=True, num_samples=exp.batch * exp.minicnt)
            val_dl = DataLoader(val_data, batch_size=exp.batch, sampler=val_sampler, drop_last=True)
        else:
            train_dl = DataLoader(train_data, batch_size=exp.batch, drop_last=True)
            val_dl = DataLoader(val_data, batch_size=exp.batch, drop_last=True)
        
        print(f"  - {len(train_data)=}, {len(val_data)=}")
        print(f"  - {len(train_dl)=}, {len(val_dl)=}")
        print(f"  - {len(next(iter(train_dl))[0])=}")

        fields = exp.to_dict()

        label = ", ".join([f"{key} {val}" for key, val in fields.items()])
        ckpt_filename = basename + "-" + label
        found_ckpt = False
        for maybe_ckpt in Path("runs").iterdir():
            if maybe_ckpt.name.endswith(".ckpt") and maybe_ckpt.name.startswith(ckpt_filename):
                found_ckpt = True
                break

        if found_ckpt:
            print(f"\033[1mskip {ckpt_filename}: already exists\033[0m")
            continue

        exp.tokenizer = treader.tokenizer
        exp.dictionary = treader.dictionary
        exp.vocablen = treader.dictionary.vocab_len
        exp.loss_fn = model.loss_fn(seqlen=exp.seqlen, vocablen=exp.vocablen)
        exp.net = text_experiment.from_experiment(exp, device=device)
        exp.train_dataloader = train_dl
        exp.val_dataloader = val_dl
        exp.label = label

        nparams = sum(p.numel() for p in exp.net.parameters())
        print(f"loaded net with {nparams/1e6:.2f}M params. start training. memory {torch.cuda.memory_allocated()/1e6:.2f}Mb")

        if exp.compile:
            print("compiling...")
            start = datetime.datetime.now()
            exp.net = torch.compile(exp.net)
            end = datetime.datetime.now()
            print(f"  compile took {end - start}")

        time_start = datetime.datetime.now()
        yield exp
        time_end = datetime.datetime.now()
        time_fmt = "%Y%m%d-%H%M%S"
        elapsed = (time_end - time_start).total_seconds()
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed) % 60

        print(f"trained. memory {torch.cuda.memory_allocated()/1e6:.2f}Mb")
        if exp.train_loss_hist is None:
            print(f"didn't train. skipping.")
            continue
        if torch.isnan(exp.train_loss_hist[-1]):
            print(f"nan. skipping.")
            continue

        ckpt_filename += f", elapsed {elapsed:.2f}s, vloss {exp.last_val_loss:.3f}.ckpt"
        checkpoint_path = Path("runs", ckpt_filename)
        checkpoint = exp.state_dict()
        with open(checkpoint_path, "wb") as torch_file:
            print(f"saving {checkpoint_path}...")
            start = datetime.datetime.now()
            torch.save(checkpoint, torch_file)
            end = datetime.datetime.now()
            print(f"  save took {end - start}")

# %%
if __name__ == "__main__":

    exp = TextExperiment(label="foo", seqlen=16, wordlen=1, vocablen=0, nhead=4, nlayers=4, emblen=384, hidlen=384*4, optim_type="sgd", sched_type="StepLR", startlr=1e-3, endlr=13-4, dropout=0.2, batch=1, minicnt=1, epochs=1)
    for i in range(10):
        for exp in gen_experiments("foo", "all_python.txt", [exp]):
            first = next(iter(exp.train_dataloader))
            inputs, truth = first
            print(f"{inputs=}")
            print(f"{truth=}")

# %%


