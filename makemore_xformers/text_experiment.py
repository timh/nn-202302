from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
import datetime

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import tokens
import model
from model import TransformerModel
from tokens import WordTextReader, TextReader

sys.path.insert(0, "..")
from experiment import Experiment
import trainer

@dataclass(kw_only=True)
class TextExperiment(Experiment):
    seqlen: int
    wordlen: int
    vocablen: int = 0
    nhead: int
    nlayers: int
    emblen: int
    hidlen: int

    optim_type: str
    sched_type: str
    startlr: float
    endlr: float
    dropout: float

    batch: int
    minicnt: int
    epochs: int

    flash = "none"
    compile = False
    seed: Optional[int] = None

    tokenizer: tokens.Tokenizer = None
    dictionary: tokens.Dictionary = None

    # from parent
    label: str = None
    net: nn.Module = None
    loss_fn: Callable[[Tensor, Tensor], Tensor] = None
    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None

    """dict for saving with torch.save"""
    def state_dict(self) -> Dict[str, any]:
        fields = [field for field in dir(self)
                  if not field.startswith("_") 
                  and type(getattr(self, field)) in [int, str, float, bool, Tensor, datetime.datetime]]
        res = {field: getattr(self, field) for field in fields}

        res["net"] = self.net.state_dict()
        res["optim"] = self.optim.state_dict()
        res["scheduler"] = self.scheduler.state_dict()
        res["tokenizer"] = self.tokenizer
        res["dictionary"] = self.dictionary
        res["pytorch_version"] = torch.__version__
        res["elapsed"] = (self.ended_at - self.started_at).total_seconds()

        return res

    """descriptive, all string dictionary, for filename generation"""
    def to_dict(self) -> Dict[str, any]:
        fields = ("seqlen wordlen nhead nlayers emblen hidlen "
                  "optim sched startlr endlr "
                  "batch minicnt epochs flash compile").split(" ")

        res: Dict[str, any] = dict()
        for field in fields:
            # shorten some of the fields to avoid > 255 filename length
            attr = {"optim": "optim_type", "sched": "sched_type"}.get(field, field)
            value = getattr(self, attr)

            if field in ["startlr", "endlr"]:
                value = format(value, ".2E")
            elif isinstance(field, float):
                value = format(value, ".4f")
            else:
                value = str(value) 
            res[field] = value
        
        if self.seed is not None:
            res["seed"] = self.seed
        
        return res

def from_experiment(exp: TextExperiment, device = "cpu") -> TransformerModel:
    return TransformerModel(seqlen=exp.seqlen,
                            vocablen=exp.vocablen, emblen=exp.emblen, 
                            nhead=exp.nhead, nlayers=exp.nlayers, hidlen=exp.hidlen, 
                            dropout=exp.dropout,
                            flash=exp.flash,
                            device=device)

# %%
def load_experiment(state_dict: Dict[str, any], device = "cpu") -> TextExperiment:
    # get rid of unwanted prefix. not sure why they're in the checkpoint.
    state_dict = state_dict.copy()
    unwanted = "_orig_mod."
    for field, value in list(state_dict["net"].items()):
        if field.startswith(unwanted):
            state_dict["net"][field[len(unwanted):]] = state_dict["net"].pop(field)

    # fields needed for the TransformerModel
    net_fields = "seqlen vocablen emblen nhead nlayers hidlen dropout flash".split(" ")
    net_args = {field: state_dict[field] for field in net_fields}
    net = TransformerModel(device=device, **net_args)
    net.load_state_dict(state_dict["net"])
    state_dict["net"] = net

    # figure out number of params
    nparams = sum(p.numel() for p in net.parameters())
    descr = ", ".join(f"{field} {value}" for field, value in net_args.items())
    print(f"TransformerModel ({descr}) has {nparams / 1e6:.2f}M params")

    # load optimizer
    optimizer_dict = state_dict["optim"]
    if state_dict["optim_type"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters())
    elif state_dict["optim_type"] == "adamw":
        optimizer = torch.optim.AdamW(net.parameters())
    else:
        raise ValueError(f"unknown {state_dict['optim_type']=}")
    optimizer.load_state_dict(optimizer_dict)

    # load scheduler
    scheduler_dict = state_dict["scheduler"]
    if state_dict["sched_type"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_dict["step_size"])
    elif state_dict["sched_type"] == "nanogpt-cosine":
        scheduler = trainer.NanoGPTCosineScheduler(optimizer, start_lr=scheduler_dict["start_lr"], min_lr=scheduler_dict["min_lr"], 
                                                   warmup_epochs=scheduler_dict["warmup_epochs"], lr_decay_epochs=scheduler_dict["lr_decay_epochs"])
    else:
        raise Exception(f"unknown {state_dict['sched_type']=}")

    state_dict["optim"] = optimizer
    state_dict["scheduler"] = scheduler

    # dataclasses doesn't like passing in optional args to the constructor, at
    # least in this TextExperiment(Experiment) child/parent config. save the
    # optional ones aside and set them after init.
    opt_fields = "flash compile seed nsamples nbatches pytorch_version started_at ended_at".split(" ")
    opt_dict = {field: state_dict.pop(field) for field in opt_fields if field in state_dict}

    exp = TextExperiment(**state_dict)
    for field, value in opt_dict.items():
        setattr(exp, field, value)
    exp.loss_fn = model.loss_fn(exp.seqlen, exp.vocablen)
    return exp

def load_model_and_reader(model_filename: str, text_filename: str, device = "cpu") -> Tuple[TransformerModel, TextReader]:
    state_dict = torch.load(model_filename)
    exp = load_experiment(state_dict, device=device)

    seqlen, wordlen = exp.seqlen, exp.wordlen
    treader = WordTextReader(seq_len=seqlen, wordlen=wordlen, include_special=True, filename=text_filename, device=device)

    return exp.net, treader


if __name__ == "__main__":
    state_dict = torch.load("runs/python-100000_10-seqlen 128, wordlen 2, nhead 4, nlayers 4, emblen 384, hidlen 1536, optim_type adamw, sched_type StepLR, startlr 1.00E-03, endlr 1.00E-04, batch 128, minicnt 2, epochs 10, elapsed 2.85s, vloss 4.878.ckpt")
    exp = load_experiment(state_dict)
    print(f"{exp.optim=}")
    print(f"{exp.scheduler=}")
    print(f"{exp.train_loss_hist[-1]=}")

    net = exp.net
    print("\n".join(state_dict["net"].keys()))
    # print(state_dict["net"].keys())
    # print(list(net.parameters()))


# %%
