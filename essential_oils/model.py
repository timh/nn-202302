import random
from typing import Tuple, Dict, List, DefaultDict
from dataclasses import dataclass
from collections import defaultdict
import datetime
import string

import torch, torch.optim
import torch.nn as nn

@dataclass
class Molecule:
    name: str
    value: float

    def __repr__(self) -> str:
        return f"{self.name} @ {self.value}"

@dataclass
class Compound:
    name: str
    molecules: List[Molecule]

    def __init__(self, name: str):
        self.name = name
        self.molecules = list()
    
    def __repr__(self) -> str:
        mols_str = ", ".join(map(repr, self.molecules))
        return f"{self.name} ({mols_str})"

@dataclass
class Oil:
    name: str
    compounds: List[Compound]

@dataclass
class Reading:
    molecules: List[Molecule]

# mine
def DistanceLoss(out, truth):
    return torch.abs((truth - out)).mean()

# https://stats.stackexchange.com/questions/438728/mean-absolute-percentage-error-returning-nan-in-pytorch
def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / (target + 1e-6)))
def RPDLoss(output, target):
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))    

@dataclass
class Config:
    compounds: List[Compound]
    all_comp_names: List[str]
    compname_idx: Dict[str, int]
    all_mol_names: List[str]
    molname_idx: Dict[str, int]

    net: nn.Module = None
    loss_fn: nn.Module = None
    optim: torch.optim.Optimizer = None
    device: str = ""

    def __init__(self, s: str):
        compounds = Config.parse_compounds(s)
        self.compounds = compounds

        self.all_comp_names = sorted([comp.name for comp in compounds])
        self.compname_idx = {name: idx for idx, name in enumerate(self.all_comp_names)}

        self.all_mol_names = [mol.name for comp in compounds for mol in comp.molecules]
        self.all_mol_names = sorted(list(set(self.all_mol_names)))
        self.molname_idx = {name: idx for idx, name in enumerate(self.all_mol_names)}

    def parse_compounds(s: str) -> List[Compound]:
        res: List[Compound] = list()
        curcomp: Compound = None
        for line in s.splitlines():
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()

            if not line:
                continue

            if line.endswith(":"):
                collname = line[:-1]
                curcomp = Compound(collname)
                res.append(curcomp)
                continue

            if ":" not in line:
                print(f"can't parse '{line}'")
                continue

            mname, mvalue = [w.strip() for w in line.split(":")]
            if "%" in mvalue:
                mvalue = float(mvalue[:-1]) / 100.0
            else:
                mvalue = float(mvalue)
            
            part = Molecule(mname, mvalue)
            curcomp.molecules.append(part)
        
        return res

    def setup(self, num_hidden: int, hidden_size: int, lr: float):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        num_molecules = len(self.all_mol_names)
        num_compounds = len(self.all_comp_names)

        self.net = nn.Sequential()
        self.net.append(nn.Linear(num_molecules, hidden_size))
        for _ in range(num_hidden):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.BatchNorm1d(hidden_size))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_size, num_compounds))
        self.net.append(nn.Softmax(dim=1))
        self.net = self.net.to(self.device)

        # experimentation leads me to believe that the weights need to be biggified.
        # with torch.no_grad():
        #     for m in self.net.modules():
        #         if not isinstance(m, nn.Linear):
        #             continue
        #         m.weight *= 10.0

        # self.loss_fn = nn.L1Loss().to(self.device)
        # self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # self.loss_fn = nn.MultiLabelSoftMarginLoss().to(self.device)
        # self.loss_fn = nn.BCELoss().to(self.device)

        # self.loss_fn = DistanceLoss()
        self.loss_fn = RPDLoss
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=lr)

def gen_examples(cfg: Config, num_batches: int, batch_size: int):
    res: List[torch.Tensor, torch.Tensor] = list()
    for example_idx in range(num_batches):
        truth_ingreds = torch.zeros((batch_size, len(cfg.all_comp_names), ))
        measurement_mols = torch.zeros((batch_size, len(cfg.all_mol_names), ))

        for batch_idx in range(batch_size):
            num_compounds = torch.randint(0, len(cfg.compounds), (1,)).item() + 1

            compounds_shuffled = cfg.compounds.copy()
            random.shuffle(compounds_shuffled)

            # print(f"example {example_idx}:")

            total_ingred = 0.0
            total_mol = 0.0
            for ingred_idx in range(num_compounds):
                ingred = compounds_shuffled[ingred_idx]
                ingred_used = torch.rand((1,)) * 10.0
                ingred_used = (torch.abs(ingred_used) + 0.1).item()

                total_ingred += ingred_used
                truth_ingreds[batch_idx][cfg.compname_idx[ingred.name]] += ingred_used

                # print(f"  used {ingred.name} @ {ingred_used:.3f}")

                # print(f"{example_idx=}, {ingred_idx=}: {ingred=}")
                for m in ingred.molecules:
                    val = m.value * ingred_used
                    measurement_mols[batch_idx][cfg.molname_idx[m.name]] += val
                    total_mol += val

            truth_ingreds[batch_idx] /= total_ingred
            measurement_mols[batch_idx] /= total_mol

            # print()
            # for name in cfg.all_mol_names:
            #     val = measurement_mols[cfg.molname_idx[name]]
            #     print(f"   mol {name} @ {val:.3f}")
            # print()
            # print(f"  measurement_mols = {measurement_mols}")
            # print(f"  truth_ingreds = {truth_ingreds}")
        
        measurement_mols = measurement_mols.to(cfg.device)
        truth_ingreds = truth_ingreds.to(cfg.device)
        res.append((measurement_mols, truth_ingreds))
    
    return res

def train(cfg: Config, 
          num_epochs: int,
          data_train: torch.Tensor, data_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    train_loss_hist = torch.zeros((num_epochs,))
    val_loss_hist = torch.zeros_like(train_loss_hist)
    val_dist_hist = torch.zeros_like(val_loss_hist)

    last_print = datetime.datetime.now()
    last_print_nsamples = 0
    total_nsamples_sofar = 0
    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch, (measurement, truth) in enumerate(data_train):
            out = cfg.net(measurement)
            loss = cfg.loss_fn(out, truth)

            lossval = loss
            if lossval.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {lossval} at epoch {epoch}, batch {batch} -- returning!")
                return train_loss_hist[:epoch], val_loss_hist[:epoch], val_dist_hist[:epoch]
            train_loss += lossval.item()

            loss.backward()
            cfg.optim.step()

            total_nsamples_sofar += len(measurement)
        
        train_loss /= len(data_train)

        with torch.no_grad():
            val_loss = 0.0
            val_dist = 0.0
            for batch, (measurement, truth) in enumerate(data_val):
                val_out = cfg.net(measurement)

                lossval = cfg.loss_fn(val_out, truth)
                if lossval.isnan():
                    print(f"!! validation loss {lossval} at epoch {epoch}, batch {batch} -- returning!")
                    return train_loss_hist[:epoch], val_loss_hist[:epoch], val_dist_hist[:epoch]

                distval = DistanceLoss(val_out, truth)
                if distval.isnan():
                    print(f"!! validation distance {distval} at epoch {epoch}, batch {batch} -- returning!")
                    return train_loss_hist[:epoch], val_loss_hist[:epoch], val_dist_hist[:epoch]

                val_loss += lossval.item()
                val_dist += distval.item()

            val_loss /= len(data_val)
            val_dist /= len(data_val)

        
        now = datetime.datetime.now()
        if (now - last_print) >= datetime.timedelta(seconds=5) or (epoch == num_epochs - 1):
            timediff = (now - last_print)
            nsamples_diff = float(total_nsamples_sofar - last_print_nsamples)
            samples_per_sec = nsamples_diff / timediff.total_seconds()

            print(f"epoch {epoch+1}/{num_epochs}: train loss {train_loss:.5f}, val loss {val_loss:.5f}, val dist {val_dist:.5f} | samp/sec {samples_per_sec:.3f}")
            last_print = now
            last_print_nsamples = total_nsamples_sofar

        train_loss_hist[epoch] = train_loss
        val_loss_hist[epoch] = val_loss
        val_dist_hist[epoch] = val_dist
      
    return train_loss_hist, val_loss_hist, val_dist_hist

def show_result(cfg: Config, examples: List[Tuple[torch.Tensor, torch.Tensor]]):
    def show_one(logits: torch.Tensor, truth: torch.Tensor):
        inf_str, truth_str, diff_str = "inference", "truth", "diff"
        part_str, part_len = "part", max([len(c) for c in cfg.all_comp_names])
        part_str = part_str.rjust(part_len, " ")

        print(f"{part_str} | {inf_str:10} | {truth_str:10} | {diff_str:10}")
        for n in range(len(logits)):
            infval, truthval = logits[n].item() * 100, truth[n].item() * 100
            compname = cfg.all_comp_names[n]
            compname = compname.rjust(part_len, " ")
            if truthval:
                diff = ((infval - truthval) / truthval) * 100.0
            else:
                diff = 0.0
            print(f"{compname} | {infval:9.3f}% | {truthval:9.3f}% | {diff:8.3f}%")

    with torch.no_grad():
        for batch in examples:
            measurements_batch, truth_batch = batch
            logits_batch = cfg.net(measurements_batch)
            for i in range(len(logits_batch)):
                logits = logits_batch[i]
                truth = truth_batch[i]
                show_one(logits, truth)
