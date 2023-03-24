import re
import argparse
import datetime
from typing import Sequence, List, Tuple, Callable
from pathlib import Path

from experiment import Experiment
import checkpoint_util
import train_util

import torch
from torch.utils.data import DataLoader

RE_OP = re.compile(r"([\w_]+)\s*([=<>!~]+)\s*(.+)")
def gen_attribute_matcher(matchers: Sequence[str]) -> Callable[[Experiment], bool]:
    def fn(exp: Experiment) -> bool:
        for matcher in matchers:
            field, op, matcher_val = RE_OP.match(matcher).groups()
            exp_val = str(getattr(exp, field, None))

            matches = True
            if op == "=":
                matches = exp_val == matcher_val
            elif op == "!=":
                matches = exp_val != matcher_val
            elif op == ">":
                matches = float(exp_val) > float(matcher_val)
            elif op == "<":
                matches = float(exp_val) < float(matcher_val)
            elif op == "~":
                matches = matcher_val in exp_val
            elif op == "!~":
                matches = matcher_val not in exp_val
            else:
                raise Exception(f"unknown {op=} for {field=} {matcher_val=}")
            
            if not matches:
                return False
        return True
    return fn

class BaseConfig(argparse.Namespace):
    no_compile: bool
    use_amp: bool
    batch_size: int
    device: str

    parser: argparse.ArgumentParser

    def __init__(self):
        if torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--batch_size", type=int, default=8)
        self.parser.add_argument("--no_compile", default=False, action='store_true')
        self.parser.add_argument("--amp", dest="use_amp", default=False, action='store_true')
        self.parser.add_argument("--device", default=default_device)

    def parse_args(self) -> 'BaseConfig':
        return self.parser.parse_args(namespace=self)

    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)
    
    def error(self, *args, **kwargs):
        return self.parser.error(*args, **kwargs)

class TrainerConfig(BaseConfig):
    no_timestamp: bool
    do_resume: bool
    resume_top_n: int

    max_epochs: int
    startlr: float
    endlr: float

    started_at: datetime.datetime

    def __init__(self, basename: str):
        super().__init__()
        self.add_argument("-n", "--max_epochs", type=int, required=True)
        self.add_argument("--startlr", type=float, default=1e-3)
        self.add_argument("--endlr", type=float, default=1e-4)
        self.add_argument("--no_timestamp", default=False, action='store_true', help="debugging: don't include a timestamp in runs/ subdir")
        self.add_argument("--resume", dest='do_resume', action='store_true', default=False)
        self.add_argument("--resume_top_n", type=int, default=0)

        self.basename = basename
        self.started_at = datetime.datetime.now()
    
    @property
    def log_dirname(self) -> str:
        res = f"runs/{self.basename}_{self.max_epochs:03}"
        if not self.no_timestamp:
            timestr = self.started_at.strftime("%Y%m%d-%H%M%S")
            res += f"_{timestr}"
        return res

    def build_experiments(self, exps_in: List[Experiment],
                          train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
        for exp in exps_in:
            exp.loss_type = exp.loss_type or "l1"

            exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
            if exp.lazy_optim_fn is None:
                exp.lazy_optim_fn = train_util.lazy_optim_fn
            if exp.lazy_sched_fn is None:
                exp.lazy_sched_fn = train_util.lazy_sched_fn
            
            exp.device = self.device
            exp.optim_type = exp.optim_type or "adamw"
            exp.sched_type = exp.sched_type or "nanogpt"
            exp.max_epochs = exp.max_epochs or self.max_epochs

            if self.no_compile:
                exp.do_compile = False
            elif exp.do_compile:
                # exp.label += ",compile"
                pass

            if self.use_amp:
                exp.use_amp = True
                # exp.label += ",useamp"

        if self.do_resume:
            exps = checkpoint_util.resume_experiments(exps_in=exps_in, max_epochs=self.max_epochs)
            if self.resume_top_n:
                # exps = sorted(exps, key=lambda exp: exp.lastepoch_val_loss)
                exps = sorted(exps, key=lambda exp: exp.lastepoch_train_loss)
                exps = exps[:self.resume_top_n]
                exps_vloss = " ".join([format(exp.lastepoch_val_loss, ".3f") for exp in exps])
                exps_tloss = " ".join([format(exp.lastepoch_train_loss, ".3f") for exp in exps])
                print(f"resumed top {self.resume_top_n} experiments: vloss = {exps_vloss}, tloss = {exps_tloss}")
        else:
            exps = exps_in

        for i, exp in enumerate(exps):
            print(f"#{i + 1} {exp.label} nepochs={exp.nepochs}")
        print()

        return exps

class QueryConfig(BaseConfig):
    pattern: str
    attribute_matchers: List[str]
    top_n: int
    sort_key: str
    run_dirs: List[Path]
    dedup_runs: bool

    DEFAULT_SORT_KEY = 'time'

    def __init__(self):
        super().__init__()
        self.add_argument("-p", "--pattern", type=str, default=None)
        self.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
        self.add_argument("--top_n", type=int, default=None)
        self.add_argument("-s", "--sort", dest='sort_key', default=self.DEFAULT_SORT_KEY)
        self.add_argument("--run_dir", dest='run_dirs', default=["runs"], nargs="+")
        self.add_argument("--dedup_runs", default=False, action='store_true',
                          help="for any checkpoint that was resumed, don't return its ancestors")

    def parse_args(self) -> 'QueryConfig':
        super().parse_args()
        if self.pattern:
            self.pattern = re.compile(self.pattern)
        self.run_dirs = [Path(run_dir) for run_dir in self.run_dirs]
        return self

    def list_checkpoints(self, dedup_runs: bool = None) -> List[Tuple[Path, Experiment]]:
        if dedup_runs is None:
            dedup_runs = self.dedup_runs

        checkpoints: List[Tuple[Path, Experiment]] = list()
        for run_dir in self.run_dirs:
            cps = \
                checkpoint_util.find_checkpoints(runs_dir=run_dir,
                                                 attr_matchers=self.attribute_matchers,
                                                 only_paths=self.pattern)
            checkpoints.extend(cps)

        if dedup_runs:
            roots = checkpoint_util.find_resume_roots(checkpoints)
            checkpoints = [checkpoints[offspring[-1]] for root, offspring in roots.items()]
        
        if self.sort_key:
            def key_fn(cp: Tuple[Path, Experiment]) -> any:
                path, exp = cp
                if "loss" in self.sort_key:
                    key = self.sort_key
                    if self.sort_key in ["val_loss", "vloss"]:
                        key = "lastepoch_val_loss"
                    elif self.sort_key in ["train_loss", "tloss"]:
                        key = "lastepoch_train_loss"
                    return getattr(exp, key)
                elif self.sort_key == "time":
                    val = exp.ended_at if exp.ended_at else exp.saved_at
                    return val
                return getattr(exp, self.sort_key)

            checkpoints = sorted(checkpoints, key=key_fn)
        
        if self.top_n:
            checkpoints = checkpoints[:self.top_n]
        
        return checkpoints

