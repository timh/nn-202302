import re
import argparse
import datetime
from typing import Sequence, List, Set, Tuple, Callable, Literal
from pathlib import Path

from experiment import Experiment, LossType
import checkpoint_util
import train_util

import torch
from torch.utils.data import DataLoader

RE_OP = re.compile(r"([\w\._]+)\s*([=<>!~]+)\s*(.+)")
def gen_attribute_matcher(matchers: Sequence[str]) -> Callable[[Experiment], bool]:
    def fn(exp: Experiment) -> bool:
        for matcher in matchers:
            field, op, matcher_val = RE_OP.match(matcher).groups()
            if field.startswith("net.") or field.startswith("sched.") or field.startswith("optim."):
                objname, objfield = field.split(".", maxsplit=1)
                objname = objname + "_args"
                obj = getattr(exp, objname)
                exp_val = str(obj.get(objfield, None))
            else:
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
    do_resume: bool
    resume_top_n: int
    just_show_experiments: bool

    max_epochs: int
    startlr: float
    endlr: float
    sched_warmup_epochs: int
    extra_tag: str

    use_best: LossType

    started_at: datetime.datetime

    def __init__(self, basename: str):
        super().__init__()
        self.add_argument("-n", "--max_epochs", type=int, required=True)
        self.add_argument("--startlr", type=float, default=1e-3)
        self.add_argument("--endlr", type=float, default=1e-4)
        self.add_argument("--sched_warmup_epochs", type=int, default=None)
        self.add_argument("--resume", dest='do_resume', action='store_true', default=False)
        self.add_argument("--resume_top_n", type=int, default=0)
        self.add_argument("--just_show_experiments", default=False, action='store_true')
        self.add_argument("--use_best", default=None, choices=['tloss', 'vloss'],
                          help="use best (tloss or vloss) checkpoint for each run, instead of the default, last")
        self.add_argument("-e", "--extra", dest='extra_tag', default=None,
                          help="extra tag added to the experiment")

        self.basename = basename
        self.started_at = datetime.datetime.now()

    def build_experiments(self, exps_in: List[Experiment],
                          train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
        for exp in exps_in:
            exp.loss_type = exp.loss_type or "l1"
            exp.startlr = self.startlr or exp.startlr
            exp.endlr = self.endlr or exp.endlr
            exp.sched_warmup_epochs = exp.sched_warmup_epochs or self.sched_warmup_epochs
            exp.batch_size = self.batch_size
            exp.device = self.device
            exp.optim_type = exp.optim_type or "adamw"
            exp.sched_type = exp.sched_type or "nanogpt"
            exp.max_epochs = exp.max_epochs or self.max_epochs

            if self.extra_tag is not None:
                exp.extra_tag = self.extra_tag
                exp.label += f",{exp.extra_tag}"

            exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
            if exp.lazy_optim_fn is None:
                exp.lazy_optim_fn = train_util.lazy_optim_fn
            if exp.lazy_sched_fn is None:
                exp.lazy_sched_fn = train_util.lazy_sched_fn
            
            if self.no_compile:
                exp.do_compile = False
            elif exp.do_compile:
                # exp.label += ",compile"
                pass

            if self.use_amp:
                exp.use_amp = True
                # exp.label += ",useamp"

        if self.do_resume:
            exps = checkpoint_util.resume_experiments(exps_in=exps_in,
                                                      use_best=self.use_best,
                                                      max_epochs=self.max_epochs)
            if self.resume_top_n:
                exps = exps[:self.resume_top_n]
                exps_vloss = " ".join([format(exp.best_val_loss, ".3f") for exp in exps])
                exps_tloss = " ".join([format(exp.best_train_loss, ".3f") for exp in exps])
                print(f"resumed top {self.resume_top_n} experiments: vloss = {exps_vloss}, tloss = {exps_tloss}")
        else:
            exps = exps_in

        for i, exp in enumerate(exps):
            print(f"{i + 1}. {exp.created_at_short}-{exp.shortcode} | {exp.nepochs} epochs | {exp.label}")
        print()

        if self.just_show_experiments:
            import sys
            import json
            for exp in exps:
                # HACK
                exp.start(exp_idx=0)
                md_path = Path("/tmp", f"checkpoints-{self.basename}", f"temp-{exp.shortcode}", "metadata.json")
                md_path.parent.mkdir(exist_ok=True)
                checkpoint_util.save_metadata(exp, md_path)

                desc_path = Path(md_path.parent, "id_values.json")
                id_values = exp.id_values()
                with open(desc_path, "w") as file:
                    json.dump(id_values, file, indent=2)

                print(f"  saved {md_path} & {desc_path}")
                exp.end()
                exp.end_cleanup()
            sys.exit(0)

        return exps

class QueryConfig(BaseConfig):
    pattern: re.Pattern
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

    def parse_args(self) -> 'QueryConfig':
        super().parse_args()
        if self.pattern:
            self.pattern = re.compile(self.pattern)
        self.run_dirs = [Path(run_dir) for run_dir in self.run_dirs]
        return self

    def list_experiments(self) -> List[Experiment]:
        exps: List[Experiment] = list()
        for run_dir in self.run_dirs:
            one_exps = checkpoint_util.list_experiments(runs_dir=run_dir)
            exps.extend(one_exps)

        if self.attribute_matchers:
            matcher_fn = gen_attribute_matcher(self.attribute_matchers)
            exps = [exp for exp in exps if matcher_fn(exp)]

        if self.sort_key:
            def key_fn(exp: Experiment) -> any:
                if "loss" in self.sort_key:
                    key = self.sort_key
                    if self.sort_key in ["val_loss", "vloss"]:
                        return exp.best_val_loss
                    elif self.sort_key in ["train_loss", "tloss"]:
                        return exp.best_train_loss
                    return getattr(exp, key)
                elif self.sort_key == "time":
                    val = exp.ended_at if exp.ended_at else exp.saved_at
                    return val
                return getattr(exp, self.sort_key)

            exps = sorted(exps, key=key_fn)
        
        if self.top_n:
            exps = exps[:self.top_n]
        
        return exps

