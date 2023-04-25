import re
import argparse
import datetime
from typing import Sequence, List, Callable
import types
from pathlib import Path

from experiment import Experiment, LossType, OptimType, SchedType
import checkpoint_util
import train_util

import torch
from torch import Tensor
from torch import nn
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
                exp_val = obj.get(objfield, None)
            else:
                exp_val = getattr(exp, field, None)
            
            if type(exp_val) in [types.FunctionType, types.MethodType]:
                exp_val = exp_val()
            
            exp_val = str(exp_val)
            matcher_val = str(matcher_val)
            
            if field == 'ago':
                if matcher_val.endswith("m"):
                    matcher_val = int(matcher_val[:-1]) * 60
                elif matcher_val.endswith("h"):
                    matcher_val = int(matcher_val[:-1]) * (60 * 60)
                elif matcher_val.endswith("d"):
                    matcher_val = int(matcher_val[:-1]) * (24 * 60 * 60)

            matches = True
            if op in ["=", "=="]:
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
    grad_accum: int
    device: str

    parser: argparse.ArgumentParser

    def __init__(self):
        if torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--batch_size", type=int, default=8)
        self.parser.add_argument("-g", "--grad_accum", type=int, default=0)
        self.parser.add_argument("--no_compile", default=False, action='store_true')
        self.parser.add_argument("--no_amp", dest="use_amp", default=True, action='store_false')
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
    resume_shortcodes: List[str]
    just_show_experiments: bool
    config_file: str

    extra_tag: str

    max_epochs: int
    startlr: float
    endlr: float
    optim_type: OptimType
    sched_type: SchedType
    sched_warmup_epochs: int

    use_best: LossType

    started_at: datetime.datetime

    def __init__(self, basename: str):
        super().__init__()
        self.add_argument("--no_resume", dest='do_resume', action='store_false', default=True)
        self.add_argument("--resume_top_n", type=int, default=0)
        self.add_argument("--resume_shortcodes", type=str, nargs='+', default=[], help="resume only these shortcodes")
        self.add_argument("--just_show_experiments", default=False, action='store_true')
        self.add_argument("-c", "--config_file", required=False)

        self.add_argument("-n", "--max_epochs", type=int, required=True)
        self.add_argument("--use_best", default=None, choices=['tloss', 'vloss'],
                          help="use best (tloss or vloss) checkpoint for resuming instead of the last")
        self.add_argument("-e", "--extra", dest='extra_tag', default=None,
                          help="extra tag added to the experiment")

        self.add_argument("--startlr", type=float, default=1e-3)
        self.add_argument("--endlr", type=float, default=1e-4)
        self.add_argument("--optim_type", choices=["adamw", "sgd"], default="adamw")
        self.add_argument("--sched_type", choices=["nanogpt", "constant", "step"], default="nanogpt")
        self.add_argument("--sched_warmup_epochs", type=int, default=2)

        self.basename = basename
        self.started_at = datetime.datetime.now()

    def build_experiments(self, *,
                          config_exps: List[Experiment] = None,
                          train_dl: DataLoader, val_dl: DataLoader,
                          loss_fn: Callable[[Experiment], Callable[[Tensor, Tensor], Tensor]] = None,
                          resume_net_fn: Callable[[Experiment], nn.Module] = None,
                          init_new_experiment: Callable[[Experiment], None] = None) -> List[Experiment]:
        """
        build experiments by either:
        - loading from config file (-c / --config-file), or
        - resuming the given shortcodes (--resume_shortcodes)

        if loaded from a config file, 
        """
        if not config_exps and not self.resume_shortcodes:
            self.error("must include one of -c (config file) or --resume_shortcodes")

        exps: List[Experiment] = list()
        if len(self.resume_shortcodes):
            exps = checkpoint_util.list_experiments()
            exps = [exp for exp in exps if exp.shortcode in self.resume_shortcodes]
            for exp in exps:
                exp.lazy_net_fn = resume_net_fn
        else:
            for exp in config_exps:
                exp.loss_type = exp.loss_type or "l1"
                if self.extra_tag is not None:
                    exp.extra_tag = self.extra_tag
                    exp.label += f",{exp.extra_tag}"

                if init_new_experiment is not None:
                    init_new_experiment(exp)
            exps = config_exps

        # populate fields needed for running the experiments. none of these fields
        # impact the shortcode, and are purely used run-by-run.
        for exp in exps:
            exp.device = self.device
            exp.lazy_dataloaders_fn = lambda _exp: (train_dl, val_dl)
            if exp.lazy_optim_fn is None:
                exp.lazy_optim_fn = train_util.lazy_optim_fn
            if exp.lazy_sched_fn is None:
                exp.lazy_sched_fn = train_util.lazy_sched_fn
            if exp.loss_fn is None:
                exp.loss_fn = loss_fn(exp)

            run = exp.get_run()
            run.startlr = self.startlr or run.startlr
            run.endlr = self.endlr or run.endlr
            run.sched_warmup_epochs = run.sched_warmup_epochs or self.sched_warmup_epochs
            run.batch_size = self.batch_size
            run.grad_accum = self.grad_accum

            run.optim_type = self.optim_type or run.optim_type
            run.sched_type = self.sched_type or run.sched_type
            run.max_epochs = self.max_epochs or run.max_epochs

            run.use_amp = self.use_amp
            if self.no_compile:
                run.do_compile = False

        if self.do_resume:
            if self.resume_top_n:
                exps = exps[:self.resume_top_n]
                exps_vloss = " ".join([format(exp.best_val_loss, ".3f") for exp in exps])
                exps_tloss = " ".join([format(exp.best_train_loss, ".3f") for exp in exps])
                print(f"resuming top {self.resume_top_n} experiments: vloss = {exps_vloss}, tloss = {exps_tloss}")

            print(f"input to resume:", ",".join([exp.shortcode for exp in exps]))
            exps = checkpoint_util.resume_experiments(exps_in=exps,
                                                      use_best=self.use_best,
                                                      max_epochs=self.max_epochs)

        if self.just_show_experiments:
            import sys
            import json
            for exp in exps:
                # HACK
                exp.start(exp_idx=0)
                md_path = Path("/tmp", f"checkpoints-{self.basename}", f"temp-{exp.shortcode}", "metadata.json")
                md_path.parent.mkdir(exist_ok=True, parents=True)
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
    shortcodes: List[str]
    net_classes: List[str]
    top_n: int
    sort_key: str
    run_dirs: List[Path]

    DEFAULT_SORT_KEY = 'time'

    def __init__(self):
        super().__init__()
        self.add_argument("--pattern", type=str, default=None)
        self.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
        self.add_argument("-sc", "--shortcode", dest='shortcodes', type=str, nargs='+', default=[])
        self.add_argument("-nc", "--net_class", dest='net_classes', type=str, nargs='+', default=[])
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
        
        if self.pattern:
            exps = [exp for exp in exps if self.pattern.match(str(exp.metadata_path))]
        
        if self.net_classes:
            exps = [exp for exp in exps if getattr(exp, 'net_class', None) in self.net_classes]

        if self.shortcodes:
            exps = [exp for exp in exps if exp.shortcode in self.shortcodes]

        if self.sort_key:
            def key_fn(exp: Experiment) -> any:
                if "loss" in self.sort_key:
                    key = self.sort_key
                    if self.sort_key in ["val_loss", "vloss"]:
                        return exp.last_val_loss
                    elif self.sort_key in ["train_loss", "tloss"]:
                        return exp.last_train_loss
                    val = getattr(exp, key, None)
                    if val is None:
                        return 1000.
                    return val
                elif self.sort_key == "time":
                    val = exp.ended_at if exp.ended_at else exp.saved_at
                    return val
                return getattr(exp, self.sort_key)

            exps = sorted(exps, key=key_fn)
        
        if self.top_n:
            exps = exps[:self.top_n]
        
        return exps

