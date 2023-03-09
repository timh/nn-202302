from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import datetime
from pathlib import Path
import json

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as torchopt
import torch.optim.lr_scheduler as torchsched

# NOTE: pytorch < 2.0.0 has _LRScheduler, where >= has LRScheduler also.
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

_compile_supported = hasattr(torch, "compile")

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OBJ_FIELDS = "net optim sched".split(" ")

@dataclass(kw_only=True)
class Experiment:
    label: str
    startlr: float = None
    endlr: float = None
    device: str = None
    max_epochs: int = 0
    batch_size: int = 0 # batch size used for training

    # loss function is not lazy generated.
    loss_fn: Callable[[Tensor, Tensor], Tensor] = None
    do_compile = _compile_supported

    """set to True to stop this experiment"""
    skip: bool = False

    # for each net, optim, sched, dataloaders, either pass the object, or the lazy
    # generator object
    net: nn.Module = None
    optim: torchopt.Optimizer = None
    sched: torchsched._LRScheduler = None
    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None

    # functions to generate the net and dataloaders
    lazy_net_fn: Callable[['Experiment'], nn.Module] = None
    lazy_dataloaders_fn: Callable[['Experiment'], Tuple[DataLoader, DataLoader]] = None

    # functions to generate the optim and sched. set these to
    # trainer.lazy_optim_fn / lazy_sched_fn for reasonable defaults. those
    # defaults use 'optim_type' and 'sched_type' below
    lazy_optim_fn: Callable[['Experiment'], torchopt.Optimizer] = None
    lazy_sched_fn: Callable[['Experiment'], torchsched._LRScheduler] = None
    optim_type: str = ""
    sched_type: str = ""

    exp_idx: int = 0
    train_loss_hist: Tensor = None          # (nepochs * batch_size,)
    val_loss_hist: Tensor = None            # (nepochs,)
    lastepoch_train_loss: float = None      # loss for last *epoch* of training (not just a batch)
    lastepoch_val_loss: float = None        # loss for last epoch of validation

    nepochs = 0    # epochs trained so far
    nsamples = 0   # samples trained against so far
    nbatches = 0   # batches (steps) trained against so far

    last_train_in: Tensor = None
    last_train_out: Tensor = None
    last_train_truth: Tensor = None
    last_val_in: Tensor = None
    last_val_out: Tensor = None
    last_val_truth: Tensor = None

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None
    elapsed: float = None

    @property
    def cur_lr(self):
        if self.sched is None:
            raise Exception(f"{self}: cur_lr called, but self.sched hasn't been set yet")
        return self.sched.get_last_lr()[0]
    
    def nparams(self) -> int:
        if self.net is None:
            raise Exception(f"{self} not initialized yet")
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    """
    returns fields suitable for the metadata file
    """
    def metadata_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()

        for field in dir(self):
            if field.startswith("_") or field == "cur_lr":
                continue

            val = getattr(self, field)
            if not type(val) in [int, str, float, bool, datetime.datetime]:
                continue
            if isinstance(val, datetime.datetime):
                val = val.strftime(TIME_FORMAT)

            res[field] = val

        res["curtime"] = datetime.datetime.now().strftime(TIME_FORMAT)
        
        return res
    
    """
    Returns fields for torch.save state_dict, including those in metadata_dict
    """
    def state_dict(self) -> Dict[str, any]:
        res = self.metadata_dict()
        for field in OBJ_FIELDS:
            val = getattr(self, field, None)
            if val is not None:
                classfield = field + "_class"
                classval = type(val).__name__
                res[classfield] = classval
                val = val.state_dict()

            res[field] = val
        return res

    """
    Loads from either a state_dict or metadata_dict.
    The experiment will not be fully formed if it's loaded from a metadata_dict.
    """
    @staticmethod
    def new_from_dict(state_dict: Dict[str, any]) -> 'Experiment':
        exp = Experiment(label=state_dict["label"])
        for field, value in state_dict.items():
            if field in ["label", "nparams"] or field in OBJ_FIELDS:
                continue
            if field in ["started_at", "ended_at", "curtime"] and isinstance(value, str):
                value = datetime.datetime.strptime(value, TIME_FORMAT)
            if field != "label":
                setattr(exp, field, value)
        return exp

    """
    Get Experiment ready to train: validate fields and lazy load any objects if needed.
    """
    def start(self, exp_idx: int):
        if self.loss_fn is None:
            raise ValueError(f"{self}: no loss_fn set")
        
        if not self.device:
            raise ValueError(f"{self}: no device set")

        self._start_net_optim_sched_dl()

        # now get ready to train
        self.exp_idx = exp_idx
        self.lastepoch_val_loss = 0.0
        self.lastepoch_train_loss = 0.0
        self.optim = self.lazy_optim_fn(self)
        self.sched = self.lazy_sched_fn(self)
        self.started_at = datetime.datetime.now()
    
    def end(self):
        self.ended_at = datetime.datetime.now()
        self.elapsed = (self.ended_at - self.started_at).total_seconds()
        if self.net is not None:
            self.net.cpu()

        self.net = None
        self.optim = None
        self.train_dataloader = None
        self.val_dataloader = None

    def _start_net_optim_sched_dl(self):
        def check(*names: List[str]):
            if all(getattr(self, name, None) is None for name in names):
                names_str = ", ".join(names)
                raise ValueError(f"{self}: none of {names} set")

        check('net', 'lazy_net_fn')
        check('optim', 'lazy_optim_fn')
        check('sched', 'lazy_sched_fn')
        check('train_dataloader', 'lazy_dataloaders_fn')

        self.net = self.lazy_net_fn(self) if self.net is None else self.net
        self.net = self.net.to(self.device)
        if self.do_compile:
            self.net = torch.compile(self.net)
        self.optim = self.lazy_optim_fn(self) if self.optim is None else self.optim
        self.sched = self.lazy_sched_fn(self) if self.sched is None else self.sched

        if self.train_dataloader is None:
            self.train_dataloader, self.val_dataloader = self.lazy_dataloaders_fn(self)

"""
Load Experiment: metadata only.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_experiment_metadata(json_path: Path) -> Experiment:
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)
    return Experiment.new_from_dict(metadata)

"""
Save experiment .ckpt and .json.
"""
def save_metadata(exp: Experiment, json_path: Path):
    metadata_dict = exp.metadata_dict()
    with open(json_path, "w") as json_file:
        json.dump(metadata_dict, json_file)

def save_ckpt_and_metadata(exp: Experiment, ckpt_path: Path, json_path: Path):
    obj_fields_none = {field: (getattr(exp, field, None) is None) for field in OBJ_FIELDS}
    if any(obj_fields_none.values()):
        raise Exception(f"refusing to save {ckpt_path}: some needed fields are None: {obj_fields_none=}")

    state_dict = exp.state_dict()
    with open(ckpt_path, "wb") as ckpt_file:
        torch.save(state_dict, ckpt_file)
    
    save_metadata(exp, json_path)
