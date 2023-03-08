from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import datetime

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as torchopt
import torch.optim.lr_scheduler as torchsched

# NOTE: pytorch < 2.0.0 has _LRScheduler, where >= has LRScheduler also.
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

_compile_supported = hasattr(torch, "compile")

NATIVE_FIELDS = ("label startlr endlr max_epochs device do_compile "
                 "last_train_in last_train_out last_train_truth last_val_in last_val_out last_val_truth "
                 "train_loss_hist last_train_loss last_val_loss started_at ended_at elapsed "
                 "nepochs nsamples nbatches").split(" ")
STATEDICT_FIELDS = "net sched optim".split(" ")

@dataclass(kw_only=True)
class Experiment:
    label: str
    startlr: float = None
    endlr: float = None
    device: str = None
    max_epochs: int = 0

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
    train_loss_hist: Tensor = None
    last_train_loss: float = None
    val_loss_hist: Tensor = None
    last_val_loss: float = None

    nepochs = 0    # epochs trained so far
    nsamples = 0   # samples trained against so far
    nbatches = 0   # batches trained against so far

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

    def state_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in NATIVE_FIELDS:
            res[field] = getattr(self, field, None)
        res["curtime"] = datetime.datetime.now()
        
        for field in STATEDICT_FIELDS:
            val = getattr(self, field, None)
            if val is not None:
                classfield = field + "_class"
                classval = str(type(val))
                res[classfield] = classval
                val = val.state_dict()

            res[field] = val

        return res

    @staticmethod
    def new_from_state_dict(state_dict: Dict[str, any]) -> 'Experiment':
        exp = Experiment(label=state_dict["label"])
        for field in NATIVE_FIELDS:
            if field != "label":
                value = state_dict.get(field, None)
                setattr(exp, field, value)
        curtime = state_dict.get("curtime", None)
        ended_at = state_dict.get("ended_at", None)
        if curtime is not None and ended_at is None:
            exp.ended_at = curtime
        return exp


    """
    get Experiment ready to train: validate fields and lazy load any objects if needed.
    """
    def start(self, exp_idx: int):
        if self.loss_fn is None:
            raise ValueError(f"{self}: no loss_fn set")
        
        if not self.device:
            raise ValueError(f"{self}: no device set")

        self._start_net_optim_sched_dl()

        # now get ready to train
        self.exp_idx = exp_idx
        self.train_loss_hist = torch.zeros((self.max_epochs,))
        self.val_loss_hist = torch.zeros_like(self.train_loss_hist)
        self.last_val_loss = 0.0
        self.last_train_loss = 0.0
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

