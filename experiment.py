from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Set, Union, Literal
import datetime

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as torchopt
import torch.optim.lr_scheduler as torchsched

# NOTE: pytorch < 2.0.0 has _LRScheduler, where >= has LRScheduler also.
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import base_model

_compile_supported = hasattr(torch, "compile")

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OBJ_FIELDS = "net optim sched".split(" ")
SAME_IGNORE_FIELDS = set('started_at ended_at saved_at elapsed nepochs nbatches nsamples exp_idx device cur_lr'.split())

@dataclass(kw_only=True)
class Experiment:
    label: str = None
    startlr: float = None
    endlr: float = None
    device: str = None
    max_epochs: int = 0
    batch_size: int = 0 # batch size used for training

    # loss function is not lazy generated.
    loss_type: str = ""
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
    # net_class: str = ""

    # functions to generate the optim and sched. set these to
    # trainer.lazy_optim_fn / lazy_sched_fn for reasonable defaults. those
    # defaults use 'optim_type' and 'sched_type' below
    lazy_optim_fn: Callable[['Experiment'], torchopt.Optimizer] = None
    lazy_sched_fn: Callable[['Experiment'], torchsched._LRScheduler] = None
    optim_type: str = ""
    sched_type: str = ""
    sched_warmup_epochs: int = 0

    exp_idx: int = 0
    train_loss_hist: Tensor = None                 # (nepochs * batch_size,)
    val_loss_hist: List[Tuple[int, Tensor]] = None # List(nepochs) X Tensor(1,)
    lastepoch_train_loss: float = None             # loss for last *epoch* of training (not just a batch)
    lastepoch_val_loss: float = None               # loss for last epoch of validation

    nepochs: int = 0    # epochs trained so far
    nsamples: int = 0   # samples trained against so far
    nbatches: int = 0   # batches (steps) trained against so far

    last_train_in: Tensor = None
    last_train_out: Tensor = None
    last_train_truth: Tensor = None
    last_val_in: Tensor = None
    last_val_out: Tensor = None
    last_val_truth: Tensor = None

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None
    saved_at: datetime.datetime = None
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
    def metadata_dict(self, update_saved_at = True) -> Dict[str, any]:
        ALLOWED = [int, str, float, bool, datetime.datetime]
        def vals_for(obj: any, ignore_fields: List[str] = None) -> Dict[str, any]:
            ires: Dict[str, any] = dict()
            for field in dir(obj):
                if field == 'cur_lr' and obj == self and self.sched is None:
                    continue
                if field.startswith('_') or (ignore_fields and field in ignore_fields):
                    continue

                val = getattr(obj, field)
                if type(val) == list and len(val) and type(val[0]) in ALLOWED:
                    pass
                elif not type(val) in ALLOWED:
                    continue

                if isinstance(val, datetime.datetime):
                    val = val.strftime(TIME_FORMAT)

                ires[field] = val
            return ires

        res: Dict[str, any] = vals_for(self)
        if self.net is not None:
            if type(self.net).__name__ == 'OptimizedModule':
                res['net_class'] = type(self.net._orig_mod).__name__
            else:
                res['net_class'] = type(self.net).__name__
            if hasattr(self.net, 'metadata_dict'):
                for nfield, nvalue in self.net.metadata_dict().items():
                    field = f"net_{nfield}"
                    res[field] = nvalue

        if self.sched is not None:
            res['sched_args'] = vals_for(self.sched)
            # res['sched_class'] = type(self.sched).__name__

        if self.optim is not None:
            res['optim_args'] = vals_for(self.optim)
            # res['optim_class'] = type(self.optim).__name__

        now = datetime.datetime.now()
        if update_saved_at:
            res['saved_at'] = now.strftime(TIME_FORMAT)

        if self.started_at:
            res['elapsed'] = (now - self.started_at).total_seconds()

        return res
    
    """
    Returns fields for torch.save model_dict, not including those in metadata_dict
    """
    def model_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in dir(self):
            if field.startswith("_"):
                continue

            if field == 'cur_lr' and self.sched is None:
                # @property cur_lr will throw an exception if we call it without
                # self.sched set.
                continue

            val = getattr(self, field, None)
            if (field not in OBJ_FIELDS and 
                type(val) not in [str, int, float, bool, datetime.datetime, Tensor, list, dict]):
                # print(f"skip {field}: {type(val)=}")
                continue

            if field in OBJ_FIELDS and val is not None:
                classfield = field + "_class"
                if type(val).__name__ == 'OptimizedModule':
                    classval = type(val._orig_mod).__name__
                else:
                    classval = type(val).__name__
                res[classfield] = classval

                if hasattr(val, 'model_dict'):
                    val = val.model_dict()
                else:
                    val = val.state_dict()

            res[field] = val
        return res

    """
    Fills in fields from the given model_dict.
    :param: fill_self_from_objargs: if set, all attributes in 'net_args', 'sched_args',
    'optim_args'
    """
    def load_model_dict(self, model_dict: Dict[str, any]) -> 'Experiment':
        for field, value in model_dict.items():
            # if field.startswith('net_') and field != 'net_class':
            #     continue
            if field in ['nparams', 'cur_lr'] or field in OBJ_FIELDS:
                # 'label' is excluded cuz it was used for construction.
                # 'nparams' is excluded cuz it's a @parameter
                continue

            if field == 'curtime':
                field = 'saved_at'

            if field in ['started_at', 'ended_at', 'saved_at'] and isinstance(value, str):
                value = datetime.datetime.strptime(value, TIME_FORMAT)
            setattr(self, field, value)

        return self
    
    """
    a.k.a. is_sameish
    RETURNS: 
            bool or (bool, List[str], List[str]) if return_tuple is set

               bool: if values in two experiments are the same, minus ignored fields
          List[str]: field names that are the same
          List[str]: field names that are different
    """
    def is_same(self, other: 'Experiment', 
                extra_ignore_fields: Set[str] = None, return_tuple = False) -> Union[bool, Tuple[bool, Set[str], Set[str]]]:
        ours = self.metadata_dict()
        other = other.metadata_dict()
        all_fields = set(list(ours.keys()) + list(other.keys()))

        ignore = list(SAME_IGNORE_FIELDS)
        ignore = ignore + [field for field in all_fields if field.startswith("lastepoch_")]
        if extra_ignore_fields:
            ignore.extend(extra_ignore_fields)
        fields = all_fields - set(ignore)

        same = True
        fields_same: Set[str] = set()
        fields_diff: Set[str] = set()
        for field in fields:
            ourval = ours.get(field, None)
            otherval = other.get(field, None)
            if ourval == otherval:
                fields_same.add(field)
            else:
                fields_diff.add(field)
                same = False
        
        if return_tuple:
            return same, fields_same, fields_diff
        return same

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
            print(f"compiling...")
            start = datetime.datetime.now()
            self.net = torch.compile(self.net)
            end = datetime.datetime.now()
            print(f"  compile took {end - start}")
        
        if self.optim is None:
            self.optim = self.lazy_optim_fn(self)
        if self.sched is None:
            self.sched = self.lazy_sched_fn(self)

        if self.train_dataloader is None:
            self.train_dataloader, self.val_dataloader = self.lazy_dataloaders_fn(self)
