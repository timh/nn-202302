from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Union, Literal
import datetime

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
    net_class: str = ""

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
    def metadata_dict(self) -> Dict[str, any]:
        def vals_for(obj: any, ignore_fields: List[str] = None) -> Dict[str, any]:
            ires: Dict[str, any] = dict()
            for field in dir(obj):
                if field.startswith('_') or (ignore_fields and field in ignore_fields):
                    continue

                val = getattr(obj, field)
                if not type(val) in [int, str, float, bool, datetime.datetime]:
                    continue
                if isinstance(val, datetime.datetime):
                    val = val.strftime(TIME_FORMAT)

                ires[field] = val
            return ires

        res: Dict[str, any] = vals_for(self, ['cur_lr'])
        if self.net is not None:
            res['net_args'] = vals_for(self.net)
            res['net_class'] = type(self.net).__name__
        if self.sched is not None:
            res['sched_args'] = vals_for(self.sched)
            res['sched_class'] = type(self.sched).__name__
        if self.optim is not None:
            res['optim_args'] = vals_for(self.optim)
            res['optim_class'] = type(self.optim).__name__
        
        now = datetime.datetime.now()
        res['saved_at'] = now.strftime(TIME_FORMAT)
        res['elapsed'] = (now - self.started_at).total_seconds()

        return res
    
    """
    Returns fields for torch.save state_dict, including those in metadata_dict
    """
    def state_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in dir(self):
            if field.startswith("_"):
                continue

            val = getattr(self, field, None)
            if (field not in OBJ_FIELDS and 
                type(val) not in [str, int, float, bool, datetime.datetime, Tensor]):
                # print(f"skip {field}: {type(val)=}")
                continue

            if field in OBJ_FIELDS and val is not None:
                classfield = field + "_class"
                classval = type(val).__name__
                res[classfield] = classval
                val = val.state_dict()

            res[field] = val
        return res

    """
    Fills in fields from the given state_dict.
    :param: fill_self_from_objargs: if set, all attributes in 'net_args', 'sched_args',
    'optim_args'
    """
    def load_state_dict(self, state_dict: Dict[str, any], 
                        fill_self_from_subobj_names: Union[bool, List[Literal['net', 'sched', 'optim']]] = False) -> 'Experiment':
        for field, value in state_dict.items():
            if field in ['nparams'] or field in OBJ_FIELDS:
                # 'label' is excluded cuz it was used for construction.
                # 'nparams' is excluded cuz it's a @parameter
                continue

            if field == 'curtime':
                field = 'saved_at'

            if field in ['started_at', 'ended_at', 'saved_at'] and isinstance(value, str):
                value = datetime.datetime.strptime(value, TIME_FORMAT)
            setattr(self, field, value)

        # possibly set our own fields based on fields that are in net_args,
        # sched_args, and optim_args.

        # build include_subobj_names based on whether a bool (all) or List 
        # (explicitly listed) subargs should be included.
        subobj_names = ['net', 'sched', 'optim']
        if not fill_self_from_subobj_names:
            include_subobj_names = []
        elif isinstance(fill_self_from_subobj_names, bool):
            include_subobj_names = subobj_names
        else:
            include_subobj_names = fill_self_from_subobj_names

        # now fill fields, if any, based on include_subobj_names
        for subobj_name in include_subobj_names:
            subobj_args_field = f"{subobj_name}_args"
            if not subobj_args_field in state_dict:
                continue
            for subfield, subvalue in state_dict[subobj_args_field].items():
                setattr(self, subfield, subvalue)
        
        return self

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
        self.optim = self.lazy_optim_fn(self) if self.optim is None else self.optim
        self.sched = self.lazy_sched_fn(self) if self.sched is None else self.sched

        if self.train_dataloader is None:
            self.train_dataloader, self.val_dataloader = self.lazy_dataloaders_fn(self)
