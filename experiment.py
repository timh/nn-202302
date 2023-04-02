import dataclasses
from typing import Callable, Tuple, Dict, List, Set, Optional, Literal
from collections import OrderedDict
import types
import datetime
from pathlib import Path
import model_util
import hashlib

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as torchopt
import torch.optim.lr_scheduler as torchsched

# NOTE: pytorch < 2.0.0 has _LRScheduler, where >= has LRScheduler also.
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

_compile_supported = hasattr(torch, "compile")

OBJ_FIELDS = 'net optim sched'.split()
# SYNTHETIC_FIELDS = ('net_class optim_class optim_args sched_class sched_args '
#                     'elapsed elapsed_str shortcode '
#                     'best_train_loss best_train_epoch best_val_loss best_val_epoch '
#                     'last_train_loss last_val_loss '
#                     'created_at').split()

LossType = Literal['tloss', 'vloss', 'train_loss', 'val_loss']

@dataclasses.dataclass(kw_only=True)
class ExpRun:
    """The completed (max_epochs) of training"""
    max_epochs: int = 0
    finished: bool = False

    batch_size: int = 0
    do_compile: bool = _compile_supported
    use_amp: bool = False

    startlr: float = None
    endlr: float = None
    optim_type: str = ""
    sched_type: str = ""
    sched_warmup_epochs: int = 0

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None

    """values representing """
    checkpoint_nepochs: int = 0
    checkpoint_nbatches: int = 0
    checkpoint_nsamples: int = 0
    checkpoint_at: datetime.datetime = None
    checkpoint_path: Path = None

    resumed_from: Path = None

    def metadata_dict(self) -> Dict[str, any]:
        return model_util.md_obj(self)
    
    def copy(self) -> 'ExpRun':
        res = ExpRun()
        for field in RUN_FIELDS:
            val = getattr(self, field)
            setattr(res, field, val)
        return res

RUN_FIELDS = model_util.md_obj_fields(ExpRun())

# NOTE backwards compatibility
class ExpResume(ExpRun): pass

@dataclasses.dataclass(kw_only=True)
class Experiment:
    label: str = None
    device: str = None

    """epochs trained so far. 0-based. A "1" means the experiment has completed epoch 1."""
    nepochs: int = 0    # total epochs trained so far
    nbatches: int = 0   # batches (steps) trained so far
    nsamples: int = 0   # samples trained so far
    created_at: datetime.datetime = None
    saved_at: datetime.datetime = None

    # loss function is not lazy generated.
    loss_type: str = ""
    loss_fn: Callable[[Tensor, Tensor], Tensor] = None

    """set to True to stop this experiment"""
    skip: bool = False

    # for each net, optim, sched, dataloaders, either pass the object, or the lazy
    # generator object
    net: nn.Module = None
    optim: torchopt.Optimizer = None
    sched: torchsched._LRScheduler = None
    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None
    _nparams: int = 0

    net_args: Dict[str, any] = dataclasses.field(default_factory=dict)
    sched_args: Dict[str, any] = dataclasses.field(default_factory=dict)
    optim_args: Dict[str, any] = dataclasses.field(default_factory=dict)

    # functions to generate the net and dataloaders
    lazy_net_fn: Callable[['Experiment'], nn.Module] = None
    lazy_dataloaders_fn: Callable[['Experiment'], Tuple[DataLoader, DataLoader]] = None
    # net_class: str = ""

    # functions to generate the optim and sched. set these to
    # trainer.lazy_optim_fn / lazy_sched_fn for reasonable defaults. those
    # defaults use 'optim_type' and 'sched_type' below
    lazy_optim_fn: Callable[['Experiment'], torchopt.Optimizer] = None
    lazy_sched_fn: Callable[['Experiment'], torchsched._LRScheduler] = None

    exp_idx: int = 0
    train_loss_hist: List[float]           = dataclasses.field(default_factory=list)  # List(nepochs X tloss)
    val_loss_hist: List[Tuple[int, float]] = dataclasses.field(default_factory=list)  # List(nepochs X (epoch, vloss))
    lr_hist: List[float]                   = dataclasses.field(default_factory=list)  # List(nepochs X lr)

    runs: List[ExpRun] = dataclasses.field(default_factory=list)
    metadata_path: Path = None

    def __post_init__(self):
        self.created_at = datetime.datetime.now()

    """
    fields that should be saved in metadata.
    """
    def md_fields(self) -> Set[str]:
        self_fields = model_util.md_obj_fields(self)
        self_fields = [field for field in self_fields
                       if not field.endswith("_fn")
                       and not field.endswith("_dataloader")
                       and field not in OBJ_FIELDS
                       and field not in {'sched_args', 'optim_args'}
                       and not field.endswith("_class")]
        return self_fields

    """
    return consistently-ordered metadata dict for net/net_args.
    """
    def md_net_args(self) -> OrderedDict[str, any]:
        if self.net is not None and hasattr(self.net, 'metadata_dict'):
            net_args = model_util.md_obj(self.net.metadata_dict())
            net_args['class'] = _get_classname(self.net)
        else:
            net_args: OrderedDict = model_util.md_obj(self.net_args)
            net_args.pop('class', None)
            net_args['class'] = self.net_args.get('class', None)
        return net_args

    """
    Return the fields that are used to do identity comparison, e.g., for shortcode.
    These are returned in sorted order.
    
    This is a subset of md_fields: it excludes arguments that can vary from run
    to run (e.g., self.device) that don't affect the identity of the experiment.
    """
    def id_fields(self) -> List[str]:
        skip_fields = set('device skip runs exp_idx runs nepochs nbatches nsamples metadata_path created_at saved_at'.split())

        fields = [field for field in self.md_fields()
                  if field not in skip_fields and not field.endswith("_hist") and not field.endswith("_loss")]

        return sorted(fields)

    """
    return fields and values used for identity comparison, e.g., for shortcode
    """
    def id_values(self) -> Dict[str, any]:
        res: Dict[str, any] = OrderedDict()

        # NOTE self.id_fields() returns fields in sorted order.
        for field in self.id_fields():
            if field == 'net_args':
                val = self.md_net_args()
            else:
                val = model_util.md_obj(getattr(self, field))

            res[field] = val

        return res

    """
    compare the identity of two experiments and return any values that are
    different.
    Returns: list of tuples (field name, self value, other value)
    """
    def id_diff(self, other: 'Experiment') -> List[Tuple[str, any, any]]:
        diffs: List[Tuple[str, any, any]] = list()

        self_values = self.id_values()
        other_values = other.id_values()

        all_fields = set(self_values.keys()) | set(other_values.keys())
        for field in all_fields:
            self_val = self_values.get(field)
            other_val = other_values.get(field)
            if self_val != other_val:
                diffs.append((field, self_val, other_val))
        
        return diffs

    """
    compute a (consistent across runs) short hash for the experiment, based
    on fields that don't change over time.
    """
    @property
    def shortcode(self) -> str:
        id_str = str(self.id_values())

        length = 6
        vocablen = 26
        hash_digest = hashlib.sha1(bytes(id_str, "utf-8")).digest()
        code = ""
        for hash_byte, _ in zip(hash_digest, range(length)):
            val = hash_byte % vocablen
            code += chr(ord('a') + val)
        return code

    @property
    def created_at_short(self) -> str:
        return self.created_at.strftime(model_util.TIME_FORMAT_SHORT)

    def saved_at_relative(self) -> str:
        if self.saved_at is None:
            return ""

        now = datetime.datetime.now()
        return model_util.duration_str((now - self.saved_at).total_seconds())
    
    @property
    def checkpoint_nepochs(self) -> int:
        return self.get_run().checkpoint_nepochs
    
    """
    get run by path, checkpoint_nepochs, minimum of cp_nepochs_min, or by best
    loss of the given type.
    """    
    def get_run(self, 
                cp_path: Path = None, 
                cp_nepochs: int = None, cp_nepochs_min: int = None,
                loss_type: LossType = None) -> Optional[ExpRun]:
        if not any([cp_path, cp_nepochs, cp_nepochs_min, loss_type]):
            if len(self.runs) == 0:
                self.runs.append(ExpRun())
            return self.runs[-1]

        # by path.
        if cp_path is not None:
            for run in self.runs:
                if run.checkpoint_path == cp_path:
                    return run
            return None

        # by epochs.
        if cp_nepochs or cp_nepochs_min:
            for run in self.runs:
                if cp_nepochs and run.checkpoint_nepochs == cp_nepochs:
                    return run
                if cp_nepochs_min and run.checkpoint_nepochs >= cp_nepochs_min:
                    return run
            return None

        # by best loss
        if loss_type in ['train_loss', 'tloss']:
            # print(f"{len(self.train_loss_hist)=}")

            # checkpoint refers to the END of an epoch. so a checkpoint @ 1 means 
            # there's 1 epoch of data, i.e., [0]. subtract 1.
            runs_sorted = \
                sorted(self.runs, 
                       key=lambda run: self.train_loss_hist[run.checkpoint_nepochs - 1])
            return runs_sorted[0]

        val_hist = sorted(self.val_loss_hist, key=lambda tup: tup[1])
        for epoch, _vloss in val_hist:
            for run in self.runs:
                if run.checkpoint_nepochs >= epoch:
                    return run
        
        return None

    @property
    def cur_lr(self):
        if self.sched is None:
            raise Exception(f"{self}: cur_lr called, but self.sched hasn't been set yet")
        return self.sched.get_last_lr()[0]
    
    def nparams(self) -> int:
        # allow returning nparams from metadata-saved value if net isn't loaded.
        if self.net is not None:
            return sum(p.numel() for p in self.net.parameters())
        return self._nparams

    def elapsed(self) -> float:
        total_elapsed: float = 0
        for run in self.runs:
            if not run.started_at:
                continue
            # elif run.ended_at:
            #     diff = (run.ended_at - run.started_at)
            elif run.checkpoint_at:
                diff = (run.checkpoint_at - run.started_at)
            else:
                continue
            total_elapsed += diff.total_seconds()
        return total_elapsed

    def elapsed_str(self) -> str:
        return model_util.duration_str(self.elapsed())

    @property
    def best_train_loss(self) -> float:
        if not self.train_loss_hist:
            return 0.0
        return sorted(self.train_loss_hist)[0]

    @property
    def best_train_epoch(self) -> float:
        if not self.train_loss_hist:
            return 0
        hist = zip(self.train_loss_hist, range(len(self.train_loss_hist)))
        return sorted(hist)[0][1]

    @property
    def best_val_loss(self) -> float:
        if not self.val_loss_hist:
            return 0.0
        return sorted(self.val_loss_hist, key=lambda tup: tup[1])[0][1]
    
    @property
    def best_val_epoch(self) -> int:
        if not self.val_loss_hist:
            return 0
        return sorted(self.val_loss_hist, key=lambda tup: tup[1])[0][0]
    
    @property
    def last_train_loss(self) -> float:
        if not self.train_loss_hist:
            return 0.0
        return self.train_loss_hist[-1]

    @property
    def last_val_loss(self) -> float:
        if not self.val_loss_hist:
            return 0.0
        return self.val_loss_hist[-1][1]
    
    """
    Delegate to current ExpRun for any fields that it has.
    e.g., Experiment.batch_size becomes Experiment.get_run().batch_size

    NOTE unfortunately we don't have these attributes showing up in 
    dir(Experiment).
    """
    def __getattr__(self, name: str) -> any:
        if name.startswith("net_"):
            name = name[4:]
            if self.net is not None:
                if name == 'class':
                    return _get_classname(self.net)
                return getattr(self.net, name)
        
            return self.net_args.get(name)

        if name in RUN_FIELDS:
            return getattr(self.get_run(), name)

        raise AttributeError(f"missing {name}")
    
    def __setattr__(self, name: str, val: any):
        if name in RUN_FIELDS:
            return setattr(self.get_run(), name, val)
        return super().__setattr__(name, val)
    
    """
    returns fields suitable for the metadata file
    """
    def metadata_dict(self, update_saved_at = True) -> Dict[str, any]:
        self_fields = self.md_fields()
        res: Dict[str, any] = model_util.md_obj(self, only_fields=self_fields)
        res.pop('net_args', None)

        if update_saved_at:
            self.saved_at = datetime.datetime.now()

        if self.saved_at:
            res['saved_at'] = model_util.md_scalar(self.saved_at)
            res['saved_at_relative'] = self.saved_at_relative()

        res['elapsed'] = self.elapsed()
        res['elapsed_str'] = self.elapsed_str()
        res['best_train_loss'] = self.best_train_loss
        res['best_train_epoch'] = self.best_train_epoch
        res['best_val_loss'] = self.best_val_loss
        res['best_val_epoch'] = self.best_val_epoch
        res['shortcode'] = self.shortcode
        res['nparams'] = self.nparams()

        res['net_args'] = self.md_net_args()
        
        # copy fields from the last run to the root level
        if len(self.runs):
            res['runs'] = list()
            for run in self.runs:
                res['runs'].append(run.metadata_dict())

        return res
    
    """
    Returns fields for torch.save model_dict, not including those in metadata_dict
    """
    def model_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in self.md_fields():
            val = getattr(self, field)
            if field == 'runs':
                val = [item.metadata_dict() for item in val]
                continue

            if val is None:
                pass

            if not model_util.md_obj_allowed(val) and not isinstance(type, Tensor):
                print(f"ignore {field=} {type(val)=}")
                continue

            res[field] = val

        for obj_name in OBJ_FIELDS:
            obj = getattr(self, obj_name, None)

            obj_dict: Dict[str, any]
            if hasattr(obj, 'model_dict'):
                obj_dict = obj.model_dict()
            else:
                obj_dict = obj.state_dict()
            res[obj_name] = obj_dict
            res[obj_name + "_class"] = _get_classname(obj)
            obj_dict['class'] = _get_classname(obj)

        return res

    """
    Fills in fields from the given model_dict.
    """
    def load_model_dict(self, model_dict: Dict[str, any]) -> 'Experiment':
        # sort the fields so that we load RUN_FIELDS fields later. 
        # @backcompat.
        fields = [field for field in model_dict.keys() if field not in RUN_FIELDS]
        fields.extend([field for field in model_dict.keys() if field in RUN_FIELDS])

        self.net_args = dict()
        self.sched_args = dict()
        self.optim_args = dict()

        exp_label = model_dict.get("label")
        for field in fields:
            value = model_dict.get(field)

            if field == 'runs':
                self.runs = list()
                for rundict in value:
                    # print("rundict", str(rundict))
                    run = ExpRun()
                    for rfield, rval in rundict.items():
                        if rfield in {'saved_at_relative', 'elapsed', 'elapsed_str'}:
                            continue

                        if rfield.endswith("_at") and isinstance(rval, str):
                            rval = datetime.datetime.strptime(rval, model_util.TIME_FORMAT)
                        elif rfield in {'resumed_from', 'checkpoint_path'} and isinstance(rval, str):
                            rval = Path(rval)

                        # back-compat: moved nepochs/nbatches/nsamples back to 
                        # Experiment, instead of ExpRun.
                        if rfield in {'nepochs', 'nbatches', 'nsamples', 'created_at', 'saved_at'}:
                            setattr(self, rfield, rval)
                            continue

                        setattr(run, rfield, rval)
                    self.runs.append(run)
                continue

            # load the value of nparams (which is generated) in _nparams, so it
            # doesn't conflict with the nparams() method, and we can persist it
            # even when self.net is None.
            if field == 'nparams':
                field = '_nparams'

            if field.endswith("_at") and isinstance(value, str):
                value = datetime.datetime.strptime(value, model_util.TIME_FORMAT)

            # skip setting any field that is actually a method, function, or property.
            if type(getattr(type(self), field, None)) in [types.MethodType, types.FunctionType, property]:
                continue

            # 'net' -> 'net_args'
            if field in OBJ_FIELDS:
                field = field + "_args"

            setattr(self, field, value)

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
        self.get_run().started_at = datetime.datetime.now()

    def end(self):
        cur_run = self.get_run()
        cur_run.ended_at = datetime.datetime.now()
        cur_run.finished = True
    
    def end_cleanup(self):
        if self.net is not None:
            for p in self.net.parameters():
                p.detach().cpu()
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
        
        if self.optim is None:
            self.optim = self.lazy_optim_fn(self)
        if self.sched is None:
            self.sched = self.lazy_sched_fn(self)

        if self.train_dataloader is None:
            self.train_dataloader, self.val_dataloader = self.lazy_dataloaders_fn(self)

def _get_classname(obj: any) -> str:
    if type(obj).__name__ == 'OptimizedModule':
        return type(obj._orig_mod).__name__
    return type(obj).__name__

