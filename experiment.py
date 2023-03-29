import dataclasses
from typing import Callable, Tuple, Dict, List, Set, Union, Literal
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

import base_model

_compile_supported = hasattr(torch, "compile")

OBJ_FIELDS = "net optim sched".split(" ")

SAME_IGNORE_DEFAULT = \
    set('started_at ended_at saved_at saved_at_relative elapsed elapsed_str '
        'resumed_at nepochs nbatches nsamples exp_idx device cur_lr nparams '
        'train_loss_hist val_loss_hist '
        'best_train_loss best_train_epoch best_val_loss best_val_epoch '
        'last_train_loss last_val_loss'.split())
SAME_IGNORE_RESUME = \
    set('max_epochs batch_size label '
        'optim_type optim_args startlr endlr lr_hist '
        'sched_type sched_args sched_warmup_epochs '
        'do_compile use_amp finished'.split()) | SAME_IGNORE_DEFAULT

@dataclasses.dataclass(kw_only=True)
class ExpRun:
    nepochs: int = 0    # epochs trained so far
    nbatches: int = 0   # batches (steps) trained against so far
    nsamples: int = 0   # samples trained against so far
    max_epochs: int = 0
    finished: bool = False

    batch_size: int = 0 # batch size used for training
    do_compile = _compile_supported

    startlr: float = None
    endlr: float = None
    optim_type: str = ""
    sched_type: str = ""
    sched_warmup_epochs: int = 0

    started_at: datetime.datetime = None
    ended_at: datetime.datetime = None
    saved_at: datetime.datetime = None

    resumed_from: Path = None

    def saved_at_relative(self) -> str:
        if self.saved_at is None:
            return ""

        now = datetime.datetime.now()
        return model_util.duration_str((now - self.saved_at).total_seconds())
    
    def metadata_dict(self) -> Dict[str, any]:
        res = model_util.md_obj(self)
        res['saved_at_relative'] = self.saved_at_relative()
        return res
    
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
    train_loss_hist: List[float] = None           # List(nepochs X float)
    val_loss_hist: List[Tuple[int, float]] = None # List(nepochs X float)
    lr_hist: List[float] = None                   # List(nepochs X float)

    runs: List[ExpRun] = dataclasses.field(default_factory=list)

    created_at: datetime.datetime = None

    def __post_init__(self):
        self.created_at = datetime.datetime.now()
    
    def _md_fields(self) -> Set[str]:
        self_fields = model_util.md_obj_fields(self)
        self_fields = [field for field in self_fields
                       if not field.endswith("_fn")
                       and not field.endswith("_dataloader")
                       and not field == "exp_idx"
                       and not field in OBJ_FIELDS]
        return self_fields
    
    """
    compute a (consistent across runs) short hash for the experiment, based
    on fields that don't change over time.
    """
    @property
    def shortcode(self) -> str:
        fields: List[str] = list()
        for field in self._md_fields():
            if (field in {'device', 'skip', 'created_at', 'runs'} or 
                field.endswith("_hist") or 
                field.endswith("_args")):
                continue
            # print(f"{field=}")
            fields.append(field)

        # put in a list (not dict) to maintain 
        values = [(field, getattr(self, field)) for field in fields]
        values_str = str(values)

        length = 6
        vocablen = 26
        hash_digest = hashlib.sha1(bytes(values_str, "utf-8")).digest()
        code = ""
        for hash_byte, _ in zip(hash_digest, range(length)):
            val = hash_byte % vocablen
            code += chr(ord('a') + val)
        return code

    @property
    def cur_lr(self):
        if self.sched is None:
            raise Exception(f"{self}: cur_lr called, but self.sched hasn't been set yet")
        return self.sched.get_last_lr()[0]
    
    def nparams(self) -> int:
        if self.net is None:
            raise Exception(f"{self} not initialized yet")
        return sum(p.numel() for p in self.net.parameters())

    def cur_run(self) -> ExpRun:
        if not len(self.runs):
            self.runs.append(ExpRun())
        return self.runs[-1]

    def elapsed(self) -> float:
        total_elapsed: float = 0
        for run in self.runs:
            if not run.started_at:
                continue
            # elif run.ended_at:
            #     diff = (run.ended_at - run.started_at)
            elif run.saved_at:
                diff = (run.saved_at - run.started_at)
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
    
    def saved_at_relative(self) -> str:
        return self.cur_run().saved_at_relative()

    """
    delegate to current ExpRun for any fields that it has.
    e.g., Experiment.nepochs becomes Experiment.cur_run().nepochs

    NOTE unfortunately we don't have these attributes showing up in 
    dir(Experiment).
    """
    def __getattr__(self, name: str) -> any:
        if name in RUN_FIELDS:
            return getattr(self.cur_run(), name)
        raise AttributeError(f"missing {name}")
    
    def __setattr__(self, name: str, val: any):
        if name in RUN_FIELDS:
            # raise Exception(f"don't call this: {name=}\n  {val=}")
            # print(f"delegate set({name=}, {val=}) to cur_run")
            return setattr(self.cur_run(), name, val)
        return super().__setattr__(name, val)
    
    """
    returns fields suitable for the metadata file
    """
    def metadata_dict(self, update_saved_at = True) -> Dict[str, any]:
        self_fields = self._md_fields()
        res: Dict[str, any] = model_util.md_obj(self, only_fields=self_fields)
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
            res['sched_args'] = model_util.md_obj(self.sched)
            # res['sched_class'] = type(self.sched).__name__

        if self.optim is not None:
            res['optim_args'] = model_util.md_obj(self.optim)
            # res['optim_class'] = type(self.optim).__name__

        if update_saved_at:
            self.saved_at = datetime.datetime.now()

        if self.saved_at:
            res['saved_at'] = self.saved_at.strftime(model_util.TIME_FORMAT)
            res['saved_at_relative'] = self.saved_at_relative()

        res['elapsed'] = self.elapsed()
        res['elapsed_str'] = self.elapsed_str()
        res['best_train_loss'] = self.best_train_loss
        res['best_train_epoch'] = self.best_train_epoch
        res['best_val_loss'] = self.best_val_loss
        res['best_val_epoch'] = self.best_val_epoch
        res['shortcode'] = self.shortcode

        if self.net is not None:
            res['nparams'] = self.nparams()

        # copy fields from the last run to the root level
        if len(self.runs):
            res['runs'] = list()
            for run in self.runs:
                res['runs'].append(run.metadata_dict())
            # res.update(self.runs[-1].metadata_dict())

        return res
    
    """
        return list of strings with short(er) field names.

        if split_label_on is set, return a list of strings for the label, instead of
        just a string.
    """
    # TODO: this is not very good.
    def describe(self, extra_field_map: Dict[str, str] = None, include_loss = True) -> List[Union[str, List[str]]]:
        field_map = {'startlr': 'startlr'}
        if include_loss:
            field_map['best_train_loss'] = 'best_tloss'
            field_map['best_val_loss'] = 'best_vloss'
            field_map['last_train_loss'] = 'last_tloss'
            field_map['last_val_loss'] ='last_vloss'

        if extra_field_map:
            field_map.update(extra_field_map)

        exp_fields = dict()
        for field, short in field_map.items():
            val = getattr(self, field, None)
            if isinstance(val, types.MethodType):
                val = val()
            if val is None:
                continue
            if 'lr' in field:
                val = format(val, ".1E")
            elif isinstance(val, float):
                val = format(val, ".3f")
            exp_fields[short] = str(val)

        strings = [f"{field} {val}" for field, val in exp_fields.items()]

        comma_parts = self.label.split(",")
        for comma_idx, comma_part in enumerate(comma_parts):
            dash_parts = comma_part.split("-")
            if len(dash_parts) == 1:
                strings.append(comma_part)
                continue

            for dash_idx in range(len(dash_parts)):
                if dash_idx != len(dash_parts) - 1:
                    dash_parts[dash_idx] += "-"
            strings.append(dash_parts)
        
        return strings

    """
    Returns fields for torch.save model_dict, not including those in metadata_dict
    """
    def model_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in self._md_fields():
            val = getattr(self, field)
            if not model_util.md_type_allowed(val) and not isinstance(type, Tensor):
                print(f"ignore {field=} {type(val)=}")
                continue
            res[field] = val

        for field in OBJ_FIELDS:
            val = getattr(self, field, None)
            if val is not None:
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
        # sort the fields so that we load RUN_FIELDS fields later. 
        # @backcompat.
        fields = [field for field in model_dict.keys() if field not in RUN_FIELDS]
        fields.extend([field for field in model_dict.keys() if field in RUN_FIELDS])

        start_created_at_id = id(self.created_at)

        runs_present = False
        if 'runs' in fields:
            runs_present = True

        for field in fields:
            value = model_dict.get(field)
            if field == 'cur_lr':
                continue

            # curval = getattr(self, field, None)
            # if type(curval) in [types.MethodType, types.FunctionType, property]:
            #     continue
            if type(getattr(type(self), field, None)) in [types.MethodType, types.FunctionType, property]:
                continue

            # backwards compatibility for older saves.
            if field == 'resumed_at':
                # load resume objects, including converting 2-field -> 3-field. conversion
                # leaves 'value' in a state of being a 3-field dict.
                if len(value) and len(value[0]) == 2:
                    resumed_from = model_dict.get('resumed_from', "")
                    new_value: List[Dict[str, any]] = list()
                    for nepochs, timestamp in value:
                        new_dict = {'nepochs': nepochs, 
                                    'timestamp': timestamp,
                                    'path': resumed_from}
                        new_value.append(new_dict)
                    value = new_value

                for resume_dict in value:
                    nepochs = resume_dict['nepochs']
                    timestamp = resume_dict['timestamp']
                    path = resume_dict['path']
                    if isinstance(timestamp, str):
                        timestamp = datetime.datetime.strptime(timestamp, model_util.TIME_FORMAT)
                    if isinstance(path, str):
                        path = Path(path)

                    # instantiate a new run for this resume, with just 3 fields.
                    new_run = ExpRun(nepochs=nepochs, started_at=timestamp, resumed_from=path)
                    cur_run = self.cur_run()
                    for rfield in RUN_FIELDS:
                        # populate fields beyond the 3 with the values in the current run.
                        if rfield in {'nepochs', 'started_at', 'resumed_from'}:
                            continue
                        setattr(new_run, rfield, getattr(cur_run, rfield))

                    self.runs.append(new_run)
                continue

            if field in ['started_at', 'ended_at', 'saved_at'] and isinstance(value, str):
                # set these values on the current run
                value = datetime.datetime.strptime(value, model_util.TIME_FORMAT)
                setattr(self.cur_run(), field, value)
                continue

            elif field == 'runs':
                for rundict in value:
                    run = ExpRun()
                    for rfield, rval in rundict.items():
                        if rfield in {'saved_at_relative', 'elapsed', 'elapsed_str'}:
                            continue

                        if rfield.endswith("_at") and isinstance(rval, str):
                            rval = datetime.datetime.strptime(rval, model_util.TIME_FORMAT)
                        elif rfield == 'resumed_from' and isinstance(rval, str):
                            rval = Path(rval)
                        setattr(run, rfield, rval)
                    self.runs.append(run)
                continue

            # TODO
            if field == 'created_at' and isinstance(value, str):
                value = datetime.datetime.strptime(value, model_util.TIME_FORMAT)

            if field in RUN_FIELDS:
                # backwards compatibility.
                if not runs_present:
                    curval = getattr(self.cur_run(), field, None)
                    setattr(self.cur_run(), field, value)
                continue

            setattr(self, field, value)

        # backwards compatibility. convert old field names to new.
        old_fields = [field for field in fields if field.startswith("lastepoch_")]
        for old_field in old_fields:
            new_field = old_field.replace("lastepoch_", "last_")
            # print(f"{old_field} -> {new_field}")
            val = getattr(self, old_field)
            delattr(self, old_field)
            if new_field == 'last_train_loss':
                self.train_loss_hist.append(val)
            elif new_field == 'last_val_loss':
                self.val_loss_hist.append((self.nepochs, val))
            else:
                setattr(self, new_field, val)

        if id(self.created_at) == start_created_at_id and self.runs[0].started_at:
            self.created_at = self.runs[0].started_at
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
                ignore_fields: Set[str] = SAME_IGNORE_DEFAULT, 
                ignore_loss_fields = True,
                return_tuple = False) -> Union[bool, Tuple[bool, Set[str], Set[str]]]:

        ignore_fields = set(ignore_fields)
        fields_same: Set[str] = set()
        fields_diff: Set[str] = set()

        def _obj_same(our_obj: Union[Experiment, ExpRun], other_obj: Union[Experiment, ExpRun]) -> bool:
            our_md = our_obj.metadata_dict()
            other_md = other_obj.metadata_dict()
            fields = set(list(our_md.keys()) + list(other_md.keys()))
            fields = fields - ignore_fields

            same = True
            for field in fields:
                # keep going through fields, even if same is False, to continue
                # all same/diff fields
                if field == 'runs':
                    for our_run, other_run in zip(our_obj.runs, other_obj.runs):
                        run_same = _obj_same(our_run, other_run)
                        if not run_same:
                            same = False
                            break
                    continue

                elif ignore_loss_fields and 'loss' in field:
                    continue

                ourval = our_md.get(field, None)
                otherval = other_md.get(field, None)
                if ourval == otherval:
                    fields_same.add(field)
                else:
                    fields_diff.add(field)
                    same = False

            return same

        res = _obj_same(self, other)
        if return_tuple:
            return res, fields_same, fields_diff
        return res

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
        self.optim = self.lazy_optim_fn(self)
        self.sched = self.lazy_sched_fn(self)
        self.cur_run().started_at = datetime.datetime.now()

        if self.train_loss_hist is None:
            self.train_loss_hist = list()

        if self.val_loss_hist is None:
            self.val_loss_hist = list()
        
        if self.lr_hist is None:
            self.lr_hist = list()
    
    """
        (other) experiment has already been setup with loss function, lazy
        functions, etc, but has no state.
        (self) experiment was loaded from a checkpoint, specified in cp_path.
    """
    def prepare_resume(self, cp_path: Path, new_exp: 'Experiment'):
        now = datetime.datetime.now()

        self.loss_fn = new_exp.loss_fn
        self.train_dataloader = new_exp.train_dataloader
        self.val_dataloader = new_exp.val_dataloader

        self.lazy_net_fn = new_exp.lazy_net_fn
        self.lazy_sched_fn = new_exp.lazy_sched_fn
        self.lazy_optim_fn = new_exp.lazy_optim_fn

        # copy fields from the new experiment
        resume = self.cur_run().copy()
        resume.resumed_from = cp_path
        resume.started_at = now
        resume.ended_at = None
        resume.max_epochs = new_exp.max_epochs
        resume.finished = False
        resume.batch_size = new_exp.batch_size
        resume.do_compile = new_exp.do_compile
        resume.startlr = new_exp.startlr
        resume.endlr = new_exp.endlr
        resume.optim_type = new_exp.optim_type
        resume.sched_type = new_exp.sched_type
        resume.sched_warmup_epochs = new_exp.sched_warmup_epochs
        self.runs.append(resume)
    
    def end(self):
        cur_run = self.cur_run()
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
