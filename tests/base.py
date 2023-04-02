import unittest
from typing import Tuple, List, Dict, Mapping, Any
from pathlib import Path
import tempfile
import datetime
import sys

import torch
from torch import nn
from torch import Tensor

from experiment import Experiment
import base_model
import checkpoint_util
from model_util import TIME_FORMAT_SHORT

class DumbNet(base_model.BaseModel):
    _metadata_fields = 'one two'.split()
    _model_fields = 'one two p'.split()

    def __init__(self, one: any, two: any):
        super().__init__()
        self.one = one
        self.two = two
        self.p = nn.Parameter(torch.zeros((4, 4)))
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs

class DumbSched:
    def state_dict(self) -> Dict[str, any]:
        return dict()

    def step(self):
        pass

    def get_cur_lr(self) -> float:
        return (1.0, 1.0)
    
    def get_last_lr(self) -> float:
        return (1.0, 1.0)
    

class DumbOptim:
    def state_dict(self) -> Dict[str, any]:
        return dict()
    
    def step(self):
        pass

    def zero_grad(self, set_to_none: bool = False):
        pass

def odict2dict(obj: any) -> any:
    if isinstance(obj, dict):
        res = dict()
        for field, val in obj.items():
            res[field] = odict2dict(val)
        return res
    elif isinstance(obj, list):
        return [odict2dict(item) for item in obj]
    return obj

class TestBase(unittest.TestCase):
    _runs_dir: Path = None

    def _ensure_runsdir(self):
        if self._runs_dir is None:
            self._runs_dir = Path(tempfile.mkdtemp())

    @property
    def runs_dir(self) -> Path:
        self._ensure_runsdir()
        return self._runs_dir

    def assertDictEqual(self, d1: Mapping[Any, object], d2: Mapping[Any, object], msg: Any = None) -> None:
        d1 = odict2dict(d1)
        d2 = odict2dict(d2)
        return super().assertDictEqual(d1, d2, msg)
    
    def assertDictContainsSubset(self, subset: Mapping[Any, Any], dictionary: Mapping[Any, Any], msg: object = None) -> None:
        subset = odict2dict(subset)
        dictionary = odict2dict(dictionary)
        return super().assertDictContainsSubset(subset, dictionary, msg)

    """
    return Experiment populated with .net, .optim, .sched, and loss_fn. ready to save
    as a checkpoint.
    """
    def create_dumb_exp(self, label: str = None, one: any = None, two: any = None) -> Experiment:
        exp = Experiment(label=label)
        exp.net = DumbNet(one=one, two=two)
        exp.sched = DumbSched()
        exp.optim = DumbOptim()
        exp.loss_fn = lambda _out, _truth: torch.tensor(0.0)
        exp.train_dataloader = lambda: True
        exp.val_dataloader = lambda: True
        exp.device = "cpu"
        return exp

    # def list_experiments(self) -> List[Experiment]:
    #     return checkpoint_util.list_experiments(_runs_dir=self._runs_dir)

    def checkpoints_dir(self, exp: Experiment = None) -> Path:
        res = Path(self._runs_dir, "checkpoints-test")
        if exp is not None:
            res = Path(res, f"{exp.created_at_short}-{exp.shortcode}--{exp.label}")
        res.mkdir(exist_ok=True, parents=True)
        return res
    
    def save_metadata(self, exp: Experiment) -> Path:
        self._ensure_runsdir()
        md_path = Path(self.checkpoints_dir(exp), "metadata.json")
        checkpoint_util.save_metadata(exp, md_path)
        return md_path

    def save_checkpoint(self, exp: Experiment) -> Tuple[Path, Path]:
        self._ensure_runsdir()

        nowstr = datetime.datetime.now().strftime(TIME_FORMAT_SHORT)

        cp_dir = self.checkpoints_dir(exp)
        md_path = Path(cp_dir, "metadata.json")
        cp_path = Path(cp_dir, f"epoch_{exp.nepochs}--{nowstr}.ckpt")

        checkpoint_util.save_checkpoint(exp, new_cp_path=cp_path, md_path=md_path)
        return md_path, cp_path

    def tearDown(self) -> None:
        super().tearDown()

        def walk(path: Path) -> int:
            num_removed = 0
            if not path.is_dir():
                # print(f"RM {path}")
                path.unlink()
                return 1
            for subpath in path.iterdir():
                num_removed += walk(subpath)
            # print(f"RMDIR {path}")
            path.rmdir()
            return num_removed

        if self._runs_dir is not None:
            walk(self._runs_dir)
        
        self._runs_dir = None
    
