from typing import Sequence, List, Tuple, Union, Dict, Callable
import collections
import types
from pathlib import Path
import datetime
import json
import re

import torch

import experiment
from experiment import Experiment

RE_OP = re.compile(r"([\w_]+)\s*([=<>!]+)\s*(.+)")
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
                matches = exp_val > matcher_val
            elif op == "<":
                matches = exp_val < matcher_val
            else:
                raise Exception(f"unknown {op=} for {field=} {matcher_val=}")
            
            if not matches:
                return False
        return True
    return fn

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def find_checkpoints(runsdir: Path = Path("runs"), 
                     attr_matchers: Sequence[str] = list(),
                     only_paths: Union[str, re.Pattern] = None) -> List[Tuple[Path, Experiment]]:
    matcher_fn = lambda _exp: True
    if attr_matchers:
        matcher_fn = gen_attribute_matcher(attr_matchers)

    res: List[Tuple[Path, Experiment]] = list()
    for run_path in runsdir.iterdir():
        if not run_path.is_dir():
            continue

        checkpoints = Path(run_path, "checkpoints")
        if not checkpoints.exists():
            continue

        for ckpt_path in checkpoints.iterdir():
            if not ckpt_path.name.endswith(".ckpt"):
                continue
            meta_path = Path(str(ckpt_path)[:-5] + ".json")

            exp = load_from_json(meta_path)
            if not matcher_fn(exp):
                continue

            if only_paths:
                if isinstance(only_paths, str):
                    if only_paths not in str(ckpt_path):
                        continue
                elif isinstance(only_paths, re.Pattern):
                    if not only_paths.match(str(ckpt_path)):
                        continue

            res.append((ckpt_path, exp))

    return res


"""
Loads from either a state_dict or metadata_dict.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_from_dict(state_dict: Dict[str, any]) -> Experiment:
    exp = Experiment(label=state_dict['label'])
    return exp.load_state_dict(state_dict)

"""
Load Experiment: metadata only.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_from_json(json_path: Path) -> Experiment:
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)
    return load_from_dict(metadata)

"""
Save experiment metadata to .json
"""
def save_metadata(exp: Experiment, json_path: Path):
    metadata_dict = exp.metadata_dict()
    with open(json_path, "w") as json_file:
        json.dump(metadata_dict, json_file, indent=2)

"""
Save experiment .ckpt and .json.
"""
def save_ckpt_and_metadata(exp: Experiment, ckpt_path: Path, json_path: Path):
    obj_fields_none = {field: (getattr(exp, field, None) is None) for field in experiment.OBJ_FIELDS}
    if any(obj_fields_none.values()):
        raise Exception(f"refusing to save {ckpt_path}: some needed fields are None: {obj_fields_none=}")

    state_dict = exp.state_dict()
    with open(ckpt_path, "wb") as ckpt_file:
        torch.save(state_dict, ckpt_file)
    
    save_metadata(exp, json_path)

TYPES = [int, float, bool, datetime.datetime, str, tuple, types.NoneType]
def print_value(value: any, field: str, level: int):
    field_str = ""
    if field:
        field_str = f"{field:20} = "

    indent = " " * level * 2
    if type(value) == str:
        print(f"{indent}{field_str}'{value}'")
    elif type(value) in TYPES:
        print(f"{indent}{field_str}{value}")
    elif type(value) == torch.Tensor:
        print(f"{indent}{field_str}Tensor {value.shape}")
    elif type(value) == list:
        if field == 'params':
            print(f"{indent}{field_str}[ .. len {len(value)} .. ]")
        elif len(value) and type(value[0]) in TYPES:
            list_str = ", ".join(map(str, value))
            print(f"{indent}{field_str}[ {list_str} ]")
        else:
            print(f"{indent}{field_str}[")
            print_list(value, level + 1)
            print(f"{indent}]")
    elif type(value) in [dict, collections.OrderedDict]:
        if field == 'state':
            print(f"{indent}{field_str}{{ .. len {len(value)} .. }}")
        else:
            print(f"{indent}{field_str}{{")
            print_dict(value, level + 1)
            print(f"{indent}}}")
    else:
        print(f"{indent}{field_str}?? {type(value)}")

def print_list(sd_list: List[any], level: int):
    for value in sd_list:
        print_value(value, "", level)

def print_dict(sd: Dict[str, any], level: int = 0):
    for field, value in sd.items():
        print_value(value, field, level)
