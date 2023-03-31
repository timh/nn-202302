from typing import List, Set, Dict, Union
from collections import OrderedDict
from pathlib import Path
import types
import inspect
import datetime

import torch
from torch import Tensor

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
    elif type(value) == Tensor:
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
    elif isinstance(value, dict):
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


###
# metadata handling - convert to bool, int, str, float and lists of same.
###

"""
return fields from an object, excluding ones that start with _, or are
properties/functions.
"""
def md_obj_fields(obj: any) -> List[str]:
    if isinstance(obj, dict):
        return sorted(obj.keys())

    type_fields = {field: getattr(type(obj), field) 
                   for field in dir(type(obj))}
    ignore_fields = {field for field, val in type_fields.items()
                     if inspect.isfunction(val) 
                     or inspect.ismethod(val)
                     or isinstance(val, property)
                     or field.startswith("_")
                     or isinstance(val, Tensor)}
    fields: Set[str] = set()
    for field in dir(obj):
        if field in ignore_fields:
            continue
        # if (field.startswith("_") or field.endswith("_fn") or 
        #     field.endswith("_dataloader") or field in {'net', 'optim', 'sched'}):
        #     continue
        fields.add(field)
    return sorted(fields)


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_FORMAT_SHORT = "%Y%m%d-%H%M%S"

def md_scalar_allowed(val: any) -> bool:
    if isinstance(val, Tensor):
        return False

    if type(val) in [bool, int, float, str, datetime.datetime] or isinstance(val, Path):
        return True
    return False

def md_obj_allowed(val: any) -> bool:
    return md_scalar_allowed(val) or isinstance(val, list) or isinstance(val, dict)

def md_scalar(val: any) -> any:
    if isinstance(val, datetime.datetime):
        return val.strftime(TIME_FORMAT)
    elif isinstance(val, Path):
        return str(val)
    return val

def md_obj(obj: any, 
           only_fields: Set[str] = None,
           ignore_fields: Set[str] = None) -> any:
    if isinstance(obj, list) or isinstance(obj, tuple):
        res: List[any] = list()
        for item in obj:
            if md_obj_allowed(item):
                res.append(md_obj(item))
        return res

    elif md_scalar_allowed(obj):
        return md_scalar(obj)

    elif obj is None:
        return None
    
    obj_fields = md_obj_fields(obj)
    if only_fields:
        obj_fields = set(obj_fields).intersection(only_fields)
    if ignore_fields is not None:
        obj_fields = set(obj_fields) - set(ignore_fields)
    
    obj_fields = sorted(obj_fields)

    res: Dict[str, any] = OrderedDict()
    if isinstance(obj, dict):
        for field in obj_fields:
            val = obj.get(field)
            if md_obj_allowed(val):
                val = md_obj(val)
                res[field] = val
        return res
    
    for field in obj_fields:
        val = getattr(obj, field)
        if md_obj_allowed(val):
            res[field] = md_obj(val)

    return res

def duration_str(total_seconds: int) -> str:
    total_seconds = int(total_seconds)

    seconds = total_seconds % 60
    minutes = total_seconds // 60
    hours = minutes // 60
    days = hours // 24
    parts = [(days, "d"), (hours % 24, "h"), (minutes % 60, "m"), (seconds % 60, "s")]
    parts = [f"{val}{short}" for val, short in parts if val]

    if len(parts) == 0:
        return "0s"

    return " ".join(parts)


