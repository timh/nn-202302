from typing import List, Set, Dict, Union
from collections import OrderedDict
import collections
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
    type_fields = {field: getattr(type(obj), field) 
                   for field in dir(type(obj))}
    ignore_fields = {field for field, val in type_fields.items()
                     if inspect.isfunction(val) 
                     or inspect.ismethod(val)
                     or isinstance(val, property)
                     or field.startswith("_")}
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

def md_type_allowed(val: any) -> bool:
    if type(val) in [bool, int, float, str, datetime.datetime]:
        return True
    if type(val) in [list, tuple]:
        return all([md_type_allowed(item) for item in val])
    return False

def md_scalar(val: any) -> Union[bool, int, float, str, list, dict]:
    if isinstance(val, datetime.datetime):
        return val.strftime(TIME_FORMAT)
    elif type(val) in [list, tuple]:
        return [md_scalar(item) for item in val]
    return val

def md_obj(obj: any, 
           only_fields: Set[str] = None,
           ignore_fields: Set[str] = None) -> Dict[str, any]:
    obj_fields = only_fields or md_obj_fields(obj)
    obj_fields = set(obj_fields)

    if ignore_fields is not None:
        obj_fields = obj_fields - ignore_fields
    
    res: Dict[str, any] = OrderedDict()
    obj_fields = sorted(obj_fields)
    for field in obj_fields:
        val = getattr(obj, field)
        if not md_type_allowed(val):
            continue

        val = md_scalar(val)
        res[field] = val
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


