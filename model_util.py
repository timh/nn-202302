from typing import List, Dict
import collections
import types
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
