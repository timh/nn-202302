# %%
import datetime
from collections import deque
from typing import List, Dict, Deque, Tuple, Union, Callable
import copy

from PIL import Image, ImageDraw, ImageFont

from torch import Tensor
from torchvision import transforms

from experiment import Experiment

_img = Image.new("RGB", (1000, 500))
_draw = ImageDraw.ImageDraw(_img)

"""
Take a list of (string | List[str]).
Lay the strings out such that each line fits within (max_width) in the given font.

If a given item from in_strs is a list, output as follows:
* join each item in the list with (list_concat)
* for any lines past the first that output, indent the items by (list_indent)

Example:
  fit_lines(in_strs=["norm1", "norm2", ["item1", "item2", "item3"]], max_width=.., font=..., list_indent="  ", list_concat=","])

*Ignoring any computation of font and needed width*, it would return: "
norm1
norm2
item1,
  item2, 
  item3
"""
def fit_strings(in_strs: List[Union[str, List[str]]], max_width: int, font: ImageFont.ImageFont, list_indent = "  ") -> Tuple[str, int]:
    def line_to_str(line: List[str], sep = ""):
        return sep.join(map(str, line))

    # queue = deque of items to render. 
    # tuple is (string, is in list?, position in list, is last in list?)
    queue: Deque[(str, bool, int, bool)] = deque()
    for in_str in in_strs:
        if isinstance(in_str, list):
            for itemidx, item in enumerate(in_str):
                is_last_in_list = (itemidx == len(in_str) - 1)
                queue.append((item, True, itemidx, is_last_in_list))
            continue
        queue.append((str(in_str), False, 0, True))

    lines: List[List[str]] = [[]]
    while len(queue):
        value, is_in_list, idx_in_list, is_last_in_list = queue[0]
        
        # if the line is currently empty, put the first chunk on it, no matter
        # the length.
        if not len(lines[-1]):
            if idx_in_list > 0:
                value = list_indent + value
            lines[-1].append(value)
            queue.popleft()
            continue

        # if we're on the first item of a list, start a new line for it.
        if is_in_list and idx_in_list == 0:
            lines.append([value])
            queue.popleft()
            continue

        # consider adding the next piece onto the line
        if is_in_list:
            new_line = lines[-1] + [value]
        else:
            new_line = lines[-1] + [" ", value]
        new_line_str = line_to_str(new_line)

        # left, top, right, bottom
        left, top, right, bottom = _draw.textbbox((0, 0), text=new_line_str, font=font)
        new_width = right
        if new_width > max_width:
            # if new_line was too wide, start a new line, without consuming remaining.
            lines.append(list())
            continue

        # new_line fits. replace it with the longer one.
        queue.popleft()
        lines[-1] = new_line

        # add a newline at the end of a list.
        if is_last_in_list:
            lines.append(list())
    
    res_lines = [line_to_str(line) for line in lines]
    res_str = "\n".join(res_lines)

    left, top, right, bottom = _draw.textbbox((0, 0), text=res_str, font=font)
    return res_str, bottom

def fit_strings_multi(in_list: List[List[Union[str, List[str]]]], 
                      max_width: int, 
                      font: ImageFont.ImageFont) -> Tuple[List[str], int]:
    labels: List[str] = list()
    heights: List[int] = list()

    for in_strs in in_list:
        label, height = fit_strings(in_strs, max_width=max_width, font=font)
        labels.append(label)
        heights.append(height)
    
    return labels, max(heights)

"""
return labels that will fit in the available_width with the given font.

return:
  List of labels
  Maximum height used
"""
def fit_exp_descrs(exps: List[Experiment],
                   *,
                   max_width: int,
                   font: ImageFont.ImageFont) -> Tuple[List[str], int]:
    exp_descrs = [exp.describe() for exp in exps]
    return fit_strings_multi(exp_descrs, max_width=max_width, font=font)

_to_pil_xform = transforms.ToPILImage("RGB")
def tensor_to_pil(image_tensor: Tensor, image_size: int) -> Image.Image:
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]

    image: Image.Image = _to_pil_xform(image_tensor)
    if image_size != image.width:
        image = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    
    return image

_to_tensor_xform = transforms.ToTensor()
def pil_to_tensor(image: Image.Image, net_size: int) -> Tensor:
    if net_size != image.width:
        image = image.resize((net_size, net_size), resample=Image.Resampling.BICUBIC)
    return _to_tensor_xform(image)

