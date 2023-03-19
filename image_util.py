# %%
import datetime
from collections import deque
from typing import List, Dict, Deque, Tuple, Union
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
def fit_strings(in_strs: List[Union[str, List[str]]], max_width: int, font: ImageFont.ImageFont, list_indent = "  ", list_concat = ",") -> Tuple[str, int]:
    def line_to_str(line: List[str]):
        return " ".join(map(str, line))

    # queue = deque of items to render. 
    # tuple is (string, index in list, was in list, ? last item in list)
    queue: Deque[(str, int, bool)] = deque()
    for in_str in in_strs:
        if isinstance(in_str, list):
            for itemidx, item in enumerate(in_str):
                last_in_list = (itemidx == len(in_str) - 1)
                queue.append((item, True, itemidx, last_in_list))
            continue
        queue.append((str(in_str), False, 0, True))

    lines: List[List[str]] = [[]]
    while len(queue):
        value, in_list, idx_in_list, last_in_list = queue[0]
        if not last_in_list:
            value = value + list_concat
        
        # if the line is currently empty, put the first chunk on it, no matter
        # the length.
        if not len(lines[-1]):
            if idx_in_list > 0:
                value = list_indent + value
            lines[-1].append(value)
            queue.popleft()
            continue

        # if we're on the first item of a list, start a new line for it.
        if in_list and idx_in_list == 0:
            lines.append([value])
            queue.popleft()
            continue

        # consider adding the next piece onto the line
        new_line = lines[-1] + [value]
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
    
    res_lines = [line_to_str(line) for line in lines]
    res_str = "\n".join(res_lines)

    left, top, right, bottom = _draw.textbbox((0, 0), text=res_str, font=font)
    return res_str, bottom

def dict_to_strings(obj: Dict[str, any]) -> List[str]:
    res: List[str] = list()
    for field, val in obj.items():
        res.append(f"{field} {val}")
    return res

"""
return labels that will fit in the available_width with the given font.

return:
  List of labels
  Maximum height used
"""
def experiment_labels(experiments: List[Experiment], 
                      *,
                      max_width: int,
                      fields_per_col: int = 0,
                      font: ImageFont.ImageFont) -> Tuple[List[str], int]:
    now = datetime.datetime.now()

    labels: List[List[str]] = list()
    heights: List[int] = list()

    for exp in experiments:
        if exp.saved_at:
            ago = int((now - exp.saved_at).total_seconds())
            ago_secs = ago % 60
            ago_mins = (ago // 60) % 60
            ago_hours = (ago // 3600)
            ago = deque([(val, desc) for val, desc in zip([ago_hours, ago_mins, ago_secs], ["h", "m", "s"])])
            while not ago[0][0]:
                ago.popleft()
            ago_str = " ".join([f"{val}{desc}" for val, desc in ago])
        else:
            ago_str = ""
        

        exp_fields = dict()
        exp_fields['startlr'] = format(exp.startlr, ".1E")
        exp_fields['tloss'] = format(exp.lastepoch_train_loss, ".3f")
        exp_fields['vloss'] = format(exp.lastepoch_val_loss, ".3f")
        if ago_str:
            exp_fields['ago'] = ago_str

        strings = dict_to_strings(exp_fields)
        strings.append(exp.label.split(","))

        label, height = fit_strings(strings, max_width=max_width, font=font)
        labels.append(label)
        heights.append(height)
    
    return labels, max(heights)

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

