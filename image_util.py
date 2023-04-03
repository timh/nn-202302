import datetime
from collections import deque
from typing import List, Dict, Deque, Tuple, Union, Literal, Callable
import types
import copy

from PIL import Image, ImageDraw, ImageFont

from torch import Tensor
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

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
    return list of strings with short(er) field names.

    if split_label_on is set, return a list of strings for the label, instead of
    just a string.
"""
def exp_descr(exp: Experiment, 
              only_fields_map: Dict[str, str] = None, 
              extra_field_map: Dict[str, str] = None, 
              include_label = True,
              include_loss = True) -> List[Union[str, List[str]]]:
    if only_fields_map is None:
        field_map = {'startlr': 'startlr', 'shortcode': 'shortcode'} #, 'created_at_short': 'created_at'}
    else:
        field_map = only_fields_map
    
    field_map = field_map.copy()

    if include_loss:
        # field_map['best_train_loss'] = 'best_tloss'
        # field_map['best_val_loss'] = 'best_vloss'
        field_map['last_train_loss'] = 'tloss'
        # field_map['last_val_loss'] ='last_vloss'

    if extra_field_map:
        field_map.update(extra_field_map)

    exp_fields = dict()
    for idx, (field, short) in enumerate(field_map.items()):
        val = getattr(exp, field, None)
        if type(val) in [types.MethodType, types.FunctionType]:
            val = val()

        if val is None:
            continue

        if 'lr' in field:
            val = format(val, ".1E")
        elif isinstance(val, float):
            val = format(val, ".3f")
        exp_fields[short] = str(val)
        if idx < len(field_map) - 1:
            exp_fields[short] += ","

    strings = [f"{field} {val}" for field, val in exp_fields.items()]

    if include_label:
        comma_parts = exp.label.split(",")
        for comma_idx, comma_part in enumerate(comma_parts):
            dash_parts = comma_part.split("-")
            if len(dash_parts) == 1:
                if comma_idx != len(comma_parts) - 1:
                    comma_part += ","
                strings.append(comma_part)
                continue

            for dash_idx in range(len(dash_parts)):
                if dash_idx != len(dash_parts) - 1:
                    dash_parts[dash_idx] += "-"
            strings.append(dash_parts)
    
    return strings


def annotate(*,
             image: Image.Image, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont,
             text: str, upper_left: Tuple[int, int],
             within_size: int, 
             ref: Literal["upper_left", "lower_left", "upper_right", "lower_right"] = "lower_left",
             text_fill: str = "white", rect_fill: str = "black"):

    left, top, right, bot = draw.textbbox(xy=(0, 0), text=text, font=font)

    ul_x, ul_y = upper_left
    if ref == "upper_left":
        textpos = upper_left
    elif ref == "lower_left":
        textpos = (ul_x, ul_y + within_size - bot)
    elif ref == "upper_right":
        textpos = (ul_x + within_size - right, ul_y)
    elif ref == "lower_right":
        textpos = (ul_x + within_size - right, ul_y + within_size - bot)
    else:
        raise ValueError(f"unknown {ref=}")

    rect_pos = (textpos[0] + left, textpos[1] + top,
                textpos[0] + right, textpos[1] + bot)
    draw.rectangle(xy=rect_pos, fill=rect_fill)
    draw.text(xy=textpos, text=text, fill=text_fill, font=font)




    # left, top, right, bot = draw.textbbox()
    # _text_list, max_height = image_util.fit_strings([text], max_width=cfg.image_size, font=font)
    # text_pos = (0, pos[1] - max_height)

    # draw.text(xy=text_pos, )
    pass
    

"""
    dataset which returns its images in (input, truth) tuple form.
"""
class PlainDataset:
    dataset: Dataset
    def __init__(self, base_dataset: Dataset):
        self.dataset = base_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            raise Exception("no work")
        src, _ = self.dataset[idx]
        return src, src


def get_datasets(*,
                 image_size: int, image_dir: str, 
                 train_split: float = 0.9,
                 limit_dataset: int = None) -> Tuple[Dataset, Dataset]:
    dataset = torchvision.datasets.ImageFolder(
        root=image_dir,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
    ]))
    dataset = PlainDataset(dataset)
    
    if limit_dataset is not None:
        dataset = data.Subset(dataset, range(0, limit_dataset))

    if train_split < 1.0:
        train_split_idx = int(len(dataset) * train_split)
        train_data = data.Subset(dataset, range(0, train_split_idx))
        val_data = data.Subset(dataset, range(train_split_idx, len(dataset)))
    else:
        train_data = data.Subset(dataset, range(0, len(dataset)))
        val_data = None

    return train_data, val_data
    
def get_dataloaders(*,
                    image_size: int, image_dir: str, batch_size: int,
                    train_split = 0.9, shuffle = True,
                    limit_dataset: int = None) -> Tuple[DataLoader, DataLoader]:

    train_data, val_data = \
        get_datasets(image_size=image_size, image_dir=image_dir,
                     train_split=train_split, limit_dataset=limit_dataset)

    train_dl = data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    if val_data is not None:
        val_dl = data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    else:
        val_dl = None
    
    return train_dl, val_dl

#######
# IMAGE/TENSOR TRANSFORMS
#######

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

