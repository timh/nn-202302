# %%
import sys
import re
from typing import Dict, List
from pathlib import Path
import tqdm
import torch

import denoise_logger

sys.path.append("..")
import experiment
from experiment import Experiment

RE_FILENAME_FIELDS = re.compile(r".*checkpoints\/(.*),(emblen.*)\.ckpt")
EXTRA_FIELDS_INT = "emblen nlin hidlen nparams batch cnt epoch image_size".split(" ")
EXTRA_FIELDS_FLOAT = "slr elr".split(" ")
EXTRA_FIELDS_BOOL = "bnorm lnorm flatconv2d".split(" ")
EXTRA_FIELDS = EXTRA_FIELDS_INT + EXTRA_FIELDS_FLOAT + EXTRA_FIELDS_BOOL + ["loss"]
REMAP_FIELDS = dict(cnt="minicnt", epoch="nepochs", batch="batch_size", 
                    nlin="nlinear",
                    slr="startlr", elr="endlr",
                    bnorm="do_batchnorm", lnorm="do_layernorm", flatconv2d="do_flatconv2d",
                    loss="loss_type")

def process_one(cp_path: Path):
    with open(cp_path, "rb") as file:
        state_dict = torch.load(file)

    # exp = Experiment.new_from_state_dict(state_dict)
    do_save = False

    mapping = {"last_train_loss": "lastepoch_train_loss", "last_val_loss": "lastepoch_val_loss"}
    for oldname, newname in mapping.items():
        if oldname in state_dict:
            print(f"  rename {oldname} -> {newname}")
            state_dict[newname] = state_dict.pop(oldname)
            do_save = True

    match = RE_FILENAME_FIELDS.match(str(cp_path))
    if match:
        path_conv_descs, rest = match.groups()

        dict_conv_descs = state_dict.get("conv_descs", None)
        if dict_conv_descs != path_conv_descs:
            print(f"  conv_descs {dict_conv_descs} -> {path_conv_descs}")
            state_dict["conv_descs"] = path_conv_descs
            do_save = True

        path_str_list = rest.split(",")
        path_fields = {}
        for path_str in path_str_list:
            if "_" in path_str:
                idx_ = path_str.index("_")
                field, val = path_str[:idx_], path_str[idx_ + 1:]
            else:
                field = path_str
                val = True

            # process single words with no value side.
            if field in ["constant", "nanogpt"]:
                val = field
                field = "sched_type"
            elif field in ["adamw", "sgd"]:
                val = field
                field = "optim_type"
            elif field in EXTRA_FIELDS_BOOL:
                val = True

            # process the rest
            elif field == "cnt":
                continue
            elif field in EXTRA_FIELDS_FLOAT:
                val = float(val)
            elif field in EXTRA_FIELDS_INT:
                if field == "nparams":
                    val = int(float(val[:-1]) * 1e-6)
                else:
                    val = int(val)
            elif field == "loss":
                field = "loss_type"
            else:
                raise Exception(f"{field=} {val=} not expected")
            path_fields[field] = val

        for field_path, val_path in path_fields.items():
            field_dict = REMAP_FIELDS.get(field_path, field_path)
            val_dict = state_dict.get(field_dict, None)
            if val_path != val_dict:
                if field_path == "loss_type" and val_dict.replace("*", "").replace("+", "") == val_path:
                    continue
                print(f"  path {field_path}:{val_path} != dict {field_dict}:{val_dict}")
                state_dict[field_dict] = val_path
                do_save = True
            else:
                # print(f"  path {field_path}:{val_path} == dict {field_dict}:{val_dict}")
                pass

    defaults = dict(
        image_size=128, loss_type="l1", 
        do_layernorm=False, do_batchnorm=False, do_flatconv2d=False,
    )
    for field, def_value in defaults.items():
        dict_value = state_dict.get(field, None)
        if dict_value is None:
            print(f"  add {field} = {def_value}")
            state_dict[field] = def_value
            do_save = True
    
    if "net" in state_dict:
        for field in "do_layernorm do_batchnorm do_flatconv2d emblen nlinear hidlen image_size".split(" "):
            dict_val = state_dict.get(field, None)
            net_val = state_dict["net"].get(field)
            if dict_val != net_val:
                print(f"  {field}: dict {dict_val} -> net {net_val}")
                state_dict[field] = net_val
                do_save = True


    loss_type = state_dict.get("loss_type", None)
    if loss_type.startswith("edge"):
        start_loss_type = loss_type
        loss_type = loss_type
        if loss_type[4:5] not in ["+", "*"]:
            loss_type = "edge+" + loss_type[4:]
        if start_loss_type != loss_type:
            print(f"  loss_type {start_loss_type} -> {loss_type}")
            state_dict["loss_type"] = loss_type
            do_save = True

    json_path = Path(str(cp_path).replace(".ckpt", ".json"))

    if not json_path.exists() or do_save:
        print(f"  write metadata {json_path}")
        exp = Experiment.new_from_dict(state_dict)
        experiment.save_metadata(exp, json_path)

    if do_save:
        print(f"  update checkpoint {cp_path}")
        with open(cp_path, "wb") as cp_file:
            torch.save(state_dict, cp_file)

def find_checkpoints(path: Path) -> List[Path]:
    res: List[Path] = list()
    for child in path.iterdir():
        if child.is_dir():
            childres = find_checkpoints(child)
            res.extend(childres)
            continue

        if child.name.endswith(".ckpt"):
            res.append(child)
    return res

if __name__ == "__main__":
    for cp_path in tqdm.tqdm(find_checkpoints(Path("runs"))):
        process_one(cp_path)

