# %%
import sys
import re
from typing import Dict, List, Type, Union
from pathlib import Path
import tqdm
import argparse
import json

import torch

import model
import model_sd
from nnexp.denoise import dn_util
sys.path.append("..")
import model_util
import experiment
from nnexp.experiment import Experiment

RE_FILENAME_FIELDS = re.compile(r".*checkpoints\/(.*),(emblen.*)\.ckpt")

def _read_filename_fields(cp_path: Path) -> Dict[str, any]:
    match = RE_FILENAME_FIELDS.match(str(cp_path))
    if not match:
        return {}
    
    path_conv_descs, rest = match.groups()

    # rest is name/value pairs, some of which are just names, which are either
    # True/False or picking from an enum
    path_str_list = rest.split(",")
    path_fields = {}
    for path_str in path_str_list:
        if "_" in path_str:
            idx_ = path_str.index("_")
            field, val = path_str[:idx_], path_str[idx_ + 1:]
        else:
            field = path_str
            val = True

        path_fields[field] = val

    return path_fields 

def process_one(exp: Experiment, cp_path: Path, doit: bool, device: str):
    do_save = False

    json_path = Path(str(cp_path).replace(".ckpt", ".json"))
    with open(json_path, "r") as md_file:
        metadata = json.load(md_file)

    with open(cp_path, "rb") as file:
        model_dict = torch.load(file)
        net_dict = model_dict['net']
        net = None

    net_class = metadata['net_class']
    if net_class != 'VarEncDec':
        raise NotImplementedError(f"{net_class=}")

    if 'encoder_kernel_size' not in net_dict:
        emblen = net_dict['emblen']
        print(f"  net.encoder_kernel_size = 0 ({emblen=})")
        net_dict['encoder_kernel_size'] = 0
        do_save = True

    if not do_save:
        return

    if not doit:
        print("  not saving cuz dry run")
        return

    print(f"  update checkpoint {cp_path}")
    temp = Path(str(cp_path) + ".tmp")
    if not isinstance(model_dict['net'], dict):
        model_util.print_dict(model_dict)
        raise Exception("model_dict[net] is not a dict")

    with open(temp, "wb") as cp_file:
        torch.save(model_dict, cp_file)
    # cp_path.unlink(missing_ok=True)
    temp.rename(cp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", type=str, default=None)
    parser.add_argument("--doit", action='store_true', default=False)

    cfg = parser.parse_args()
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern)

    for cp_path, exp in tqdm.tqdm(model_util.find_checkpoints(only_paths=cfg.pattern)):
        try:
            process_one(exp, cp_path, cfg.doit, "cuda")
        except Exception as e:
            print(f"error processing {cp_path}", file=sys.stderr)
            raise e

