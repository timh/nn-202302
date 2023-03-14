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
import dn_util
sys.path.append("..")
import model_util
import experiment
from experiment import Experiment

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

def process_one(exp: Experiment, cp_path: Path, device: str):
    json_path = Path(str(cp_path).replace(".ckpt", ".json"))
    with open(json_path, "r") as md_file:
        metadata = json.load(md_file)

    with open(cp_path, "rb") as file:
        model_dict = torch.load(file)
        net_dict = model_dict['net']
        net = None

    do_save = False
    filename_fields = _read_filename_fields(cp_path)

    if 'do_flatten' in net_dict:
        net_dict.pop('do_flatten')
        print(f"  remove net.do_flatten")
        do_save = True
    
        if hasattr(exp, 'net_do_flatten'):
            print(f"  remove exp.net_do_flatten")
            del exp.net_do_flatten
            do_save = True
    
    if not 'varlen' in net_dict:
        net_dict['varlen'] = 0
        print(f"  add net.varlen = 0")
        do_save = True
    
    # copy fields from Experiment -> net
    net_type = dn_util.get_model_type(model_dict)
    for field in net_type._model_fields:
        if field in net_dict:
            continue

        if field not in model_dict:
            print(f"  ! can't fix net.{field}: not in model_dict")
            continue

        val = model_dict[field]
        if field == 'use_bias':
            # HACK
            val = True
        net_dict[field] = val
        print(f"  net.{field} = {val}")
        do_save = True

    # copy fields from net -> Experiment.net_{field}
    for field in net_type._metadata_fields:
        prefix_field = f"net_{field}"
        if hasattr(exp, prefix_field):
            continue

        if field not in net_dict:
            print(f"  ! can't fix exp.{prefix_field}: {field} not in net_dict")
            continue

        val = net_dict[field]
        setattr(exp, prefix_field, val)
        print(f"  exp.{prefix_field} = net.{field} = {val}")
        do_save = True

    if net_type == model.ConvEncDec:
        # model_util.print_dict(model_dict)
        if 'descs' not in net_dict:
            conv_descs_str = model_dict.get('conv_descs')
            net_dict['descs'] = model.gen_descs(conv_descs_str)
            print(f"  net.descs = gen_descs('{conv_descs_str}')")
            do_save = True
        
        net = dn_util.load_model(model_dict).to(device)

        if not hasattr(exp, 'net_latent_dim') or exp.net_latent_dim != net.latent_dim:
            # figure out the latent dim.
            # inputs = torch.zeros((1, net.nchannels, net.image_size, net.image_size))
            # inputs = inputs.to(device)
            # out = net.encoder(inputs)
            exp.net_latent_dim = net.latent_dim
            print(f"  exp.net_latent_dim = net.latent_dim = {net.latent_dim}")
            do_save = True


    if not json_path.exists() or do_save:
        print(f"  write metadata {json_path}")
        model_util.save_metadata(exp, json_path)

    if do_save:
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

    cfg = parser.parse_args()
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern)

    for cp_path, exp in tqdm.tqdm(model_util.find_checkpoints(only_paths=cfg.pattern)):
        try:
            process_one(exp, cp_path, "cuda")
        except Exception as e:
            print(f"error processing {cp_path}", file=sys.stderr)
            raise e

