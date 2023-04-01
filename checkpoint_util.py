from pathlib import Path
from typing import Sequence, List, Set, Dict, Tuple, Union, Callable, Literal
from collections import defaultdict
import json
import re
import datetime

import torch
from torch import nn, Tensor

import experiment
from experiment import Experiment, LossType

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def list_experiments(runs_dir: Path = Path("runs")) -> List[Experiment]:
    all_cp_dirs: List[Path] = list()
    for runs_subdir in runs_dir.iterdir():
        if not runs_subdir.is_dir() or not runs_subdir.name.startswith("checkpoints-"):
            continue

        for exp_dir in runs_subdir.iterdir():
            # directories within checkpoints-{basename} are subdirs with 
            # checkpoints in them.
            if "--" in exp_dir.name:
                all_cp_dirs.append(exp_dir)

    exp_by_shortcode: Dict[str, Experiment] = dict()
    cps_by_shortcode: Dict[str, List[Path]] = defaultdict(list)

    for cp_dir in all_cp_dirs:
        cp_paths = {file for file in cp_dir.iterdir() if file.name.endswith(".ckpt")}

        metadata_path = Path(cp_dir, "metadata.json")
        if metadata_path.exists():
            # new layout: associate the same metadata.json with all 
            # the checkpoints.
            exp = load_from_json(metadata_path)
            exp.metadata_path = metadata_path
            exp_by_shortcode[exp.shortcode] = exp
            for cp_path in cp_paths:
                cps_by_shortcode[exp.shortcode].append(cp_path)

    res: List[Experiment] = list()
    for shortcode in exp_by_shortcode.keys():
        exp = exp_by_shortcode[shortcode]
        cp_paths = cps_by_shortcode[shortcode]

        for cp_path in cp_paths:
            exp.run_for_path(cp_path)
        
        res.append(exp)
    
    return res

"""
    lazy net/sched/optim loader for a resumed checkpoint. call the lazy function
    passed in (from exp_in) to initialize the object with the correct parameters,
    then call load_state_dict on it with data from the checkpoint.
"""
def _resume_lazy_fn(exp: Experiment, 
                    mod_state_dict: Dict[str, any], 
                    exp_in_lazy_fn: Callable[[Experiment], nn.Module]) -> Callable[[Experiment], nn.Module]:
    def fn(exp: Experiment) -> nn.Module:
        mod = exp_in_lazy_fn(exp)
        if hasattr(mod, '_model_fields'):
            for field in mod._model_fields:
                # print(f"remove {field}")
                mod_state_dict.pop(field, None)

        if isinstance(mod, torch.optim.lr_scheduler.LRScheduler) or isinstance(mod, torch.optim.Optimizer):
            # these load_state_dict functions don't take strict.
            mod.load_state_dict(mod_state_dict)
        else:
            # BUG: need to use strict=False in case the model has placed some 
            # other bits into its state_dict, and we can't remove them purely 
            # by removing model_fields.
            load_res = mod.load_state_dict(mod_state_dict, False)
            if load_res.missing_keys:
                raise ValueError(f"missing_keys = {load_res.missing_keys} for {exp.label=}")
        return mod

    return fn

"""
Match incoming experiments with any checkpoints that are the same.

For each exp_in, return:
* the matching checkpoint experiment with the highest nepochs,
* or if none, the exp_in itself.
"""
def resume_experiments(*,
                       exps_in: List[Experiment], 
                       max_epochs: int,
                       use_best: LossType = None,
                       runs_dir: Path = Path("runs")) -> List[Experiment]:
    use_last = use_best is None

    res: List[Experiment] = list()

    existing_exps = list_experiments(runs_dir=runs_dir)
    existing_by_shortcode = {exp.shortcode: exp for exp in existing_exps}
    
    exps_in_shortcodes: Set[str] = set()
    for exp_in in exps_in:
        # start the experiment, to make it fully instantiate fields of net, sched,
        # optim and possibly others.
        exp_in.start(0)

        # make copy of exp_in, for 1) sanity check, and 2) as somewhere to keep
        # the settings (e.g., batch_size, startlr) for prepare_resume, below
        exp_in_copy = Experiment().load_model_dict(exp_in.metadata_dict())
        if exp_in.shortcode != exp_in_copy.shortcode:
            raise Exception(f"{exp_in.shortcode=} != {exp_in_copy.shortcode=}!")

        if exp_in.shortcode in exps_in_shortcodes:
            raise ValueError(f"duplicate incoming experiments: {exp_in.shortcode=}")
        exps_in_shortcodes.add(exp_in.shortcode)

        if exp_in.shortcode in existing_by_shortcode:
            existing_exp = existing_by_shortcode[exp_in.shortcode]
            exp_in.end()
            exp_in.load_model_dict(existing_exp.metadata_dict())

            # NOTE: there may still be lingering off-by-one errors for nepochs.
            if exp_in.nepochs >= max_epochs:
                print(f"* \033[1;31mskipping {exp_in.shortcode}: checkpoint already has {exp_in.nepochs} epochs\033[0m")
                continue

            if use_last:
                resume_from = exp_in.cur_run()
            else:
                # print(f"best {use_best}")
                resume_from = exp_in.run_best_loss(loss_type=use_best)

            print(f"* \033[1;32mresuming {exp_in.shortcode}: using checkpoint with {resume_from.checkpoint_nepochs} epochs\033[0m")
            run_in_copy = exp_in_copy.cur_run()
            run_in_copy.max_epochs = max_epochs
            exp_in.prepare_resume(from_run=resume_from, with_settings=run_in_copy)
            with open(resume_from.checkpoint_path, "rb") as file:
                state_dict = torch.load(file)

            exp_in.lazy_net_fn = _resume_lazy_fn(exp_in, state_dict['net'], exp_in.lazy_net_fn)
            exp_in.lazy_optim_fn = _resume_lazy_fn(exp_in, state_dict['optim'], exp_in.lazy_optim_fn)
            res.append(exp_in)

            # NOTE: do NOT resume scheduler. this will wipe out any learning rate changes we've made.
            # match_exp.lazy_sched_fn = _resume_lazy_fn(match_exp, state_dict['sched'], match_exp.lazy_sched_fn)
        else:
            print(f"* \033[1mcouldn't find resume checkpoint for {exp_in.shortcode}; starting a new one\033[0m")
            res.append(exp_in)

    # stop all experiments passed in (including those that won't be returned)
    for exp_in in exps_in:
        exp_in.net.cpu()
        exp_in.net = None
        exp_in.sched = None
        exp_in.optim = None
    
    # return only those that weren't skipped
    return res

"""
Load Experiment: metadata only.
"""
def load_from_json(json_path: Path) -> Experiment:
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)
    res = Experiment()
    res.metadata_path = json_path
    return res.load_model_dict(metadata)

"""
Save experiment metadata to .json
"""
def save_metadata(exp: Experiment, json_path: Path):
    metadata_dict = exp.metadata_dict()

    temp_path = Path(str(json_path) + ".tmp")
    with open(temp_path, "w") as json_file:
        json.dump(metadata_dict, json_file, indent=2)
    temp_path.rename(json_path)

"""
Save experiment .ckpt and .json.
"""
def save_ckpt_and_metadata(exp: Experiment, ckpt_path: Path, json_path: Path):
    obj_fields_none = {field: (getattr(exp, field, None) is None) for field in experiment.OBJ_FIELDS}
    if any(obj_fields_none.values()):
        raise Exception(f"refusing to save {ckpt_path}: some needed fields are None: {obj_fields_none=}")

    exp.update_for_checkpoint(ckpt_path)

    save_metadata(exp, json_path)

    model_dict = exp.model_dict()

    temp_path = Path(str(ckpt_path) + ".tmp")
    with open(temp_path, "wb") as ckpt_file:
        torch.save(model_dict, ckpt_file)
    temp_path.rename(ckpt_path)
