from pathlib import Path
from typing import Sequence, List, Dict, Tuple, Union, Callable
import json
import re
import datetime

import torch
from torch import nn, Tensor

import experiment
from experiment import Experiment
import cmdline

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def find_checkpoints(runsdir: Path = Path("runs"), 
                     attr_matchers: Sequence[str] = list(),
                     only_paths: Union[str, re.Pattern] = None) -> List[Tuple[Path, Experiment]]:
    matcher_fn = lambda _exp: True
    if attr_matchers:
        # TODO
        matcher_fn = cmdline.gen_attribute_matcher(attr_matchers)

    res: List[Tuple[Path, Experiment]] = list()
    for run_path in runsdir.iterdir():
        if not run_path.is_dir():
            continue

        checkpoints = Path(run_path, "checkpoints")
        if not checkpoints.exists():
            continue

        for ckpt_path in checkpoints.iterdir():
            if not ckpt_path.name.endswith(".ckpt"):
                continue
            meta_path = Path(str(ckpt_path)[:-5] + ".json")

            exp = load_from_json(meta_path)
            if not matcher_fn(exp):
                continue

            if only_paths:
                if isinstance(only_paths, str):
                    if only_paths not in str(ckpt_path):
                        continue
                elif isinstance(only_paths, re.Pattern):
                    if not only_paths.match(str(ckpt_path)):
                        continue

            res.append((ckpt_path, exp))

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
                print(f"remove {field}")
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
                raise ValueError(f"missing_keys = {load_res.missing_keys} for {exp_in.label=}")
        return mod

    return fn

def resume_experiments(exps_in: List[Experiment], 
                       max_epochs: int,
                       checkpoints: List[Tuple[Path, Experiment]] = None) -> List[Experiment]:
    resume_exps: List[Experiment] = list()
    if checkpoints is None:
        checkpoints = find_checkpoints()
    
    now = datetime.datetime.now()

    for exp_in in exps_in:
        # starting the experiment causes it to fully populate fields.
        exp_in.start(0)

        match_exp: Experiment = None
        match_path: Path = None
        for cp_path, cp_exp in checkpoints:
            # sched_args / optim_args won't be set until saving metadata, 
            # which means 'exp_in' doesn't have them. the others aren't relevant
            # for resume.
            ignore = set('max_epochs batch_size label sched_args optim_args '
                         'do_compile use_amp'.split())
            is_same, same_fields, diff_fields = exp_in.is_same(cp_exp, extra_ignore_fields=ignore, return_tuple=True)
            if not is_same:
                continue

            # the checkpoint experiment won't have its lazy functions set. but we 
            # know based on above sameness comparison that the *type* of those
            # functions is the same. So, set the lazy functions based on the 
            # exp_in's, which has already been setup by the prior loop before
            # being passed in.
            cp_exp.loss_fn = exp_in.loss_fn
            cp_exp.train_dataloader = exp_in.train_dataloader
            cp_exp.val_dataloader = exp_in.val_dataloader
            # cp_exp.label += f",resume_{cp_exp.nepochs}"
            cp_exp.max_epochs = max_epochs

            # TODO: move this stuff to Experiment.resume?
            cp_exp.train_loss_hist = exp_in.train_loss_hist
            cp_exp.val_loss_hist = exp_in.val_loss_hist
            cp_exp.saved_at = None
            cp_exp.resumed_at.append((cp_exp.nepochs, now))

            cp_exp.lazy_net_fn = exp_in.lazy_net_fn
            cp_exp.lazy_sched_fn = exp_in.lazy_sched_fn
            cp_exp.lazy_optim_fn = exp_in.lazy_optim_fn
            cp_exp.resumed_from = str(cp_path)

            if match_exp is None or cp_exp.nepochs > match_exp.nepochs:
                match_exp = cp_exp
                match_path = cp_path

        if match_exp is not None:
            if (match_exp.nepochs + 1) >= max_epochs:
                print(f"* \033[1;31mskipping {match_exp.label}: checkpoint already has {match_exp.nepochs} epochs\033[0m")
                continue

            print(f"* \033[1;32mresuming {exp_in.label}: using checkpoint with {match_exp.nepochs} epochs\033[0m")
            resume_exps.append(match_exp)

            with open(match_path, "rb") as file:
                state_dict = torch.load(file)

            match_exp.lazy_net_fn = _resume_lazy_fn(match_exp, state_dict['net'], match_exp.lazy_net_fn)
            match_exp.lazy_optim_fn = _resume_lazy_fn(match_exp, state_dict['optim'], match_exp.lazy_optim_fn)
            match_exp.lazy_sched_fn = _resume_lazy_fn(match_exp, state_dict['sched'], match_exp.lazy_sched_fn)
        else:
            exp_in.net = None
            exp_in.sched = None
            exp_in.optim = None
            print(f"* \033[1mcouldn't find resume checkpoint for {exp_in.label}; starting a new one\033[0m")
            resume_exps.append(exp_in)

    return resume_exps

"""
Loads from either a model_dict or metadata_dict.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_from_dict(model_dict: Dict[str, any]) -> Experiment:
    exp = Experiment(label=model_dict['label'])
    return exp.load_model_dict(model_dict)

"""
Load Experiment: metadata only.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_from_json(json_path: Path) -> Experiment:
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)
    return load_from_dict(metadata)

"""
Save experiment metadata to .json
"""
def save_metadata(exp: Experiment, json_path: Path):
    metadata_dict = exp.metadata_dict()
    with open(json_path, "w") as json_file:
        json.dump(metadata_dict, json_file, indent=2)

"""
Save experiment .ckpt and .json.
"""
def save_ckpt_and_metadata(exp: Experiment, ckpt_path: Path, json_path: Path):
    obj_fields_none = {field: (getattr(exp, field, None) is None) for field in experiment.OBJ_FIELDS}
    if any(obj_fields_none.values()):
        raise Exception(f"refusing to save {ckpt_path}: some needed fields are None: {obj_fields_none=}")

    model_dict = exp.model_dict()
    with open(ckpt_path, "wb") as ckpt_file:
        torch.save(model_dict, ckpt_file)
    
    save_metadata(exp, json_path)