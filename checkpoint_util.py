from pathlib import Path
from typing import Sequence, List, Set, Dict, Tuple, Union, Callable, Literal
from collections import defaultdict
import json
import re
import datetime

import torch
from torch import nn, Tensor

import experiment
from experiment import Experiment, ExpRun, LossType

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def list_experiments(runs_dir: Path = Path("runs"), filter_invalid = True) -> List[Experiment]:
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
            exp = load_from_metadata(metadata_path)
            exp.metadata_path = metadata_path
            exp_by_shortcode[exp.shortcode] = exp
            for cp_path in cp_paths:
                cps_by_shortcode[exp.shortcode].append(cp_path)

    res: List[Experiment] = list()
    for shortcode in exp_by_shortcode.keys():
        exp = exp_by_shortcode[shortcode]
        cp_paths = cps_by_shortcode[shortcode]

        # filter out runs that are missing their checkpoint_path. NOTE I think this
        # happens when an experiment is aborted before it writes a single checkpoint.
        fixed_runs: List[ExpRun] = list()
        for i, run in enumerate(exp.runs):
            if run.checkpoint_path is None:
                if run.checkpoint_nepochs > 0:
                    raise Exception(f"! {exp.shortcode}: run {i + 1}/{len(exp.runs)} with cp_nepochs {run.checkpoint_nepochs} has no cp_path!")
                continue
            elif filter_invalid and not run.checkpoint_path.exists():
                # print(f"{exp.created_at_short}-{exp.shortcode}: run {i + 1}/{len(exp.runs)} with cp_nepochs {run.checkpoint_nepochs} doesn't exist")
                continue
            fixed_runs.append(run)
        exp.runs = fixed_runs
        if not len(exp.runs):
            raise Exception(f"{exp.shortcode} has no runs after filtering! {exp.nepochs=}")

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
            if (exp_in.nepochs + 1) >= max_epochs:
                print(f"* \033[1;31mskipping {exp_in.shortcode}: checkpoint already has {exp_in.nepochs + 1} epochs\033[0m")
                continue

            if use_last:
                resume_from = exp_in.get_run()
            else:
                # print(f"best {use_best}")
                resume_from = exp_in.get_run(loss_type=use_best)

            print(f"* \033[1;32mresuming {exp_in.shortcode}: using checkpoint with {resume_from.checkpoint_nepochs} epochs\033[0m")
            run_in_copy = exp_in_copy.get_run()
            run_in_copy.max_epochs = max_epochs
            prepare_resume(exp_in, from_run=resume_from, with_settings=run_in_copy)
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
prepare the experiment to resume
- from_run: a run from *this* experiment that should be used as the
    resume point.
- with_settings: a new (unstarted) ExpRun with batch_size, LR, etc
    settings
"""
def prepare_resume(exp: Experiment, from_run: ExpRun, with_settings: ExpRun):
    # truncate whatever history might have happened after this checkpoint
    # was written.
    exp.nepochs = from_run.checkpoint_nepochs
    exp.nbatches = from_run.checkpoint_nbatches
    exp.nsamples = from_run.checkpoint_nsamples

    exp.val_loss_hist = [(epoch, vloss) for epoch, vloss in exp.val_loss_hist if epoch <= exp.nepochs]
    exp.train_loss_hist = exp.train_loss_hist[:exp.nepochs]
    exp.runs = [run for run in exp.runs if run.checkpoint_nepochs <= exp.nepochs]

    # copy fields from the with_settings Run.
    resume = ExpRun()
    resume.resumed_from = from_run.checkpoint_path

    fields = ('batch_size do_compile use_amp max_epochs '
              'startlr endlr optim_type sched_type sched_warmup_epochs').split()
    for field in fields:
        in_val = getattr(with_settings, field)
        setattr(resume, field, in_val)

    exp.runs.append(resume)

def remove_checkpoint(exp: Experiment, old_cp_path: Path, unlink: bool) -> ExpRun:
    # copy the old one to make the new.
    new_runs: List[ExpRun] = list()
    found_run: ExpRun = None
    for run in exp.runs:
        # print(f"before: run: epochs {run.checkpoint_nepochs}")
        if run.checkpoint_path == old_cp_path:
            found_run = run
            continue
        new_runs.append(run)

    if found_run is None:
        raise Exception(f"can't find run for {old_cp_path}")

    if unlink:
        old_cp_path.unlink()

    exp.runs = new_runs
    return found_run

"""
Save a checkpoint.
old_cp_path:
- if set, this will look for an existing checkpoint with that path, and 
  replace its path/nepochs/timestamp with new stats.
- if not set:
  - if the existing run is blank (no cp_nepochs, cp_path), it will be modified.
  - if the existing run is not blank, a new one will be created, copying from
    the last, if any.
"""    
def save_checkpoint(exp: Experiment, new_cp_path: Path, md_path: Path,
                    old_cp_path: Path = None):
    if not len(exp.runs):
        raise ValueError(f"{exp.shortcode}: invalidate Experiment: no runs")

    if old_cp_path is not None:
        found_run = remove_checkpoint(exp, old_cp_path, unlink=False)
        run = found_run.copy()
        exp.runs.append(run)
    else:
        run = exp.get_run()
        if run.checkpoint_path is None and (len(exp.runs) == 1 or run.resumed_from is not None):
            # this is either the blank run that an experiment starts with, or a resumed
            # run. use it.
            pass
        elif run.checkpoint_nepochs == exp.nepochs and run.checkpoint_path == new_cp_path:
            # this is just another checkpoint for the same epoch. overwrite it.
            pass
        else:
            run = exp.get_run().copy()
            exp.runs.append(run)

    run.checkpoint_nepochs = exp.nepochs
    run.checkpoint_nbatches = exp.nbatches
    run.checkpoint_nsamples = exp.nsamples
    run.checkpoint_at = datetime.datetime.now()
    run.checkpoint_path = new_cp_path

    _save_checkpoint_and_metadata(exp, cp_path=new_cp_path, md_path=md_path)

    # this is belt & suspenders: validate that the experiment is in a consistent state.
    # HACK?
    for run in exp.runs:
        # print(f"after: run: epochs {run.checkpoint_nepochs}")
        cp_path = run.checkpoint_path
        if cp_path is None or not cp_path.exists():
            import sys
            print(f"!! {exp.shortcode} checkpoint_path is None or doesn't exist", file=sys.stderr)
            print(f"  exp.nepochs {exp.nepochs}", file=sys.stderr)
            print(f"   cp_nepochs {run.checkpoint_nepochs}", file=sys.stderr)
            print(f"      cp_path {cp_path.name if cp_path else None}", file=sys.stderr)
            print(f"  old_cp_path {old_cp_path.name if old_cp_path else None}", file=sys.stderr)
            print(f"  new_cp_path {new_cp_path.name}", file=sys.stderr)
            raise Exception(f"inconsistent state: invalid ExpRun for {exp.shortcode}")

    if old_cp_path is not None:
        old_cp_path.unlink()

"""
Load Experiment: metadata only.
"""
def load_from_metadata(md_path: Path) -> Experiment:
    with open(md_path, "r") as json_file:
        metadata = json.load(json_file)
    res = Experiment()
    res.metadata_path = md_path
    return res.load_model_dict(metadata)

"""
Save experiment metadata to .json
"""
def save_metadata(exp: Experiment, md_path: Path):
    # for run in exp.runs:
    #     cp_path = run.checkpoint_path
    #     if not cp_path:
    #         continue

    metadata_dict = exp.metadata_dict()

    temp_path = Path(str(md_path) + ".tmp")
    with open(temp_path, "w") as json_file:
        json.dump(metadata_dict, json_file, indent=2)
    temp_path.rename(md_path)

"""
Save experiment .ckpt and .json.
"""
def _save_checkpoint_and_metadata(exp: Experiment, cp_path: Path, md_path: Path):
    obj_fields_none = {field: (getattr(exp, field, None) is None) for field in experiment.OBJ_FIELDS}
    if any(obj_fields_none.values()):
        raise Exception(f"refusing to save {cp_path}: some needed fields are None: {obj_fields_none=}")

    model_dict = exp.model_dict()

    # save checkpoint before updating or saving the metadata
    temp_path = Path(str(cp_path) + ".tmp")
    with open(temp_path, "wb") as ckpt_file:
        torch.save(model_dict, ckpt_file)
    temp_path.rename(cp_path)

    save_metadata(exp, md_path)
