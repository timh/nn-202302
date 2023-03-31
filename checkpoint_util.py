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

PathExpTup = Tuple[Path, Experiment]

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def list_checkpoints(runs_dir: Path = Path("runs"), 
                     only_one: bool = False) -> List[PathExpTup]:
    # TODO: return Experiments, not tuples. they have everything needed.

    # combine runs from old and new directory structure:
    # OLD: runs/{basename}-{timestamp}/checkpoints
    # NEW: runs/checkpoints-{basename}/{exp_base}
    all_cp_dirs: List[Path] = list()
    for runs_subdir in runs_dir.iterdir():
        if not runs_subdir.is_dir():
            continue

        # old layout.
        old_path = Path(runs_subdir, "checkpoints")
        if old_path.exists():
            all_cp_dirs.append(old_path)
            continue

        # new layout.
        if "checkpoints-" in runs_subdir.name:
            # directories within checkpoints-{basename} are subdirs with checkpoints in them.
            grand_subdirs = [path for path in runs_subdir.iterdir() if path.is_dir()]
            all_cp_dirs.extend(grand_subdirs)

    exp_by_shortcode: Dict[str, Experiment] = dict()
    cps_by_shortcode: Dict[str, List[Path]] = defaultdict(list)

    for cp_dir in all_cp_dirs:
        paths = [path for path in cp_dir.iterdir() if path.is_file()]
        cp_paths = {file for file in paths if file.name.endswith(".ckpt")}

        metadata_path = Path(cp_dir, "metadata.json")
        if metadata_path.exists():
            # new layout: associate the same metadata.json with all 
            # the checkpoints.
            exp = load_from_json(metadata_path)
            exp_by_shortcode[exp.shortcode] = exp
            for cp_path in cp_paths:
                cps_by_shortcode[exp.shortcode].append(cp_path)
        
        else:
            md_paths = [file for file in paths if file.name.endswith(".json")]

            for md_path in md_paths:
                md_base = str(md_path.name).replace(".json", "")

                found_ckpt: Path = None
                for cp_path in cp_paths:
                    if cp_path.name.startswith(md_base):
                        found_ckpt = cp_path
                        cp_paths.remove(cp_path)
                        break

                if found_ckpt is not None:
                    exp = load_from_json(md_path)
                    existing_exp = exp_by_shortcode.get(exp.shortcode, None)
                    if existing_exp is None or exp.nepochs > existing_exp.nepochs:
                        exp_by_shortcode[exp.shortcode] = exp
                    cps_by_shortcode[exp.shortcode].append(found_ckpt)
            
            for leftover in cp_paths:
                print(f"couldn't find .json for checkpoints:\n  {leftover}")

    # back compat: populate checkpoint_at, checkpoint_nepochs, checkpoint_path for all the runs
    _fix_runs(exp_by_shortcode=exp_by_shortcode, cps_by_shortcode=cps_by_shortcode)

    res: List[PathExpTup] = list()
    for shortcode in exp_by_shortcode.keys():
        exp = exp_by_shortcode[shortcode]
        cp_paths = cps_by_shortcode[shortcode]

        if only_one:
            best_run = exp.run_best_loss(loss_type='train_loss')
            res.append((best_run.checkpoint_path, exp))
            continue

        for cp_path in cp_paths:
            res.append((cp_path, exp))
    
    return res

def _fix_runs(exp_by_shortcode: Dict[str, Experiment], cps_by_shortcode: Dict[str, List[Path]]):
    for shortcode in exp_by_shortcode.keys():
        exp = exp_by_shortcode[shortcode]
        cp_paths = cps_by_shortcode[shortcode]

        if all([run.checkpoint_at and run.checkpoint_nepochs and run.checkpoint_path for run in exp.runs]):
            # this experiment is already setup correctly.
            continue

        for cp_path in cp_paths:
            nepochs = _get_checkpoint_nepochs(cp_path)
            if nepochs == len(exp.train_loss_hist):
                # back compat: some older metadata has nepochs + 1 written in it, instead of
                # nepochs. truncate.
                # print(f"{exp.created_at_short}-{exp.shortcode}: {nepochs=} but {len(exp.train_loss_hist)=}. clamping.")
                nepochs = len(exp.train_loss_hist) - 1

            run = exp.run_for_path(cp_path)
            if run is None:
                run = exp.run_for_nepochs(nepochs)
            if run is None:
                import sys
                # raise Exception(f"can't find run for {exp.shortcode=}, {nepochs=}")
                print(f"couldn't find run; using current", file=sys.stderr)
                print(f"  {nepochs=}", file=sys.stderr)
                print(f"  {exp.nepochs=}", file=sys.stderr)
                print(f"  {exp.created_at_short=}", file=sys.stderr)
                print(f"  {exp.shortcode=}", file=sys.stderr)
                print(f"  {str(cp_path)}", file=sys.stderr)
                print(f"  runs =", file=sys.stderr)
                for run in exp.runs:
                    field_strs: List[str] = list()
                    run_md = run.metadata_dict()
                    for field in 'nepochs max_epochs created_at'.split():
                        field_strs.append(f"{field}={run_md.get(field)}")
                    field_strs = ", ".join(field_strs)
                    print(f"    {field_strs}", file=sys.stderr)
                print(file=sys.stderr)
                run = exp.cur_run()

            cp_created_at = cp_path.lstat().st_ctime
            cp_created_at = datetime.datetime.fromtimestamp(cp_created_at)
            run.checkpoint_at = cp_created_at
            run.checkpoint_nepochs = nepochs
            run.checkpoint_path = cp_path
        
# HACK: this filename pattern should be centralized.
RE_CP_FILENAME = re.compile(r".*epoch_(\d+)[^\d].*")

def _get_checkpoint_nepochs(cp_path: Path) -> int:
    match = RE_CP_FILENAME.match(cp_path.name)
    if match:
        return int(match.group(1))
    raise Exception(f"can't determine nepochs for {cp_path}")



"""
Given a list of checkpoints, return a set of root/children indexes.
Returned indexes are indexes into the input checkpoints list.

Returns:
    List[
        Tuple[PathExpTup, List[PathExpTup]]
    ]

each entry is a root (Path, Experiment), and its offspring List[(Path, Experiment)]
"""
def find_resume_roots(checkpoints: List[PathExpTup]) -> List[Tuple[PathExpTup, List[PathExpTup]]]:
    # index (key) is the same as (values indexes)
    # key = inner, values = outer
    paths, exps = zip(*checkpoints)

    path_idx = {path: idx for idx, path in enumerate(paths)}
    parent_for_idx: Dict[int, int] = dict()

    for expidx, exp in enumerate(exps):
        lastrun = exp.cur_run()
        if not lastrun.resumed_from:
            continue

        resumed_from_idx = path_idx.get(lastrun.resumed_from, None)
        if not resumed_from_idx:
            continue
        parent_for_idx[expidx] = resumed_from_idx

    def find_root(idx: int):
        if idx not in parent_for_idx:
            return idx
        idx = parent_for_idx[idx]
        return find_root(idx)

    # figure out roots/list of offspring for indexes.
    roots_and_offspring: Dict[int, List[int]] = defaultdict(list)
    for expidx, exp in enumerate(exps):
        root = find_root(expidx)
        roots_and_offspring[root].append(expidx)
    
    # convert to result form.
    res: List[Tuple[PathExpTup, List[PathExpTup]]] = list()
    for root_idx, offspring_idxs in roots_and_offspring.items():
        offspring_idxs = sorted(offspring_idxs, key=lambda idx: exps[idx].saved_at)

        root_tup = (paths[root_idx], exps[root_idx])
        offspring_tups = [(paths[off_idx], exps[off_idx]) for off_idx in offspring_idxs]
        res.append((root_tup, offspring_tups))

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
def resume_experiments(exps_in: List[Experiment], 
                       max_epochs: int,
                       checkpoints: List[PathExpTup] = None,
                       extra_ignore_fields: Set[str] = None) -> List[Experiment]:
    if extra_ignore_fields is None:
        extra_ignore_fields = set()

    resume_exps: List[Experiment] = list()
    if checkpoints is None:
        checkpoints = list_checkpoints()
    
    exps_in_shortcodes: Set[str] = set()
    for exp_in in exps_in:
        # start the experiment, to make it fully instantiate fields of net, sched,
        # optim and possibly others. then create a meta-ony Experiment out of it,
        # to capture the net/etc fields that get copied from the sub-objects, like
        # net_class
        exp_in.start(0)
        exp_in_meta = Experiment().load_model_dict(exp_in.metadata_dict())

        if exp_in_meta.shortcode in exps_in_shortcodes:
            raise ValueError(f"duplicate incoming experiments: {exp_in_meta.shortcode=}")
        exps_in_shortcodes.add(exp_in_meta.shortcode)

        match_exp: Experiment = None
        cp_matching = [(path, exp) for path, exp in checkpoints if exp_in_meta.shortcode == exp.shortcode]
        for cp_path, cp_exp in cp_matching:
            # the checkpoint experiment won't have its lazy functions set. but we 
            # know based on above sameness comparison that the *type* of those
            # functions is the same. So, set the lazy functions based on the 
            # exp_in's, which has already been setup by the prior loop before
            # being passed in.
            cp_exp.max_epochs = max_epochs
            cp_exp.prepare_resume(cp_path=cp_path, new_exp=exp_in)

            if match_exp is None or cp_exp.nepochs > match_exp.nepochs:
                match_exp = cp_exp
                match_path = cp_path

        if match_exp is not None:
            checkpoints.remove((match_path, match_exp))

            # BUG: there are off-by-one errors around "nepochs" all over.
            if match_exp.nepochs >= (max_epochs - 1):
                print(f"* \033[1;31mskipping {match_exp.label}: checkpoint already has {match_exp.nepochs} epochs\033[0m")
                continue
            if match_exp.max_epochs >= (max_epochs - 1) and match_exp.cur_run().finished:
                print(f"* \033[1;31mskipping {match_exp.label}: checkpoint finished at {match_exp.max_epochs} epochs\033[0m")
                continue
            # TODO: can't do this because 'finished' isn't a real attribute. it's
            # the presence of a '.status' file in a run directory..somewhere.
            # if match_exp.finished and match_exp.max_epochs >= max_epochs:
            #     print(f"* \033[1;31mskipping {match_exp.label}: checkpoint is finished and had {match_exp.max_epochs} max_epochs\033[0m")
            #     continue

            print(f"* \033[1;32mresuming {match_exp.label}: using checkpoint with {match_exp.nepochs} epochs\033[0m")
            resume_exps.append(match_exp)

            with open(match_path, "rb") as file:
                state_dict = torch.load(file)

            match_exp.lazy_net_fn = _resume_lazy_fn(match_exp, state_dict['net'], match_exp.lazy_net_fn)
            match_exp.lazy_optim_fn = _resume_lazy_fn(match_exp, state_dict['optim'], match_exp.lazy_optim_fn)

            # NOTE: do NOT resume scheduler. this will wipe out any learning rate changes we've made.
            # match_exp.lazy_sched_fn = _resume_lazy_fn(match_exp, state_dict['sched'], match_exp.lazy_sched_fn)
        else:
            exp_in.net.cpu()
            exp_in.net = None
            exp_in.sched = None
            exp_in.optim = None
            print(f"* \033[1mcouldn't find resume checkpoint for {exp_in.label}; starting a new one\033[0m")
            resume_exps.append(exp_in)

    return resume_exps

"""
Load Experiment: metadata only.
"""
def load_from_json(json_path: Path) -> Experiment:
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)
    return Experiment().load_model_dict(metadata)

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
