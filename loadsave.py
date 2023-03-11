from typing import Sequence, List, Tuple, Union, Dict
from pathlib import Path
import json
import re

import torch

import experiment
from experiment import Experiment

"""
Find all checkpoints in the given directory. Loads the experiments' metadata and returns it,
along with the path to the .ckpt file.

only_net_classes: Only return checkpoints with any of the given net_class values.
only_paths: Only return checkpoints matching the given string or regex pattern in their path.
"""
def find_checkpoints(runsdir: Path = Path("runs"), 
                     only_net_classes: Sequence[str] = None,
                     only_paths: Union[str, re.Pattern] = None) -> List[Tuple[Path, Experiment]]:
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
            if only_net_classes and exp.net_class not in only_net_classes:
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
Loads from either a state_dict or metadata_dict.

NOTE: this cannot load the Experiment's subclasses as it doesn't know how to
      instantiate them. They could come from any module.
"""
def load_from_dict(state_dict: Dict[str, any]) -> Experiment:
    exp = Experiment(label=state_dict['label'])
    return exp.load_state_dict(state_dict)

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

    state_dict = exp.state_dict()
    with open(ckpt_path, "wb") as ckpt_file:
        torch.save(state_dict, ckpt_file)
    
    save_metadata(exp, json_path)
