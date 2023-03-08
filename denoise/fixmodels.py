# %%
import sys
import re
from typing import Dict
from pathlib import Path
import tqdm
import torch

import denoise_logger

sys.path.append("..")
from experiment import Experiment
from denoise_logger import CheckpointResult

RE_BATCH = re.compile(r".*batch_(\d+).*")
def process_one(cres: CheckpointResult, state_dict: Dict[str, any]):
    # exp = Experiment.new_from_state_dict(state_dict)
    do_save = False

    match = RE_BATCH.match(cres.path.name)
    if match:
        batch_in_path = int(match.group(1))
        batch_in_state_dict = state_dict.get("batch_size", None)
        if not batch_in_state_dict:
            print(f"  update batch_size from {batch_in_state_dict} to {batch_in_path}")
            state_dict["batch_size"] = batch_in_path
            do_save = True
    
    if "do_flatconv2d" not in state_dict["net"]:
        print(f"  add do_flatconv2d = False")
        state_dict["net"]["do_flatconv2d"] = False
        do_save = True
    
    if "do_flat_conv2d" in state_dict["net"]:
        print(f"  remove typo field do_flat_conv2d")
        del state_dict["net"]["do_flat_conv2d"]
        do_save = True


    if do_save:
        print(f"  update checkpoint {cres.path}")
        with open(cres.path, "wb") as file:
            torch.save(state_dict, file)

if __name__ == "__main__":
    for cres in tqdm.tqdm(denoise_logger.find_all_checkpoints()):
        # print(cres.path)
        with open(cres.path, "rb") as file:
            state_dict = torch.load(file)
            process_one(cres, state_dict)

