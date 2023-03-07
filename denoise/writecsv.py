import sys
from pathlib import Path
from typing import List
import re
import csv
import datetime

import torch

sys.path.append("..")
import model
from experiment import Experiment


def all_checkpoints(dir: Path, pattern: re.Pattern) -> List[Path]:
    res: List[Path] = list()
    for path in dir.iterdir():
        if path.is_dir():
            child_res = all_checkpoints(path, pattern)
            res.extend(child_res)
            continue

        if pattern.match(str(path)) is None:
            continue

        if path.name.endswith(".ckpt"):
            res.append(path)
    return res

def load_checkpoint(path: Path) -> Experiment:
    with open(path, "rb") as file:
        state_dict = torch.load(file)
        for k, v in state_dict["net"].items():
            if 'emblen' in k:
                print(f"{k=} {v=}")
        exp = Experiment.new_from_state_dict(state_dict)
        exp.net = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to("cuda")
    return exp

all_fields = "filename emblen nlinear hidlen nparams started_at ended_at elapsed samp_per_sec vloss".split(" ")

if __name__ == "__main__":
    pattern = re.compile(r".*encdec2.*")
    if len(sys.argv) > 1:
        pattern = re.compile(sys.argv[1])

    csv_file = open("encdec2.csv", "w")
    writer = csv.DictWriter(csv_file, fieldnames=all_fields)
    writer.writeheader()

    checkpoints = all_checkpoints(Path("runs"), pattern)
    for i, path in enumerate(checkpoints):
        print(f"{i + 1}/{len(checkpoints)}: {path}:")
        try:
            exp = load_checkpoint(path)
        except Exception as e:
            print(f"  \033[1;31m{e}\033[0m")
            continue

        net: model.ConvEncDec = exp.net
        nsamples = exp.nsamples
        elapsed = (exp.ended_at - exp.started_at).total_seconds()
        samp_per_sec = nsamples / elapsed
        row = dict(
            filename=str(path),
            emblen=net.emblen,
            nlinear=net.nlinear,
            hidlen=net.hidlen,
            nparams=exp.nparams(),
            started_at=exp.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            ended_at=exp.ended_at.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed=format(elapsed, ".2f"),
            samp_per_sec=format(samp_per_sec, ".2f"),
            vloss=format(exp.last_val_loss, ".3f"),
        )
        writer.writerow(row)
