import sys
from pathlib import Path
from typing import List
import re
import csv
import datetime
import tqdm

import torch

sys.path.append("..")
import model
from experiment import Experiment
import denoise_logger


def all_checkpoints(dir: Path, pattern: re.Pattern) -> List[denoise_logger.CheckpointResult]:
    all_checkpoints = denoise_logger.find_all_checkpoints()
    return [checkpoint for checkpoint in all_checkpoints if pattern.match(str(checkpoint.path))]

def load_checkpoint(path: Path) -> Experiment:
    with open(path, "rb") as file:
        state_dict = torch.load(file)
        exp = Experiment.new_from_state_dict(state_dict)
        exp.net = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to("cuda")
    return exp

all_fields = "filename conv_descs emblen nlinear hidlen batch_size nparams started_at ended_at epochs elapsed samp_per_sec tloss vloss".split(" ")

if __name__ == "__main__":
    pattern = re.compile(r".*encdec3.*")
    if len(sys.argv) > 1:
        pattern = re.compile(sys.argv[1])

    writer = csv.DictWriter(sys.stdout, fieldnames=all_fields)
    writer.writeheader()

    checkpoints = all_checkpoints(Path("runs"), pattern)
    for i, cpres in tqdm.tqdm(list(enumerate(checkpoints))):
        # print(f"{i + 1}/{len(checkpoints)}: {path}", file=sys.stderr)
        try:
            exp = load_checkpoint(cpres.path)
        except Exception as e:
            print(e.with_traceback())
            print(f"  \033[1;31m{e}\033[0m", file=sys.stderr)
            continue

        net: model.ConvEncDec = exp.net
        nsamples = exp.nsamples
        elapsed = (exp.ended_at - exp.started_at).total_seconds()
        samp_per_sec = nsamples / elapsed
        row = dict(
            filename=str(cpres.path),
            conv_descs=cpres.conv_descs,
            emblen=net.emblen,
            nlinear=net.nlinear,
            hidlen=net.hidlen,
            nparams=exp.nparams(),
            batch_size=exp.batch_size,
            started_at=exp.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            ended_at=exp.ended_at.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed=format(elapsed, ".2f"),
            epochs=exp.nepochs + 1,
            samp_per_sec=format(samp_per_sec, ".2f"),
            tloss=format(exp.last_train_loss, ".3f"),
            vloss=format(exp.last_val_loss, ".3f"),
        )
        writer.writerow(row)

