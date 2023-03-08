import sys
from pathlib import Path
from typing import List, Dict
import re
import csv
import datetime
import tqdm

import torch

sys.path.append("..")
import model
from experiment import Experiment
import denoise_logger
import noised_data


def all_checkpoints(dir: Path, pattern: re.Pattern) -> List[denoise_logger.CheckpointResult]:
    all_checkpoints = denoise_logger.find_all_checkpoints()
    return [checkpoint for checkpoint in all_checkpoints if pattern.match(str(checkpoint.path))]

def load_checkpoint(path: Path) -> Experiment:
    with open(path, "rb") as file:
        state_dict = torch.load(file)
        exp = Experiment.new_from_state_dict(state_dict)
        exp.net = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to("cuda")
    return exp

# these fields are maintained here because I want them in a certain order.
all_fields = ("filename conv_descs emblen nlinear hidlen batch_size "
              "sched_type normalization flat_conv2d "
              "nparams started_at ended_at "
              "nsamples epochs elapsed samp_per_sec "
              "tloss vloss l2_vloss").split(" ")

if __name__ == "__main__":
    pattern = re.compile(r".*encdec3.*")
    if len(sys.argv) > 1:
        pattern = re.compile(sys.argv[1])
    
    do_l2_vloss = False
    if do_l2_vloss:
        # NOTE: this is not necessarily the same batch/minicnt as the checkpoints we're
        # loading. so the l1 loss could even be different.
        batch_size = 64
        minicnt = 1
        dataset = noised_data.load_dataset(image_dirname="alex-many-128", image_size=128)
        _train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, minicnt=minicnt, val_all_data=False)
        loss_fn = noised_data.twotruth_loss_fn(False, "l2")

    writer = csv.DictWriter(sys.stdout, fieldnames=all_fields)
    writer.writeheader()

    checkpoints = all_checkpoints(Path("runs"), pattern)
    for i, cpres in tqdm.tqdm(list(enumerate(checkpoints))):
        try:
            exp = load_checkpoint(cpres.path)
        except Exception as e:
            print(e.with_traceback())
            print(f"  \033[1;31m{e}\033[0m", file=sys.stderr)
            continue

        if do_l2_vloss:
            exp.net.eval()
            l2_vloss = 0.0
            cnt = 0
            for inputs, truth in val_dl:
                inputs, truth = inputs.to("cuda"), truth.to("cuda")
                out = exp.net(inputs)
                loss = loss_fn(out, truth)
                l2_vloss += loss.item()
                cnt += 1
            l2_vloss /= cnt
        else:
            l2_vloss = 0.0

        net: model.ConvEncDec = exp.net
        nsamples = exp.nsamples
        elapsed = (exp.ended_at - exp.started_at).total_seconds()
        samp_per_sec = nsamples / elapsed

        normalization: List[str] = []
        if cpres.do_batch_norm:
            normalization.append("batch")
        if cpres.do_layer_norm:
            normalization.append("layer")
        normalization = ", ".join(normalization)

        row = dict(
            filename=str(cpres.path),
            conv_descs=cpres.conv_descs,
            emblen=net.emblen, 
            nlinear=net.nlinear, 
            hidlen=net.hidlen,
            batch_size=exp.batch_size,

            sched_type=cpres.sched_type, 
            normalization=normalization,
            flat_conv2d="yes" if net.do_flatconv2d else "no",

            nparams=exp.nparams(), 
            started_at=exp.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            ended_at=exp.ended_at.strftime("%Y-%m-%d %H:%M:%S"),
            nsamples=exp.nsamples,
            epochs=exp.nepochs + 1,
            elapsed=format(elapsed, ".2f"),
            samp_per_sec=format(samp_per_sec, ".2f"),

            tloss=format(exp.lastepoch_train_loss, ".3f"),
            vloss=format(exp.lastepoch_val_loss, ".3f"),
            l2_vloss=format(l2_vloss, ".3f")
        )
        writer.writerow(row)

