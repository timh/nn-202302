import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re
import csv
import datetime
import tqdm
import argparse

import torch

sys.path.append("..")
import model
import loadsave
import experiment
from experiment import Experiment
import noised_data

def parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", default=None)
    parser.add_argument("--add_vloss", dest='add_vloss_types', type=str, nargs="+", choices=loss_types)
    parser.add_argument("--image_size", type=int, default=128)

    cfg = parser.parse_args()
    if cfg.pattern:
        cfg.pattern = re.compile(cfg.pattern)
    
    return cfg

# these fields are maintained here because I want them in a certain order.
all_fields = ("filename conv_descs emblen nlinear hidlen batch_size "
              "sched_type optim_type startlr endlr normalization loss_type "
              "nparams started_at saved_at "
              "nsamples nepochs max_epochs elapsed samp_per_sec "
              "tloss vloss").split(" ")
base_loss_types = "l1 l2 mse distance mape rpd".split(" ")
loss_types = base_loss_types.copy()
loss_types.extend(["edge+" + t for t in base_loss_types])
loss_types.extend(["edge*" + t for t in base_loss_types])

if __name__ == "__main__":
    cfg = parse_cmdline()
    if cfg.add_vloss_types:
        # NOTE: this is not necessarily the same batch/minicnt as the checkpoints we're
        # loading. so the l1 loss could even be different.
        batch_size = 64
        dataset = noised_data.load_dataset(image_dirname="alex-many-128", image_size=cfg.image_size)
        _train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size)

        vloss_fns = list()
        vloss_columns = list()

        for loss_type in cfg.add_vloss_types:
            vloss_fns.append(noised_data.twotruth_loss_fn(loss_type, device="cuda"))
            vloss_column = f"vloss_{loss_type}"
            vloss_columns.append(vloss_column)
            all_fields.append(vloss_column)

    writer = csv.DictWriter(sys.stdout, fieldnames=all_fields)
    writer.writeheader()

    checkpoints = loadsave.find_checkpoints(only_paths=cfg.pattern)
    for i, (cp_path, exp) in tqdm.tqdm(list(enumerate(checkpoints))):
        # net: model.ConvEncDec = exp.net
        with open(cp_path, "rb") as cp_file:
            state_dict = torch.load(cp_file)
        try:
            exp.net = model.ConvEncDec.new_from_state_dict(state_dict['net']).to("cuda")
        except Exception as e:
            print(f"error processing {cp_path}", file=sys.stderr)
            raise e

        nsamples = exp.nsamples

        elapsed = (exp.saved_at - exp.started_at).total_seconds()
        samp_per_sec = nsamples / elapsed
        saved_at = exp.saved_at.strftime(experiment.TIME_FORMAT)

        normalization: List[str] = []
        if getattr(exp, "do_batchnorm", None):
            normalization.append("batch")
        if getattr(exp, "do_layernorm", None):
            normalization.append("layer")
        normalization = ", ".join(normalization)

        row = dict(
            filename=str(cp_path),
            conv_descs=exp.conv_descs,
            emblen=exp.emblen, 
            nlinear=exp.nlinear, 
            hidlen=exp.hidlen,
            batch_size=exp.batch_size,

            sched_type=exp.sched_type,
            optim_type=exp.optim_type,
            startlr=format(exp.startlr, ".1E"),
            endlr=format(exp.startlr, ".1E"),

            normalization=normalization,
            loss_type=exp.loss_type,

            nparams=exp.nparams(), 
            started_at=exp.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            saved_at=saved_at,
            nsamples=exp.nsamples,
            nepochs=exp.nepochs,
            max_epochs=exp.max_epochs,
            elapsed=format(elapsed, ".2f"),
            samp_per_sec=format(samp_per_sec, ".2f"),

            tloss=format(exp.lastepoch_train_loss, ".5f"),
            vloss=format(exp.lastepoch_val_loss, ".5f")
        )

        if cfg.add_vloss_types:
            for vloss_fn, vloss_column in zip(vloss_fns, vloss_columns):
                exp.net.eval()

                vloss, cnt = 0.0, 0
                for inputs, truth in list(val_dl):
                    inputs, truth = inputs.to("cuda"), truth.to("cuda")
                    out = exp.net(inputs)
                    vloss += vloss_fn(out, truth).item()
                    cnt += 1

                row[vloss_column] = format(vloss / cnt, ".4f")

        writer.writerow(row)
