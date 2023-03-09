# %%
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
import experiment
from experiment import Experiment
import denoise_logger
import noised_data


def all_checkpoints(dir: Path, cfg: argparse.Namespace) -> List[Tuple[Path, Experiment]]:
    all_checkpoints = denoise_logger.find_all_checkpoints(dir)
    if cfg.pattern:
        all_checkpoints = [(path, exp) for path, exp in all_checkpoints if cfg.pattern.match(str(path))]
    return all_checkpoints

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
              "sched_type normalization flat_conv2d "
              "nparams started_at ended_at "
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

    for i, (cp_path, exp) in tqdm.tqdm(list(enumerate(all_checkpoints(Path("runs"), cfg)))):
        # net: model.ConvEncDec = exp.net
        with open(cp_path, "rb") as cp_file:
            state_dict = torch.load(cp_file)
        exp.net = model.ConvEncDec.new_from_state_dict(state_dict['net']).to("cuda")

        nsamples = exp.nsamples
        if exp.ended_at:
            elapsed = (exp.ended_at - exp.started_at).total_seconds()
            samp_per_sec = nsamples / elapsed
            ended_at = exp.ended_at.strftime(experiment.TIME_FORMAT)
        else:
            elapsed = 0
            samp_per_sec = 0
            curtime = getattr(exp, 'curtime', None)
            if curtime:
                ended_at = curtime.strftime(experiment.TIME_FORMAT)
            else:
                ended_at = ""

        normalization: List[str] = []
        if getattr(exp, "do_batch_norm", None):
            normalization.append("batch")
        if getattr(exp, "do_layer_norm", None):
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
            normalization=normalization,
            flat_conv2d="yes" if exp.do_flatconv2d else "no",

            nparams=exp.nparams(), 
            started_at=exp.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            ended_at=ended_at,
            nsamples=exp.nsamples,
            nepochs=exp.nepochs + 1,
            max_epochs=exp.max_epochs,
            elapsed=format(elapsed, ".2f"),
            samp_per_sec=format(samp_per_sec, ".2f"),

            tloss=format(exp.lastepoch_train_loss, ".4f"),
            vloss=format(exp.lastepoch_val_loss, ".4f")
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


# %%
