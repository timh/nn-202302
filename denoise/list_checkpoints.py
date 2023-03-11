# %%
import sys
import datetime
import torch
import argparse
from typing import Tuple
from pathlib import Path

import torchsummary

sys.path.append("..")
import loadsave
import experiment
from experiment import Experiment
from denoise_exp import DNExperiment
import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", type=str, default=None)
    parser.add_argument("-n", "--net", dest='show_net', action='store_true', default=False)
    parser.add_argument("-S", "--summary", dest='show_summary', action='store_true', default=False)
    parser.add_argument("-s", "--sort", default='time', choices="nepochs max_epochs val_loss train_loss time".split(" "))

    cfg = parser.parse_args()
    checkpoints = loadsave.find_checkpoints(only_paths=cfg.pattern)

    if cfg.sort:
        def key_fn(cp: Tuple[Path, DNExperiment]) -> any:
            path, exp = cp
            if cfg.sort in ["val_loss", "train_loss"]:
                if cfg.sort == "val_loss":
                    key = "lastepoch_val_loss"
                else:
                    key = "lastepoch_train_loss"
                return -getattr(exp, key)
            elif cfg.sort == "time":
                val = exp.ended_at if exp.ended_at else exp.saved_at
                return val
            return getattr(exp, cfg.sort)
        checkpoints = sorted(checkpoints, key=key_fn)

    for path, exp in checkpoints:
        exp: DNExperiment
        print(f"{path}:")
        start = exp.started_at.strftime(experiment.TIME_FORMAT) if exp.started_at else ""
        end = exp.ended_at.strftime(experiment.TIME_FORMAT) if exp.ended_at else ""
        relative, saved_at = "", ""
        if exp.saved_at:
            saved_at = exp.saved_at.strftime(experiment.TIME_FORMAT)
            relative = int((datetime.datetime.now() - exp.saved_at).total_seconds())
        
        status_file = Path(path.parent.parent, exp.label + ".status")
        finished = status_file.exists()

        print(f"   net_class: {exp.net_class}")
        print(f"       label: {exp.label}")
        print(f"     nepochs: {exp.nepochs}")
        print(f"  max_epochs: {exp.max_epochs}")
        print(f"    nsamples: {exp.nsamples}")
        print(f"    val_loss: {exp.lastepoch_val_loss:.5f}")
        print(f"  train_loss: {exp.lastepoch_train_loss:.5f}")
        print(f"  started_at: {start}")
        print(f"    ended_at: {end}")
        print(f"    saved_at: {saved_at}")
        print(f"    relative: {relative}s ago")
        print(f"       optim: {exp.optim_type}")
        print(f"       sched: {exp.sched_type} @ LR {exp.startlr:.1E} - {exp.endlr:.1E}")
        print(f"   loss_type: {exp.loss_type}")
        print(f"    finished: {finished}")

        if cfg.show_net or cfg.show_summary:
            with open(path, "rb") as ckpt_file:
                state_dict = torch.load(path)
            
            net = model.ConvEncDec.new_from_state_dict(state_dict['net'])
            if cfg.show_net:
                print(net)
            
            if cfg.show_summary:
                net.to("cuda")
                size = (exp.nchannels, exp.image_size, exp.image_size)
                inputs = torch.rand(size, device="cuda")
                torchsummary.summary(net, input_size=size, batch_size=1)

        print()
