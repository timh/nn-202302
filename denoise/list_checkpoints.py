# %%
import sys
import datetime
import argparse
from typing import Tuple, Dict
from pathlib import Path

import torch
import torchsummary

sys.path.append("..")
import model_util
import experiment
from experiment import Experiment
from denoise_exp import DNExperiment
import model
import dn_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", type=str, default=None)
    parser.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])
    parser.add_argument("-n", "--net", dest='show_net', action='store_true', default=False)
    parser.add_argument("-S", "--summary", dest='show_summary', action='store_true', default=False)
    parser.add_argument("-s", "--sort", default='time', choices="nepochs max_epochs val_loss train_loss vloss tloss time".split(" "))
    parser.add_argument("--raw", dest='show_raw', default=False, action='store_true')

    cfg = parser.parse_args()
    if cfg.pattern:
        import re
        cfg.pattern = re.compile(cfg.pattern)
    
    checkpoints = model_util.find_checkpoints(only_paths=cfg.pattern, attr_matchers=cfg.attribute_matchers)

    if cfg.sort:
        def key_fn(cp: Tuple[Path, DNExperiment]) -> any:
            path, exp = cp
            if cfg.sort in ["val_loss", "train_loss", "vloss", "tloss"]:
                if cfg.sort in ["val_loss", "vloss"]:
                    key = "lastepoch_val_loss"
                else:
                    key = "lastepoch_train_loss"
                return -getattr(exp, key)
            elif cfg.sort == "time":
                val = exp.ended_at if exp.ended_at else exp.saved_at
                return val
            return getattr(exp, cfg.sort)
        checkpoints = sorted(checkpoints, key=key_fn)

    last_values: Dict[str, any] = dict()
    for cp_idx, (path, exp) in enumerate(checkpoints):
        exp: DNExperiment
        print()
        print(f"{cp_idx + 1}/{len(checkpoints)}")
        print(f"{path}:")

        if not cfg.show_raw:
            start = exp.started_at.strftime(experiment.TIME_FORMAT) if exp.started_at else ""
            end = exp.ended_at.strftime(experiment.TIME_FORMAT) if exp.ended_at else ""
            relative, saved_at = "", ""
            if exp.saved_at:
                saved_at = exp.saved_at.strftime(experiment.TIME_FORMAT)
                relative = int((datetime.datetime.now() - exp.saved_at).total_seconds())
            
            status_file = Path(path.parent.parent, exp.label + ".status")
            finished = status_file.exists()

            fields = ("label nepochs max_epochs batch_size nsamples val_loss train_loss "
                      "started_at ended_at saved_at relative "
                      "optim_type sched_type startlr endlr loss_type "
                      "finished").split()
            fields.extend([f for f in dir(exp) if f.startswith("net_")])

            for field in fields:
                if field == "relative":
                    val = f"{relative}s ago"
                elif field == "finished":
                    val = finished
                elif field in ["train_loss", "val_loss"]:
                    field = "lastepoch_" + field
                    val = getattr(exp, field)
                else:
                    val = getattr(exp, field)

                valstr = str(val)                
                if isinstance(val, datetime.datetime):
                    valstr = val.strftime(experiment.TIME_FORMAT)
                elif isinstance(val, float):
                    valstr = format(val, ".4f")
                elif val is None:
                    valstr = ""

                last_val = last_values.get(field, val)
                if val != last_val and field not in 'nsamples started_at ended_at saved_at relative nepochs'.split():
                    if True or type(val) in [int, float]:
                        if val < last_val:
                            scolor = "\033[1;31m"
                        else:
                            scolor = "\033[1;32m"
                    else:
                        scolor = f"\033[1m"
                    print(f"  {scolor}{field:20} = {valstr}\033[0m")
                else:
                    print(f"  {field:20} = {valstr}")
                last_values[field] = val

        if cfg.show_net or cfg.show_summary or cfg.show_raw:
            with open(path, "rb") as ckpt_file:
                model_dict = torch.load(path)
            if cfg.show_raw:
                print("{")
                model_util.print_dict(model_dict, 1)
                print("}")

            net = dn_util.load_model(model_dict).to('cuda')
            
            # net = model.ConvEncDec.new_from_model_dict(model_dict['net'])
            # net = t
            if cfg.show_net:
                print(net)
            
            if cfg.show_summary:
                net.to("cuda")
                size = (exp.nchannels, exp.image_size, exp.image_size)
                inputs = torch.rand(size, device="cuda")
                torchsummary.summary(net, input_size=size, batch_size=1)

        print()
