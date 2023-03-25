# %%
import sys
import datetime
import argparse
from typing import Tuple, List, Dict
from pathlib import Path
import csv

import torch
import torchsummary

sys.path.append("..")
import model_util
import experiment
from experiment import Experiment
import dn_util
import cmdline

class Config(cmdline.QueryConfig):
    show_net: bool
    show_summary: bool
    show_raw: bool
    output_csv: bool

    only_diff: bool
    only_filenames: bool
    fields: List[str]
    addl_fields: List[str] = list()

    def __init__(self):
        super().__init__()
        self.add_argument("-n", "--net", dest='show_net', action='store_true', default=False)
        self.add_argument("-S", "--summary", dest='show_summary', action='store_true', default=False)
        self.add_argument("--raw", dest='show_raw', default=False, action='store_true')
        self.add_argument("-d", "--only_diff", "--diff", action='store_true', default=False, help="only show fields that changed from the prior entry")
        self.add_argument("-f", "--fields", type=str, nargs='+', help="only list these fields, or in --only_diff, add these fields")
        self.add_argument("-F", "--filenames", dest='only_filenames', default=False, action='store_true')
        self.add_argument("--csv", dest='output_csv', default=False, action='store_true')
    
    def parse_args(self) -> 'Config':
        super().parse_args()

        # if only_diff is set, treat --fields as additional fields.
        if self.only_diff and self.fields:
            self.addl_fields = self.fields.copy()
            self.fields = list()
        
        if self.output_csv and any([self.show_net, self.show_summary, self.show_raw]):
            self.error(f"--output_csv can't be used with --net, --summary, or --raw")

def fields_to_str(exp_fields: Dict[str, any]) -> Dict[str, str]:
    res: Dict[str, str] = dict()
    for field, val in exp_fields.items():
        valstr = str(val)                

        if isinstance(val, float):
            if 'lr' in field:
                valstr = format(val, ".1E")
            elif 'kld_weight' in field:
                valstr = format(val, ".2E")
            else:
                valstr = format(val, ".5f")
        elif val is None:
            valstr = ""
        res[field] = valstr
    
    return res

def print_row_human(cfg: Config, 
                    exp_fields: Dict[str, any],
                    last_values: Dict[str, any],
                    last_values_str: Dict[str, str]):
    max_field_len = max([len(field) for field in exp_fields.keys()])
    green = "\033[1;32m"
    red = "\033[1;31m"
    other = "\033[1;35m"

    ignorediff_fields = set('nsamples started_at ended_at saved_at resumed_at saved_at_relative '
                            'nepochs nbatches nsamples exp_idx elapsed label'.split())

    exp_fields_str = fields_to_str(exp_fields)
    for field in exp_fields.keys():
        val = exp_fields[field]
        valstr = exp_fields_str[field]

        fieldstr = field.rjust(max_field_len)
        last_val = last_values.get(field, val)

        if val != last_val and field not in ignorediff_fields:
            last_val_str = last_values_str.get(field)
            if last_val_str:
                last_val_str = f" ({last_val_str})"

            if 'loss' in field:
                scolor = red if val > last_val else green
            else:
                scolor = other
            # print(f"  {scolor}{fieldstr} = {valstr}\033[0m{last_val_str}")
            print(f"  {scolor}{fieldstr} = {valstr}\033[0m")
        elif not cfg.only_diff or field in cfg.addl_fields:
            print(f"  {fieldstr} = {valstr}")

        last_values[field] = val
        last_values_str[field] = valstr

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    now = datetime.datetime.now()

    checkpoints = cfg.list_checkpoints()
    if cfg.sort_key and 'loss' in cfg.sort_key:
        # show the lowest loss at the end.
        checkpoints = list(reversed(checkpoints))
    
    if cfg.output_csv:
        checkpoints_fields: List[Dict[str, any]] = list()

    last_values: Dict[str, any] = dict()
    last_values_str: Dict[str, str] = dict()

    for cp_idx, (path, exp) in enumerate(checkpoints):
        exp: Experiment

        if cfg.only_filenames:
            base = str(path.name).replace(".ckpt", "")
            for path in path.parent.iterdir():
                if not path.name.startswith(base):
                    continue
                print(path)
            continue

        if not cfg.output_csv and not cfg.only_filenames:
            print()
            print(f"{cp_idx + 1}/{len(checkpoints)}")
            print(f"{path}:")

        if not cfg.show_raw:
            start = exp.started_at.strftime(experiment.TIME_FORMAT) if exp.started_at else ""
            end = exp.ended_at.strftime(experiment.TIME_FORMAT) if exp.ended_at else ""

            status_file = Path(path.parent.parent, exp.label + ".status")
            finished = status_file.exists()

            exp_fields = exp.metadata_dict(update_saved_at=False)
            exp_fields['finished'] = finished

            if cfg.fields:
                exp_fields = {field: val for field, val in exp_fields.items() if field in cfg.fields}

            # if 'val_loss_hist' in exp_fields:
            #     val_loss_hist = [f"{epoch}:{loss:.3f}" 
            #                      for epoch, loss in exp.val_loss_hist[-5:]]
            #     exp_fields['val_loss_hist'] = ", ".join(val_loss_hist)
            # if 'train_loss_hist' in exp_fields:
            #     train_loss_hist = [f"{loss:.3f}" 
            #                        for loss in exp.train_loss_hist[-5:]]
            #     exp_fields['train_loss_hist'] = ", ".join(train_loss_hist)

            exp_fields.pop('val_loss_hist', None)
            exp_fields.pop('train_loss_hist', None)

            if cfg.output_csv:
                exp_fields['path'] = str(path)
                exp_fields_str = fields_to_str(exp_fields)
                checkpoints_fields.append(exp_fields_str)
            else:
                print_row_human(cfg, exp_fields, last_values, last_values_str)
        
        if cfg.show_net or cfg.show_summary or cfg.show_raw:
            with open(path, "rb") as ckpt_file:
                model_dict = torch.load(path)
            if cfg.show_raw:
                print("{")
                model_util.print_dict(model_dict, 1)
                print("}")

            net = dn_util.load_model(model_dict).to('cuda')
            
            if cfg.show_net:
                print(net)
            
            if cfg.show_summary:
                net.to("cuda")
                size = (exp.nchannels, exp.image_size, exp.image_size)
                inputs = torch.rand(size, device="cuda")
                torchsummary.summary(net, input_size=size, batch_size=1)

        if not cfg.output_csv:
            print()
    
    if cfg.output_csv:
        fnames_set = {field
                      for cp_fields in checkpoints_fields
                      for field in cp_fields.keys()}
        
        first_fields = ('path lastepoch_train_loss lastepoch_val_loss lastepoch_kl_loss '
                        'nepochs max_epochs '
                        'net_layers_str loss_type saved_at_relative elapsed').split()
        first_fields_add: List[str] = list()
        for field in first_fields:
            if field in fnames_set:
                fnames_set.remove(field)
                first_fields_add.append(field)
        
        fnames = first_fields_add + list(sorted(fnames_set))

        writer = csv.DictWriter(sys.stdout, fieldnames=fnames)
        writer.writeheader()
        writer.writerows(checkpoints_fields)
