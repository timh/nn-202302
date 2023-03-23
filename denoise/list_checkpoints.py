# %%
import sys
import datetime
import argparse
from typing import Tuple, List, Dict
from pathlib import Path

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

    only_diff: bool
    fields: List[str]
    addl_fields: List[str] = list()

    def __init__(self):
        super().__init__()
        self.add_argument("-n", "--net", dest='show_net', action='store_true', default=False)
        self.add_argument("-S", "--summary", dest='show_summary', action='store_true', default=False)
        self.add_argument("-d", "--only_diff", action='store_true', default=False, help="only show fields that changed from the prior entry")
        self.add_argument("-f", "--fields", type=str, nargs='+', help="only list these fields, or in --only_diff, add these fields")
        self.add_argument("--raw", dest='show_raw', default=False, action='store_true')
    
    def parse_args(self) -> 'Config':
        super().parse_args()

        # if only_diff is set, treat --fields as additional fields.
        if self.only_diff and self.fields:
            self.addl_fields = self.fields.copy()
            self.fields = list()


if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    last_values: Dict[str, any] = dict()
    last_value_strs: Dict[str, str] = dict()
    now = datetime.datetime.now()
    checkpoints = cfg.list_checkpoints()
    for cp_idx, (path, exp) in enumerate(checkpoints):
        exp: Experiment
        print()
        print(f"{cp_idx + 1}/{len(checkpoints)}")
        print(f"{path}:")

        if not cfg.show_raw:
            start = exp.started_at.strftime(experiment.TIME_FORMAT) if exp.started_at else ""
            end = exp.ended_at.strftime(experiment.TIME_FORMAT) if exp.ended_at else ""
            relative = ""

            if exp.saved_at:
                saved_at = exp.saved_at.strftime(experiment.TIME_FORMAT)
                total_seconds = int((now - exp.saved_at).total_seconds())
                seconds = total_seconds % 60
                minutes = (total_seconds // 60)  % 60
                hours = total_seconds // (60 * 60)
                rel_list = [(hours, "h"), (minutes, "m"), (seconds, "s")]
                rel_list = [f"{val}{short}" for val, short in rel_list if val]
                relative = " ".join(rel_list)

            status_file = Path(path.parent.parent, exp.label + ".status")
            finished = status_file.exists()

            fields = exp.metadata_dict(update_saved_at=False)
            fields['relative'] =f"{relative} ago"
            fields['finished'] = finished

            if cfg.fields:
                fields = {field: val for field, val in fields.items() if field in cfg.fields}
            
            max_field_len = max([len(field) for field in fields.keys()])

            ignorediff_fields = set('nsamples started_at ended_at saved_at resumed_at relative '
                                    'nepochs nbatches nsamples exp_idx elapsed label'.split())
            green = "\033[1;32m"
            red = "\033[1;31m"
            other = "\033[1;35m"
            for field, val in fields.items():
                fieldstr = field.rjust(max_field_len)
                valstr = str(val)                

                if field in {'val_loss_hist', 'train_loss_hist'}:
                    continue

                if isinstance(val, float):
                    if 'lr' in field:
                        valstr = format(val, ".1E")
                    else:
                        valstr = format(val, ".4f")
                elif val is None:
                    valstr = ""

                last_val = last_values.get(field, val)
                if val != last_val and field not in ignorediff_fields:
                    last_val_str = last_value_strs.get(field)
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
                last_value_strs[field] = valstr

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

        print()
