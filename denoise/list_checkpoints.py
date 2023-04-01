# %%
import sys
import datetime
from typing import List, Dict
from collections import OrderedDict
import csv

import torch
import torchsummary

sys.path.append("..")
import model_util
from experiment import Experiment
import dn_util
import cmdline
import model_util

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

def fields_to_str(exp_fields: Dict[str, any], max_field_len: int) -> Dict[str, str]:
    res: Dict[str, str] = OrderedDict()
    for field, val in exp_fields.items():
        if field == 'runs':
            # put the 'runs' field last.
            continue
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

    if 'runs' in exp_fields:
        # pretty print runs
        run_pad = " " * (max_field_len + 5)
        run_fields_lines = [
            'started_at saved_at ended_at finished'.split(),
            'checkpoint_nepochs checkpoint_at'.split(),
            'checkpoint_path'.split(),
            'batch_size do_compile'.split(),
            'startlr endlr optim_type sched_type sched_warmup_epochs'.split(),
            'resumed_from'.split()
        ]

        res['runs'] = ""
        run_strs: List[str] = list()
        for run_idx, one_run in enumerate(exp_fields['runs']):
            run_lines: List[str] = list()
            for run_line_no, run_fields in enumerate(run_fields_lines):
                one_line: List[str] = list()
                for rfield in run_fields:
                    rval = one_run.get(rfield)
                    if rval is None:
                        continue
                    one_line.append(f"{rfield} {rval}")
                
                if not len(one_line):
                    continue

                # pad the lines appropriatlely:
                # first run, first line: no padding
                # other run, first line: padding to align with 'value' column
                #   any run,   > line 3: indented from 'value' column
                one_line = ", ".join(one_line)
                if run_idx == 0 and run_line_no == 0:
                    one_line = "- " + one_line
                elif run_line_no == 0:
                    one_line = run_pad + "- " + one_line
                else:
                    one_line = run_pad + "  " + one_line
                
                run_lines.append(one_line)
            
            # join the lines of a single run together    
            run_strs.append("\n".join(run_lines))

        # join the runs together, and add an extra newline
        res['runs'] += "\n".join(run_strs)
    
    return res

def print_row_human(cfg: Config, 
                    exp_fields: Dict[str, any],
                    last_values: Dict[str, any],
                    last_values_str: Dict[str, str]):
    max_field_len = max([len(field) for field in exp_fields.keys()])
    green = "\033[1;32m"
    red = "\033[1;31m"
    other = "\033[1;35m"

    ignorediff_fields = set('nbatches nsamples runs '
                            'started_at saved_at saved_at_relative ended_at '
                            'exp_idx elapsed label'.split())

    exp_fields_str = fields_to_str(exp_fields, max_field_len)
    for field in exp_fields_str.keys():
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

    exps = cfg.list_experiments()
    if cfg.sort_key and 'loss' in cfg.sort_key:
        # show the lowest loss at the end.
        exps = list(reversed(exps))
    
    if cfg.output_csv:
        csv_fields: List[Dict[str, any]] = list()

    last_values: Dict[str, any] = dict()
    last_values_str: Dict[str, str] = dict()

    for exp_idx, exp in enumerate(exps):
        md_path = exp.metadata_path
        if cfg.only_filenames:
            for run in exp.runs:
                print(run.checkpoint_path)
            continue

        if not cfg.output_csv and not cfg.only_filenames:
            print()
            print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}")
            print(f"{md_path}:")

        if not cfg.show_raw:
            start = exp.started_at.strftime(model_util.TIME_FORMAT) if exp.started_at else ""
            end = exp.ended_at.strftime(model_util.TIME_FORMAT) if exp.ended_at else ""

            exp_fields = exp.metadata_dict(update_saved_at=False)

            nloss = 5
            exp_fields['val_loss_hist'] = "... " + ", ".join(f"{vloss:.5f}" for _epoch, vloss in exp.val_loss_hist[-nloss:])
            exp_fields['train_loss_hist'] = "... " + ", ".join(f"{tloss:.5f}" for tloss in exp.train_loss_hist[-nloss:])
            exp_fields['best_train_loss'] = f"{exp.best_train_loss:.5f} @ {exp.best_train_epoch}"
            exp_fields['best_val_loss'] = f"{exp.best_val_loss:.5f} @ {exp.best_val_epoch}"
            exp_fields.pop('best_train_epoch', None)
            exp_fields.pop('best_val_epoch', None)
            exp_fields.pop('lr_hist', None)

            for field, val in list(exp_fields.items()):
                # convert e.g., 
                #   net_args = {class=foo, dim=2}
                # into 
                #   net.class = foo
                #   net.dim = 2
                if field.endswith("_args") and isinstance(val, dict):
                    objname = field[:-5]
                    val = {f"{objname}.{dfield}": dval for dfield, dval in val.items() 
                           if not cfg.fields or dfield in cfg.fields}
                    exp_fields.update(val)
                    exp_fields.pop(field)

            if cfg.fields:
                exp_fields = {field: val for field, val in exp_fields.items() if field in cfg.fields}

            if cfg.output_csv:
                exp_fields['path'] = str(md_path)
                exp_fields_str = fields_to_str(exp_fields)
                csv_fields.append(exp_fields_str)
            else:
                print_row_human(cfg, exp_fields, last_values, last_values_str)
        
        if cfg.show_net or cfg.show_summary or cfg.show_raw:
            last_path = exp.cur_run().checkpoint_path
            model_dict = torch.load(last_path, map_location='cpu')
            if cfg.show_raw:
                print("{")
                model_util.print_dict(model_dict, 1)
                print("}")

            net = dn_util.load_model(model_dict) #.to('cuda')
            
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
                      for cp_fields in csv_fields
                      for field in cp_fields.keys()}
        
        first_fields = ('path last_train_loss last_val_loss last_kl_loss '
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
        writer.writerows(csv_fields)
