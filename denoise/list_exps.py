import sys
import datetime
from typing import List, Tuple, Dict
import types
from collections import defaultdict, OrderedDict
import math

sys.path.append("..")
import model_util
import cmdline
from experiment import Experiment, ExpRun

MARGIN = " " * 2
RUN_MARGIN = " " * 4

class Config(cmdline.QueryConfig):
    show_header: bool
    show_net_class: bool
    show_epochs: bool
    show_label: bool
    show_tloss: bool
    show_vloss: bool
    show_both_loss: bool
    show_runs: bool

    show_none_exps: bool
    show_all_diffs: bool

    fields: List[str]
    field_map: Dict[str, str]
    exclude_fields: List[str]

    do_denoise: bool
    do_vae: bool

    def __init__(self):
        super().__init__()

        self.add_argument("-f", "--fields", type=str, nargs='+', default=list(), action='append', help="list these fields, too")
        # self.add_argument("-l", "--long", dest='long_format', default=False, action='store_true')
        # self.add_argument("-nc", "--net_class", dest='only_net_classes', type=str, nargs='+')

        self.add_argument("--labels", dest='show_label', default=False, action='store_true')
        self.add_argument("-L", "--both_loss", dest='show_both_loss', default=False, action='store_true')
        self.add_argument("-t", "--tloss", dest='show_tloss', default=False, action='store_true')
        self.add_argument("-v", "--vloss", dest='show_vloss', default=False, action='store_true')
        self.add_argument("-r", "--runs", dest='show_runs', default=False, action='store_true')

        self.add_argument("--show_none_exps", default=False, action='store_true', 
                          help="show exps having fields in --fields with a value of None (default False)")
        self.add_argument("-nf", "--exclude_fields", default=list(), nargs='+')
        self.add_argument("-d", "--diffs", dest='show_all_diffs', default=False, action='store_true',
                          help="if set, all fields with diffs from exp-to-exp will be included")

        self.add_argument("--no_net_class", dest='show_net_class', default=True, action='store_false')
        self.add_argument("--no_epochs", dest='show_epochs', default=True, action='store_false')
        self.add_argument("-H", "--no_header", dest='show_header', default=True, action='store_false')

        self.add_argument("-dn", "--denoise", dest='do_denoise', default=False, action='store_true',
                          help="reasonable defaults for denoise networks")
        self.add_argument("-vae", "--vae", dest='do_vae', default=False, action='store_true',
                          help="reasonable defaults VAE networks")

    def parse_args(self) -> 'Config':
        res = super().parse_args()

        # flatten the fields, which get parsed into a list of lists of str.
        self.fields = [field for fields in self.fields for field in fields]
        self.field_map = OrderedDict()
        for field_str in self.fields:
            field, short = field_str, field_str
            if field.startswith("net."):
                field = "net_" + field[4:]
            if '=' in field:
                field, short = field.split('=', maxsplit=1)
            elif ':' in field_str:
                field, short = field.split(':', maxsplit=1)
            self.field_map[field] = short
        
        if self.do_vae:
            # -f net.image_size=size net.layers_str=layers -nc VarEncDec
            if 'net_layers_str' not in self.field_map:
                self.field_map['net_layers_str'] = 'layers'
            if 'net_norm_num_groups' not in self.field_map:
                self.field_map['net_norm_num_groups'] = 'normg'
            if not self.net_classes:
                self.net_classes.append("VarEncDec")
            pass

        if self.do_denoise:
            # "-nc Unet -a 'loss_type = l1' -s tloss -f nparams"
            # "vae_shortcode=vae net_resnet_block_groups=blk net_self_condition=selfcond"
            # "-d -f elapsed_str --run"
            if 'image_size' not in self.field_map:
                self.field_map['image_size'] = 'size'
            if 'vae_shortcode' not in self.field_map:
                self.field_map['vae_shortcode'] = 'vae'
            if 'net_resnet_block_groups' not in self.field_map:
                self.field_map['net_resnet_block_groups'] = 'rblks'
            if 'net_self_condition' not in self.field_map:
                self.field_map['net_self_condition'] = 'selfcond'
            if not self.net_classes:
                self.net_classes.append("Unet")
                self.net_classes.append("DenoiseModel")

            self.attribute_matchers.append("is_denoiser = True")

        if self.show_both_loss:
            self.show_tloss = True
            self.show_vloss = True

def get_max_value_len(exps: List[Experiment], field_map: Dict[str, str], include_field_names: bool) -> Dict[str, int]:
    res: Dict[str, int] = defaultdict(lambda: 0)
    if include_field_names:
        for field, short in field_map.items():
            res[field] = len(short)

    for exp in exps:
        for field in field_map.keys():
            val = getattr(exp, field, None)
            if val is None:
                continue

            val = model_util.str_value(val, field)
            res[field] = max(res[field], len(val))

    return res

def get_run_str(exp: Experiment, run: ExpRun) -> str:
    tloss = exp.train_loss_hist[run.checkpoint_nepochs]
    run_parts = [f"{run.checkpoint_nepochs:4} epochs",
                 f"tloss {tloss:2.5f}",
                 run.checkpoint_at_relative()]
    return ", ".join(run_parts)

def get_runs_perline_maxlen(exps: List[Experiment], max_value_len: Dict[str, int]) -> Tuple[int, int]:
    max_line_len = sum(max_value_len.values())
    max_line_len += (len(max_value_len) - 1) * len(MARGIN)

    max_run_len = 0
    for exp in exps:
        for run in exp.runs:
            run_str = get_run_str(exp, run)
            max_run_len = max(max_run_len, len(run_str))
    
    # max_run_len += len(RUN_MARGIN)  # add margin between runs, including before the first one

    return max_line_len // max_run_len, max_run_len

def filter_exps_add_fields(cfg: Config, exps: List[Experiment], field_map: Dict[str, str]) -> List[Experiment]:
    fields = cfg.field_map.keys()
    exps_in = exps.copy()
    exps.clear()

    last_val_dict: Dict[str, any] = dict()
    for exp in exps_in:
        any_fields_none = any([getattr(exp, field, None) is None for field in fields])
        if any_fields_none and not cfg.show_none_exps:
            continue

        exps.append(exp)

        if not cfg.show_all_diffs:
            continue

        id_fields = exp.id_fields()
        id_fields.extend([f"net_{afield}" for afield in exp.net_args.keys()])
        for field in id_fields:
            val = getattr(exp, field, None)
            if field.endswith("_args") or field == 'label':
                continue
            if field in cfg.exclude_fields or field in field_map:
                continue

            # if type(val) not in [int, float, bool, str]:
            #     continue
            last_val = last_val_dict.get(field, None)
            if last_val is not None and val != last_val:
                field_map[field] = field
            last_val_dict[field] = val

    return exps

def main():
    cfg = Config()
    cfg.parse_args()

    now = datetime.datetime.now()

    field_map: Dict[str, str] = OrderedDict()
    field_map['shortcode'] = 'code'
    if cfg.show_net_class:
        field_map['net_class'] = 'net_class'
    field_map['saved_at_relative'] = 'saved (rel)'
    if cfg.show_epochs:
        field_map['nepochs'] = 'epoch'
    if cfg.show_tloss:
        field_map['last_train_loss'] = 'tloss'
    if cfg.show_vloss:
        field_map['last_val_loss'] = 'vloss'
    if cfg.show_label:
        field_map['label'] = 'label'
    field_map.update(cfg.field_map)

    exps = cfg.list_experiments()
    if cfg.show_all_diffs or (not cfg.show_none_exps):
        exps = filter_exps_add_fields(cfg, exps, field_map)
                    
    if cfg.sort_key and 'loss' in cfg.sort_key:
        # show the lowest loss at the end.
        exps = list(reversed(exps))

    max_value_len = get_max_value_len(exps, field_map, include_field_names=cfg.show_header)
    if cfg.show_header:
        line_parts: List[str] = list()
        for field, short in field_map.items():
            line_parts.append(short.ljust(max_value_len[field]))
        print(MARGIN.join(line_parts))
    
    if cfg.show_runs:
        runs_per_line, max_run_len = get_runs_perline_maxlen(exps, max_value_len)
    
    last_value: Dict[str, any] = dict()
    for exp in exps:
        line_parts: List[str] = list()
        for field in field_map.keys():
            val = getattr(exp, field, None)
            if type(val) in [types.MethodType, types.FunctionType]:
                val = val()

            valstr = model_util.str_value(val, field)
            valstr = valstr.ljust(max_value_len[field])

            if field in last_value and last_value[field] != val:
                if type(val) in [int, float]:
                    # green if new value is above last, red if less.
                    # colors are inverted for 'loss' fields.
                    if last_value[field] is None:
                        better = True
                    else:
                        better = bool(val > last_value[field])
                    if 'loss' in field:
                        better = not better

                    if better:
                        valstr = f"\033[1;32m{valstr}\033[0m"
                    else:
                        valstr = f"\033[1;31m{valstr}\033[0m"
                else:
                    valstr = f"\033[1m{valstr}\033[0m"
            line_parts.append(valstr)

            last_value[field] = val
        
        print(MARGIN.join(line_parts))

        if cfg.show_runs:
            nlines = math.ceil(len(exp.runs) / runs_per_line)
            run_lines = [""] * nlines
            for run_idx, run in enumerate(exp.runs):
                line_idx = run_idx % nlines

                run_str = get_run_str(exp, run)
                run_str = run_str.ljust(max_run_len)
                if run_lines[line_idx]:
                    run_lines[line_idx] += RUN_MARGIN
                run_lines[line_idx] += run_str
            
            run_lines = [MARGIN + line for line in run_lines]
            print("\n".join(run_lines))
            print()

if __name__ == "__main__":
    main()