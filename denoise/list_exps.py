import sys
import datetime
from typing import List, Dict
import types
from collections import defaultdict, OrderedDict

sys.path.append("..")
import model_util
import cmdline
from experiment import Experiment

class Config(cmdline.QueryConfig):
    # long_format: bool
    show_header: bool
    show_net_class: bool
    show_epochs: bool
    show_label: bool
    show_tloss: bool
    show_vloss: bool
    show_both_loss: bool
    show_none_exps: bool
    show_all_diffs: bool

    fields: List[str]
    field_map: Dict[str, str]
    exclude_fields: List[str]

    def __init__(self):
        super().__init__()

        self.add_argument("-f", "--fields", type=str, nargs='+', default=list(), action='append', help="list these fields, too")
        # self.add_argument("-l", "--long", dest='long_format', default=False, action='store_true')
        # self.add_argument("-nc", "--net_class", dest='only_net_classes', type=str, nargs='+')

        self.add_argument("--labels", dest='show_label', default=False, action='store_true')
        self.add_argument("-L", dest='show_both_loss', default=False, action='store_true')
        self.add_argument("-t", dest='show_tloss', default=False, action='store_true')
        self.add_argument("-v", dest='show_vloss', default=False, action='store_true')

        self.add_argument("--show_none_exps", default=False, action='store_true', 
                          help="show exps having fields in --fields with a value of None (default False)")
        self.add_argument("-nf", "--exclude_fields", default=list(), nargs='+')
        self.add_argument("-d", "--diffs", dest='show_all_diffs', default=False, action='store_true',
                          help="if set, all fields with diffs from exp-to-exp will be included")

        self.add_argument("--no_net_class", dest='show_net_class', default=True, action='store_false')
        self.add_argument("--no_epochs", dest='show_epochs', default=True, action='store_false')
        self.add_argument("-H", "--no_header", dest='show_header', default=True, action='store_false')

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
        
        if self.show_both_loss:
            self.show_tloss = True
            self.show_vloss = True

def get_max_value_len(cfg: Config, exps: List[Experiment], field_map: Dict[str, str]) -> Dict[str, int]:
    res: Dict[str, int] = defaultdict(lambda: 0)
    if cfg.show_header:
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

if __name__ == "__main__":
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

    between = "  "
    max_value_len = get_max_value_len(cfg, exps, field_map)

    if cfg.show_header:
        line_parts: List[str] = list()
        for field, short in field_map.items():
            line_parts.append(short.ljust(max_value_len[field]))
        print(between.join(line_parts))

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
        print(between.join(line_parts))
