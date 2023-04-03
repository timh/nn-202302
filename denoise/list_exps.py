import sys
import datetime
from typing import List, Dict
from collections import defaultdict, OrderedDict

sys.path.append("..")
import model_util
import cmdline
from experiment import Experiment

class Config(cmdline.QueryConfig):
    # long_format: bool
    show_header: bool
    show_labels: bool
    show_tloss: bool
    show_vloss: bool
    show_both_loss: bool

    only_net_classes: List[str]
    fields: List[str]
    field_map: Dict[str, str]

    def __init__(self):
        super().__init__()

        self.add_argument("-f", "--fields", type=str, nargs='+', default=list(), help="list these fields, too")
        # self.add_argument("-l", "--long", dest='long_format', default=False, action='store_true')
        self.add_argument("-nc", "--net_class", dest='only_net_classes', type=str, nargs='+')
        self.add_argument("-H", "--no_header", dest='show_header', default=True, action='store_false')
        self.add_argument("--labels", dest='show_labels', default=False, action='store_true')
        self.add_argument("-L", dest='show_both_loss', default=False, action='store_true')
        self.add_argument("-t", dest='show_tloss', default=False, action='store_true')
        self.add_argument("-v", dest='show_vloss', default=False, action='store_true')
    
    def parse_args(self) -> 'Config':
        res = super().parse_args()

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

            val = model_util.str_value(val)
            res[field] = max(res[field], len(val))

    return res

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    now = datetime.datetime.now()

    exps = cfg.list_experiments()
    if cfg.only_net_classes:
        exps = [exp for exp in exps if exp.net_class in cfg.only_net_classes]
    if cfg.sort_key and 'loss' in cfg.sort_key:
        # show the lowest loss at the end.
        exps = list(reversed(exps))

    field_map: Dict[str, str] = OrderedDict()
    field_map['shortcode'] = 'code'
    field_map['net_class'] = 'net_class'
    field_map['saved_at_relative'] = 'saved (rel)'
    if cfg.show_labels:
        field_map['label'] = 'label'
    if cfg.show_tloss:
        field_map['last_train_loss'] = 'tloss'
    if cfg.show_vloss:
        field_map['last_val_loss'] = 'vloss'
    field_map.update(cfg.field_map)

    between = "  "
    max_value_len = get_max_value_len(cfg, exps, field_map)

    if cfg.show_header:
        line_parts: List[str] = list()
        for field, short in field_map.items():
            line_parts.append(short.ljust(max_value_len[field]))
        print(between.join(line_parts))

    last_value: Dict[str, str] = dict()
    for exp in exps:
        line_parts: List[str] = list()
        for field in field_map.keys():
            val = getattr(exp, field, None)
            val = model_util.str_value(val)
            val = val.ljust(max_value_len[field])

            showval = val
            if field in last_value and last_value[field] != val:
                showval = f"\033[1m{val}\033[0m"
            line_parts.append(showval)
            last_value[field] = val
        print(between.join(line_parts))
