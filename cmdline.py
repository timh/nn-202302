import re
from typing import Sequence, Callable

from experiment import Experiment

RE_OP = re.compile(r"([\w_]+)\s*([=<>!~]+)\s*(.+)")
def gen_attribute_matcher(matchers: Sequence[str]) -> Callable[[Experiment], bool]:
    def fn(exp: Experiment) -> bool:
        for matcher in matchers:
            field, op, matcher_val = RE_OP.match(matcher).groups()
            exp_val = str(getattr(exp, field, None))

            matches = True
            if op == "=":
                matches = exp_val == matcher_val
            elif op == "!=":
                matches = exp_val != matcher_val
            elif op == ">":
                matches = float(exp_val) > float(matcher_val)
            elif op == "<":
                matches = float(exp_val) < float(matcher_val)
            elif op == "~":
                matches = matcher_val in exp_val
            elif op == "!~":
                matches = matcher_val not in exp_val
            else:
                raise Exception(f"unknown {op=} for {field=} {matcher_val=}")
            
            if not matches:
                return False
        return True
    return fn

