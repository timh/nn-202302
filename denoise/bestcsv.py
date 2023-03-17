import csv
import sys
from typing import List, Dict, Tuple
sys.path.append("..")
import model_util

best_row: Dict[str, Dict[str, any]] = dict()
best_row_tloss: Dict[str, float] = dict()

skip_fields = set("started_at ended_at saved_at path lastepoch_train_loss lastepoch_val_loss elapsed cur_lr nbatches nepochs nsamples runpath exp_idx".split(" "))
with open("runs/experiments.csv") as file:
    reader = csv.reader(file)
    fields = next(reader)
    for row in reader:
        row_dict = {field: val for field, val in zip(fields, row)}
        filtered = {field: val for field, val in zip(fields, row)
                    if field not in skip_fields}
        key = "|".join([f"{f}={v}" for f, v in filtered.items()])

        tloss = float(row_dict['lastepoch_train_loss'])
        if key not in best_row or tloss < best_row_tloss[key]:
            for field, val in row_dict.items():
                if val.isnumeric():
                    val = int(val)
                elif all([c.isnumeric() or c == '.' for c in val]):
                    val = float(val)
                # elif val == 'True':
                #     val = bool(val)
                row_dict[field] = val
            row_dict.pop('net_conv_cfg_metadata')
            best_row[key] = row_dict
            best_row_tloss[key] = tloss

out = csv.DictWriter(sys.stdout, fields)
out.writeheader()

best_rows_sorted = sorted(best_row.values(), key=lambda row: -row['lastepoch_train_loss'])
for i, row in enumerate(best_rows_sorted):
    out.writerow(row)
    # print(f"{i+1}/{len(best_row)}:", file=sys.stderr)
    # model_util.print_dict(row)
    # print()
