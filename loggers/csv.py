import datetime
import csv
from typing import List, Set, Dict, Tuple
from pathlib import Path

from trainer import TrainerLogger
from experiment import Experiment

class CsvLogger(TrainerLogger):
    def __init__(self, path: Path, runpath: Path):
        self.path = path
        self.runpath = runpath

    def _existing_rows(self) -> Tuple[List[str], List[Dict[str, any]]]:
        if not self.path.exists():
            return list(), dict()

        rows: List[Dict[str, any]] = list()
        with open(self.path, "r") as file:
            reader = csv.reader(file)
            field_names = next(reader)
            for row_values in reader:
                row: Dict[str, any] = dict()
                for i, (field, value) in enumerate(zip(field_names, row_values)):
                    row[field] = value
                rows.append(row)
        return field_names, rows

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        exp_dict = exp.metadata_dict()
        exp_dict = {field: val for field, val in exp_dict.items()
                    if not field.endswith("_args")}
        exp_dict['runpath'] = str(self.runpath)

        field_names, existing_rows = self._existing_rows()
        if not field_names:
            field_names = list(exp_dict.keys())
        else:
            # add all fields in experiment that weren't in the csv we loaded
            field_names = field_names.copy()
            field_names_set = set(field_names)

            for exp_field in exp_dict.keys():
                if exp_field in field_names_set:
                    continue
                field_names.append(exp_field)
        
        # existing_rows = [row for row in existing_rows 
        #                  if row['label'] != exp.label or row['runpath'] != self.runpath]

        # keep these (long) fields at the end
        field_names.remove('label')
        field_names.remove('runpath')
        field_names = field_names + ['label', 'runpath']
        
        nrows = 1
        temp_path = Path(str(self.path) + ".tmp")
        with open(temp_path, "w") as file:
            writer = csv.DictWriter(file, field_names)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
                nrows += 1
            writer.writerow(exp_dict)

        temp_path.rename(self.path)
        print(f"  wrote {nrows} rows to {self.path}")

