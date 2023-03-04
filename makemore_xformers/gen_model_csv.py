# %%
import sys
from typing import List, Tuple, Set, Dict, Callable
from pathlib import Path
import csv
import re

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, "..")
import model_utils
import model
import tokens
from experiment import Experiment
import text_experiment

device = "cuda"
batch_size = 256
loss_nexamples = 32 * batch_size

input_fields = ("seqlen wordlen vocablen nhead nlayers hidlen emblen dropout "
                "startlr endlr optim_type epochs batch minicnt "
                "pytorch_version compile flash nsamples").split(" ")

all_fields = ["filename"] + input_fields + "elapsed samples_per_sec loss output".split(" ")

def gen_model_key(row: Dict[str, str]) -> str:
    return " ".join([f"{key}={row.get(key)}" for key in input_fields])

def new_experiments():
    res: List[Tuple[Path, Dict[str, str]]] = list()
    for torchfile in Path("runs").iterdir():
        if torchfile.is_dir() or not torchfile.name.endswith(".ckpt") or not torchfile.name.startswith(basename):
            continue

        state_dict = torch.load(torchfile)
        try:
            exp = text_experiment.load_experiment(state_dict, device=device)
        except Exception as e:
            print(f"can't load model in {torchfile}. pytorch version mismatch?")
            print(e)
            continue

        fields = {field: getattr(exp, field, "") for field in input_fields}
        model_key = gen_model_key(fields)
        if model_key in csv_has_models:
            print("  skip")
            continue

        yield (torchfile, exp)


if __name__ == "__main__":
    basename = sys.argv[1]
    num_pred = 250
    num_lines = 10

    csv_has_models: Set[str] = set()
    existing_csv_rows: List[Dict[str, str]] = list()
    csv_path = Path(f"{basename}.csv")
    if csv_path.exists():
        with open(csv_path, "r") as csv_in:
            reader = csv.DictReader(csv_in)
            for row in reader:
                csv_has_models.add(gen_model_key(row))
                existing_csv_rows.append(row)

    csv_path_temp = csv_path.with_suffix(".tmp")        
    csv_out = open(csv_path_temp, "w")
    writer = csv.DictWriter(csv_out, fieldnames=all_fields)
    writer.writeheader()
    for row in existing_csv_rows:
        writer.writerow(row)

    for i, (torchfile, exp) in enumerate(new_experiments()):
        print(f"#{i} {torchfile}: ")

        loss = exp.val_loss_hist[-1].item()
        print(f"loss = \033[1;31m{loss:.5f}\033[0m")

        start_text = "\n"
        try:
            text = model_utils.predict(net=exp.net, seq_len=exp.seqlen, num_preds=num_pred, 
                                    tokenizer=exp.tokenizer, dictionary=exp.dictionary, 
                                    start_text=start_text, device=device)
        except Exception as e:
            print(f"can't predict model: pytorch version mismatch?")
            print(e)
            continue

        if num_lines:
            text = "\n".join(text.split("\n")[:num_lines])
        textout = text.replace("\n", "\n  ")
        print("text:")
        print(f"\033[1;32m  {textout}\033[0m")

        fields = {field: getattr(exp, field, "") for field in all_fields}
        fields["loss"] = loss
        fields["filename"] = str(torchfile)
        fields["output"] = text
        fields["compile"] = exp.compile
        elapsed = float(re.compile(r".*elapsed ([\d\.]+).*").match(str(torchfile)).group(1))
        fields["elapsed"] = elapsed

        samples_per_sec = exp.nsamples / elapsed
        fields["samples_per_sec"] = samples_per_sec

        writer.writerow(fields)
        csv_out.flush()
        print()

    csv_out.close()
    csv_path_temp.rename(csv_path)
