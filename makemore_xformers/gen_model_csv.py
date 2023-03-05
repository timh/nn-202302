# %%
import sys
from typing import List, Tuple, Set, Dict, Callable
from pathlib import Path
import csv
import re
import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, "..")
import model_utils
import model
import tokens
from text_experiment import TextExperiment
import text_experiment

device = "cuda"
batch_size = 256
loss_nexamples = 32 * batch_size

input_fields = ("seqlen wordlen vocablen nhead nlayers hidlen emblen dropout "
                "startlr endlr optim_type epochs batch minicnt "
                "pytorch_version compile flash nsamples").split(" ")

all_fields = ["filename"] + input_fields + "nparams elapsed samples_per_sec train_loss val_loss output".split(" ")

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
            # print(f"{str(e)=}")
            # this likely happens when loading an experiment done on a different
            # version of pytorch, especially with a different attention mechanism.
            if "Error(s) in loading state_dict" in str(e):
                print(f"- skip {torchfile}: state_dict issue; python version mismatch?")
                continue
            raise e

        # skip if major numbers of pytorch version don't match
        exp_major = str(exp.pytorch_version).split(".")[0]
        running_major = str(torch.__version__).split(".")[0]
        if exp_major != running_major:
            print(f"- skip: {exp.pytorch_version=} != {torch.__version__=}.")
            continue

        fields = {field: getattr(exp, field, "") for field in input_fields}
        model_key = gen_model_key(fields)
        if model_key in csv_has_models:
            print("- skip: exists.")
            continue

        yield (torchfile, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--basename", required=True)
    parser.add_argument("-n", "--num_pred", type=int, default=250)
    parser.add_argument("-l", "--num_lines", type=int, default=10)
    parser.add_argument("-t", "--start_text", default="\n")
    cfg = parser.parse_args()

    basename = cfg.basename
    num_pred = cfg.num_pred
    num_lines = cfg.num_lines
    start_text = cfg.start_text.replace("\\n", "\n")

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
        print()
        print(f"\033[1m#{i} {torchfile}:\033[0m")

        # loss = compute_loss(exp)
        val_loss = exp.last_val_loss
        train_loss = exp.train_loss_hist[-1].item()
        print(f"- train loss = \033[1;31m{train_loss:.5f}\033[0m")
        print(f"-   val loss = \033[1;31m{val_loss:.5f}\033[0m")

        text = model_utils.predict(net=exp.net, seq_len=exp.seqlen, num_preds=num_pred, 
                                    tokenizer=exp.tokenizer, dictionary=exp.dictionary, 
                                    start_text=start_text, device=device)

        if num_lines:
            text = "\n".join(text.split("\n")[:num_lines])
        textout = text.replace("\n", "\n  ")
        print("text:")
        print(f"\033[1;32m  {textout}\033[0m")

        fields = {field: getattr(exp, field, "") for field in all_fields}
        fields["train_loss"] = train_loss
        fields["val_loss"] = val_loss
        fields["filename"] = str(torchfile)
        fields["output"] = text
        fields["compile"] = exp.compile

        samples_per_sec = exp.nsamples / exp.elapsed
        fields["samples_per_sec"] = samples_per_sec

        fields["nparams"] = sum(p.numel() for p in exp.net.parameters())

        writer.writerow(fields)
        csv_out.flush()
        print()

    csv_out.close()
    csv_path_temp.rename(csv_path)
