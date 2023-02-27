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
import model_xformers_tutorial
from model_utils import TextMapper

batch_size = 1024
textmap_by_params: Dict[str, TextMapper] = dict()
loss_fn_by_params: Dict[str, Callable[[Tensor, Tensor], Tensor]] = dict()
def get_examples(model: model_xformers_tutorial.TransformerModel, 
                 num_examples: int,
                 fields: Dict[str, any]) -> List[Tuple[Tensor, Tensor]]:
    seq_len: int = fields["seq_len"]
    wordmaxlen: int = fields["wordmaxlen"]

    key = f"{seq_len=} {wordmaxlen=}"
    if key not in textmap_by_params:
        textmap_by_params[key] = \
            TextMapper(seq_len=seq_len, filename="shakespeare.txt", 
                        words_or_chars="words", wordmaxlen=wordmaxlen, 
                        device=device,
                        dtype=torch.long)
        vocab_len = textmap_by_params[key].vocab_len
        loss_fn_by_params[key] = model_xformers_tutorial.loss_fn(seq_len=seq_len, vocab_len=vocab_len)

    textmap = textmap_by_params[key]
    return textmap.as_pairs()[:num_examples], loss_fn_by_params[key]

def get_loss(model: model_xformers_tutorial.TransformerModel, num_examples: int, field: Dict[str, any]) -> float:
    examples, loss_fn = get_examples(model, num_examples, field)
    dataloader = DataLoader(examples, batch_size)

    model.eval()
    total_loss = 0.0
    count = 0
    for input_tokens, truth in dataloader:
        outputs = model(input_tokens)
        loss = loss_fn(outputs, truth)
        total_loss += loss.item()
        count += 1
    return total_loss / count

# fixed_fields = "batch_size batches_per_epoch dropout numchar nblock nhead emb_len".split(" ")
fixed_fields = "seq_len wordmaxlen vocab_len nhead nlayers hidden_len emb_len".split(" ")
fixed_fields += "batch_size batches_per_epoch total_epochs".split(" ")
all_fields = fixed_fields + "loss output".split(" ")
def gen_model_key(row: Dict[str, str]) -> str:
    return " ".join([f"{key}={row.get(key)}" for key in fixed_fields])

device = "cuda"
basename = sys.argv[1]
num_pred = 100
num_examples = 2048
# batch_size 1024, batches_per_epoch   4, dropout 0.2, numchar  32, nblock   4, nhead   2, emb_len  96.torch:

csv_has_models: Set[str] = set()
existing_csv_rows: List[Dict[str, str]] = list()
csv_path = Path(f"{basename}.csv")
if csv_path.exists():
    with open(csv_path, "r") as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            csv_has_models.add(gen_model_key(row))
            existing_csv_rows.append(row)
        
csv_out = open(csv_path, "w")
writer = csv.DictWriter(csv_out, fieldnames=all_fields)
writer.writeheader()
for row in existing_csv_rows:
    writer.writerow(row)

RE_MULTISPACE = re.compile(r"\s+")

for torchfile in Path("runs").iterdir():
    # print(f"torchfile {torchfile}")
    # if not torchfile.is_dir():
    #     print(f"{torchfile.is_dir()=} {torchfile.name.endswith('.torch')=} {torchfile.name.startswith(basename)=}")
    if torchfile.is_dir() or not torchfile.name.endswith(".torch") or not torchfile.name.startswith(basename):
        continue

    fields_str = torchfile.name.replace(".torch", "")
    while True:
        if "-" not in fields_str:
            break
        first_dash = fields_str.index("-")
        fields_str = fields_str[first_dash + 1:]

    fields_list = fields_str.split(", ")
    fields = {key: "" for key in all_fields}
    for field_str in fields_list:
        key, value = RE_MULTISPACE.split(field_str)
        value = float(value) if key == "dropout" else int(value)
        fields[key] = value
    
    model_key = gen_model_key(fields)
    if model_key in csv_has_models:
        print(f"skip {torchfile} cuz it's already in the CSV")
        continue

    print(f"{torchfile}:")
    model: model_xformers_tutorial.TransformerModel = torch.load(torchfile).to(device)
    loss = get_loss(model, num_examples, fields)
    fields["loss"] = loss

    print(f"loss = \033[1;33m{loss:.5f}\033[0m")
    text = model.predict(num_pred, device=device)
    fields["output"] = text

    textout = text.replace("\n", "\n  ")
    print(f"\033[1;32m  {textout}\033[0m")
    print()

    writer.writerow(fields)
    csv_out.flush()

csv_out.close()
