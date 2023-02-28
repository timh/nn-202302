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
import model_utils
from model_utils import TextMapper

batch_size = 1024
loss_nexamples = 32 * batch_size
textmap_by_params: Dict[str, TextMapper] = dict()
loss_fn_by_params: Dict[str, Callable[[Tensor, Tensor], Tensor]] = dict()
def get_textmap_and_lossfn(model: model_xformers_tutorial.TransformerModel, 
                           fields: Dict[str, any]) -> Tuple[TextMapper, Callable[[Tensor, Tensor], Tensor]]:
    seq_len: int = int(fields["seq_len"])
    wordmaxlen: int = int(fields["wordmaxlen"])

    key = f"{seq_len=} {wordmaxlen=}"
    if key not in textmap_by_params:
        textmap_by_params[key] = \
            TextMapper(seq_len=seq_len, filename="shakespeare.txt", 
                        wordmaxlen=wordmaxlen, 
                        device=device,
                        dtype=torch.long)
        vocab_len = textmap_by_params[key].vocab_len
        loss_fn_by_params[key] = model_xformers_tutorial.loss_fn(seq_len=seq_len, vocab_len=vocab_len)

    return textmap_by_params[key], loss_fn_by_params[key]

def get_loss(model: model_xformers_tutorial.TransformerModel, field: Dict[str, any]) -> float:
    textmap, loss_fn = get_textmap_and_lossfn(model, fields)
    examples = textmap.as_pairs()[:loss_nexamples]
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

def list_torchfiles():
    res: List[Tuple[Path, Dict[str, str]]] = list()
    for torchfile in Path("runs").iterdir():
        # print(f"torchfile {torchfile}")
        # if not torchfile.is_dir():
        #     print(f"{torchfile.is_dir()=} {torchfile.name.endswith('.torch')=} {torchfile.name.startswith(basename)=}")
        if torchfile.is_dir() or not torchfile.name.endswith(".torch") or not torchfile.name.startswith(basename):
            continue

        filename_fields = model_xformers_tutorial._parse_model_filename(str(torchfile))
        fields = {key: filename_fields.get(key, "") for key in all_fields}
        
        model_key = gen_model_key(fields)
        if model_key in csv_has_models:
            print(f"skip {torchfile}")
            continue

        res.append((torchfile, fields))
    return res

# fixed_fields = "batch_size batches_per_epoch dropout numchar nblock nhead emb_len".split(" ")
fixed_fields = "filename seq_len wordmaxlen vocab_len nhead nlayers hidden_len emb_len dropout do_layernorm".split(" ")
fixed_fields += "start_lr end_lr optim_type total_epochs batch_size".split(" ")
all_fields = fixed_fields + "loss output".split(" ")
def gen_model_key(row: Dict[str, str]) -> str:
    return " ".join([f"{key}={row.get(key)}" for key in fixed_fields])

device = "cuda"
basename = sys.argv[1]
num_pred = 100
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

csv_path_temp = csv_path.with_suffix(".tmp")        
csv_out = open(csv_path_temp, "w")
writer = csv.DictWriter(csv_out, fieldnames=all_fields)
writer.writeheader()
for row in existing_csv_rows:
    writer.writerow(row)

all_new_torch = list_torchfiles()
for i, (torchfile, fields) in enumerate(all_new_torch):
    print(f"{i + 1}/{len(all_new_torch)} {torchfile}:")
    model: model_xformers_tutorial.TransformerModel = torch.load(torchfile).to(device)

    loss = get_loss(model, fields)
    print(f"loss = \033[1;31m{loss:.5f}\033[0m")

    seq_len = int(fields["seq_len"])
    textmap, _lossfn = get_textmap_and_lossfn(model, fields)
    text = model_utils.predict(net=model, textmap=textmap, num_preds=num_pred, seq_len=seq_len, device=device)
    textout = text.replace("\n", "\n  ")
    print("text:")
    print(f"\033[1;32m  {textout}\033[0m")

    fields["output"] = text
    fields["loss"] = loss
    fields["filename"] = str(torchfile)

    writer.writerow(fields)
    csv_out.flush()
    print()

csv_out.close()
csv_path_temp.rename(csv_path)
