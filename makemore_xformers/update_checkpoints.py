from pathlib import Path
import re
import datetime
import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import text_experiment
from text_experiment import TextExperiment
import tokens

device = "cuda"

def compute_loss(exp: TextExperiment, val_text_filename: str, num_samples: int = 0) -> float:
    treader = tokens.WordTextReader(seq_len=exp.seqlen, wordlen=exp.wordlen, filename=val_text_filename, include_special=True, device=device)
    # ntrain = int(0.8 * len(treader))
    # _train_data, val_data = treader.train_val_split(ntrain)

    # make it a batch that was the same size as it was during training.
    if num_samples == 0:
        num_samples = exp.batch * exp.minicnt
    val_data = treader[:num_samples]
    dataloader = DataLoader(val_data, batch_size=exp.batch)

    total_ninputs = 0
    total_loss = 0.0
    for inputs, truth in dataloader:
        out = exp.net(inputs)
        loss = exp.loss_fn(out, truth)
        total_loss += loss.item()
        total_ninputs += len(inputs)
    return total_loss / total_ninputs


RE_ELAPSED = re.compile(r".*elapsed ([\d\.]+).*")
def process_path(torchfile: Path, val_text_filename: str, num_samples: int = 0):
    state_dict = torch.load(torchfile)
    try:
        exp = text_experiment.load_experiment(state_dict, device=device)
    except Exception as e:
        # print(f"{str(e)=}")
        # this likely happens when loading an experiment done on a different
        # version of pytorch, especially with a different attention mechanism.
        if "Error(s) in loading state_dict" in str(e):
            print(f"* skip {torchfile}: state_dict issue; python version mismatch?")
            return
        raise e

    # skip if major numbers of pytorch version don't match
    exp_major = str(exp.pytorch_version).split(".")[0]
    running_major = str(torch.__version__).split(".")[0]
    if exp_major != running_major:
        print(f"- skip. mismatched pytorch version: {exp.pytorch_version=} {torch.__version__=}.")
        return

    resave = False

    if exp.started_at is None:
        print(f"* updating started_at, ended_at, elapsed")
        elapsed = exp.elapsed
        if not elapsed:
            elapsed_match = RE_ELAPSED.match(str(torchfile))
            elapsed_str = elapsed_match.group(1)
            elapsed = float(elapsed_str)
        
        exp.ended_at = datetime.datetime.now()
        exp.started_at = exp.ended_at - datetime.timedelta(seconds=elapsed)
        exp.elapsed = elapsed

        resave = True
    
    if isinstance(exp.elapsed, datetime.timedelta):
        exp.elapsed = exp.elapsed.total_seconds()
        resave = True
        
    if not exp.last_val_loss:
        print(f"* checkpoint missing loss: computing..")
        exp.last_val_loss = compute_loss(exp, val_text_filename, num_samples)

        if not isinstance(exp.val_loss_hist, Tensor):
            exp.val_loss_hist = Tensor()
        resave = True
    
    if resave:
        print(f"\033[1m* resaving\033[0m")
        checkpoint = exp.state_dict()
        torch.save(checkpoint, torchfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--val_text_filename", required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=0)

    cfg = parser.parse_args()

    for path in Path("runs").iterdir():
        if not path.name.endswith(".ckpt"):
            continue

        print(f"{path}:")
        process_path(path, cfg.val_text_filename, cfg.num_samples)
        print()


