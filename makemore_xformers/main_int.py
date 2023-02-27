# %%
import sys
import importlib
from typing import List, Callable
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator

sys.path.insert(0, "..")
import notebook
import trainer
import experiment
from experiment import Experiment
import model_utils
import model_xformers
import model_xformers_tutorial


for m in notebook, trainer, model_xformers, experiment:
    importlib.reload(m)

# %%
device = "cuda"

# accel = Accelerator()
accel = None

def get_optimizer_fn(exp: Experiment, lr: float) -> torch.optim.Optimizer:
    optim = torch.optim.AdamW(exp.net.parameters(), lr)
    if accel is not None:
        optim = accel.prepare(optim)
    return optim
        
class MakemoreLogger(trainer.TensorboardLogger):
    def __init__(self, num_pred: int, basename: str):
        # super().__init__("mm-ssnat")
        super().__init__(basename)
        self.num_pred = num_pred
    
    def on_epoch_end_infrequent(self, exp: Experiment, exp_epoch: int, lr_epoch: int):
        super().on_epoch_end_infrequent(exp, exp_epoch, lr_epoch)

        # res = exp.net.predict(self.num_pred, device=device)
        res = model_utils.predict(exp.net, textmap=exp.textmap, num_preds=self.num_pred, device=device)
        res = res.replace("\n", "\n  ")
        print(f"predict({self.num_pred}): {exp.label} @ {exp.cur_lr:.1E}")
        print(f"\033[1;32m  {res}\033[0m")
        print()

def experiments(filename = "shakespeare.txt"):
    print("make experiments")
    for seq_len in seq_len_values:
        for wordmaxlen in wordmaxlen_values:
            print(f"make_data({seq_len=}, {wordmaxlen=})")
            textmap = model_xformers.TextMapper(seq_len, filename=filename, device=device, dtype=torch.long, words_or_chars="words", wordmaxlen=wordmaxlen)
            all_examples = textmap.as_pairs()
            num_train = int(len(all_examples) * 0.8)

            # NOTE: karpathy uses a single mini-batch per epoch, of size (seq_len)
            train_data = all_examples[:num_train]
            train_sampler = RandomSampler(train_data, num_samples=batches_per_epoch * batch_size)
            train_dl = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

            val_data = all_examples[num_train:]
            val_sampler = RandomSampler(val_data, num_samples=batches_per_epoch * batch_size)
            val_dl = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)
            print(f"  {len(train_data)=}, {len(val_data)=}")
            print(f"  {len(train_dl)=}, {len(val_dl)=}")

            if accel is not None:
                train_dl, val_dl = accel.prepare(train_dl, val_dl)

            first_inputs, first_truth = next(iter(val_dl))
            print(f"{len(first_inputs)=}")
            print(f"{len(first_truth)=}")

            for nhead in nhead_values:
                for nlayers in nlayers_values:
                    for hidden_len in hidden_len_values:
                        for emb_len in emb_len_values:
                            fields = dict(
                                seq_len=seq_len,
                                wordmaxlen=wordmaxlen,
                                vocab_len=textmap.vocab_len,
                                nhead=nhead,
                                nlayers=nlayers,
                                hidden_len=hidden_len,
                                emb_len=emb_len,
                                dropout=format(dropout, ".1f"),
                                batch_size=batch_size,
                                batches_per_epoch=batches_per_epoch,
                                total_epochs=total_epochs,
                            )
                            label = ", ".join([f"{key} {val}" for key, val in fields.items()])
                            ptr_path = Path("runs", basename + "-" + label)
                            if ptr_path.exists():
                                print(f"\033[1;32m{ptr_path} exists, skipping\033[0m")
                                continue

                            model = model_xformers_tutorial.TransformerModel(vocab_len=textmap.vocab_len, emb_len=emb_len, nhead=nhead, 
                                                                            nlayers=nlayers, hidden_len=hidden_len, 
                                                                            dropout=dropout, device=device)

                            if accel is not None:
                                model = accel.prepare(model)

                            loss_fn = model_xformers_tutorial.loss_fn(seq_len=seq_len, vocab_len=textmap.vocab_len)
                            exp = Experiment(label, model, loss_fn, train_dl, val_dl)
                            exp.seq_len = seq_len
                            exp.textmap = textmap
                            yield exp

                            # fields['last_train_loss'] = exp.train_loss_hist[-1]
                            # fields['last_val_loss'] = exp.val_loss_hist[-1]
                            # fields['elapsed_sec'] = (exp.ended_at - exp.started_at).total_seconds()

                            # model_utils.update_csv

                            torch_path = str(ptr_path) + ".torch"
                            with open(torch_path, "wb") as torch_file:
                                torch.save(model, torch_file)
                                print(f"saved {torch_path}")

                            with open(ptr_path, "w") as file:
                                log_filename = str(Path(logger.dirname, label))
                                print(f"write {ptr_path}")
                                print(log_filename, file=file)


learning_rates = [
    (1e-4, 1000)
    # (1e-4, 1000),  4tut2
    # (5e-5, 5000),
    # (1e-5, 5000),

    # # (3e-4,  5000), # karpathy
    # (3e-4,   500),
    # (1e-4,  1000), # could be more
    # (7e-5,  1000),
    # (5e-5,  4000),
    # # (3e-4,  1000),
    # # (1e-4,  1000),
    # # (1e-5,  1000),
    # # (5e-6,  1000),
    # # (1e-6,  1000),
]
total_epochs = sum([epochs for _lr, epochs in learning_rates])

do_layernorm = True
do_residual = True

# nc 64, nb 2, nh 4, el 24
nhead_values = [2, 4]
seq_len_values = [64, 128]
nlayers_values = [2, 4]
hidden_len_values = [64, 128]
emb_len_values = [32, 64]
dropout = 0.2
batch_size = 1024
batches_per_epoch = 4
wordmaxlen_values = [2, 4, 0]
dtype = "fp32"
basename = "mm-ss4tut2" # + dtype
if accel is not None:
    basename = basename + "-accel"

    

# %%
print("train")

# for debug only TODO
# learning_rates = [(lrpair[0], max(1, lrpair[1]//100)) for lrpair in learning_rates]

filename = "shakespeare.txt"
if (len(sys.argv) > 1 and sys.argv[1] == "-d"):
    filename = "shakespeare-1000.txt"

tcfg = trainer.TrainerConfig(learning_rates, get_optimizer_fn, experiments(filename), accel=accel)
logger = MakemoreLogger(num_pred=50, basename=basename)
tr = trainer.Trainer(logger=logger)
tr.train(tcfg)

# %%
