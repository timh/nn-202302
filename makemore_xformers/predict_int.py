# %%
import sys
import importlib
import argparse

import torch

import model_utils
import model
import text_experiment

# for interactive sessions
for m in [model_utils, model]:
    importlib.reload(m)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "-f", "--model", dest="model_filename", required=True)
parser.add_argument("-n", "--pred_len", dest="pred_len", type=int, default=200)
parser.add_argument("-s", "--start_text", default="\n")

cfg = parser.parse_args()

state_dict = torch.load(cfg.model_filename)
exp = text_experiment.load_experiment(state_dict, device="cuda")

# %%
pred_len = cfg.pred_len
start_text = cfg.start_text.replace("\\n", "\n")

while True:
    s = model_utils.predict(net=exp.net, seq_len=exp.seqlen, num_preds=pred_len, tokenizer=exp.tokenizer, dictionary=exp.dictionary, start_text=start_text, device="cuda")
    print("\033[1;32m" + s + "\033[0m")
    # s2 = model_utils.predict2(net=exp.net, seq_len=exp.seqlen, num_preds=pred_len, tokenizer=exp.tokenizer, dictionary=exp.dictionary, start_text=start_text, device="cuda")
    # print("\033[1;32m" + s2 + "\033[0m")
    print("-" * 20)

