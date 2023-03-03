# %%
import sys
import torch
import importlib

import model_utils
import model

for m in [model_utils, model]:
    importlib.reload(m)

filename = "runs/python-100000_3000-seqlen 128, wordlen 2, nhead 6, nlayers 6, emblen 192, hidlen 768, optim_type adamw, sched_type StepLR, startlr 1.00E-03, endlr 1.00E-04, batch 256, minicnt 1, epochs 3000, elapsed 607.40s, vloss 2.379.ckpt"
state_dict = torch.load(filename)
exp = model.load_experiment(state_dict, device="cuda")

# %%
pred_len = 200
start_text = "\n"
if len(sys.argv) > 1:
    pred_len = int(sys.argv[1])
if len(sys.argv) > 2:
    start_text = sys.argv[2]
while True:
    s = model_utils.predict(net=exp.net, seq_len=exp.seqlen, num_preds=pred_len, tokenizer=exp.tokenizer, dictionary=exp.dictionary, start_text=start_text, device="cuda")
    print("\033[1;32m" + s + "\033[0m")
    # s2 = model_utils.predict2(net=exp.net, seq_len=exp.seqlen, num_preds=pred_len, tokenizer=exp.tokenizer, dictionary=exp.dictionary, start_text=start_text, device="cuda")
    # print("\033[1;32m" + s2 + "\033[0m")
    print("-" * 20)

# %%
textmap.to_str(torch.Tensor([0]))
