# %%
import sys
import torch
import importlib

import model_utils
import model

for m in [model_utils, model]:
    importlib.reload(m)

filename = "runs/shakespeare_2000-seqlen 128, wordlen 1, nhead 2, nlayers 2, emblen 384, hidlen 1536, optim_type adamw, sched_type StepLR, startlr 1.00E-03, endlr 1.00E-04, batch 128, minicnt 2, epochs 2000, elapsed 213.33s, vloss 1.679.ckpt"
model, treader = model.load_model_and_reader(filename, "shakespeare.txt", device="cuda")

# %%
pred_len = 200
start_text = ""
if len(sys.argv) > 1:
    pred_len = int(sys.argv[1])
if len(sys.argv) > 2:
    start_text = sys.argv[2]
while True:
    s = model_utils.predict(net=model, seq_len=treader.seq_len, num_preds=pred_len, tokenizer=treader.tokenizer, dictionary=treader.dictionary, start_text=start_text, device="cuda")
    print(s)
    print("-" * 20)

# %%
textmap.to_str(torch.Tensor([0]))
