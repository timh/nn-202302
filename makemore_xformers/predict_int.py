# %%
import torch
import importlib

import model_utils
import model_xformers_tutorial as mxt

for m in [model_utils, mxt]:
    importlib.reload(m)

filename = "runs/mm-ss4tut-sgd-fast-seq_len 32, wordmaxlen 1, nhead 2, nlayers 2, hidden_len 64, emb_len 64, vocab_len 65, dropout 0.2, batch_size 1024, batches_per_epoch 4, total_epochs 10.torch"
fieldstr = filename.replace(".torch", "")
while "-" in fieldstr:
    fieldstr = fieldstr[fieldstr.index("-") + 1:]

field_list = fieldstr.split(", ")
fields = {}
for fieldstr in field_list:
    key, value = fieldstr.split(" ")
    fields[key] = value
model: mxt.TransformerModel = torch.load(filename)

seq_len = int(fields["seq_len"])
wordmaxlen = int(fields["wordmaxlen"])
textmap = model_utils.TextMapper(seq_len=seq_len, filename="shakespeare.txt", wordmaxlen=wordmaxlen, device="cuda", dtype=torch.long)

# %%
pred = model_utils.predict(net=model, textmap=textmap, seq_len=seq_len, num_preds=50, device="cuda")
print(pred)

# %%
