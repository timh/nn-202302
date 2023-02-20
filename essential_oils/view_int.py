# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib, matplotlib.cm
import matplotlib.pyplot as plt
import importlib

import data
import model
from model import show_result, gen_examples, Config
import notebook
importlib.reload(model)
importlib.reload(notebook)

compounds_str = data.compounds_str

net = torch.load("outputs/20230219-122631-num_hidden_6-hidden_size_90.torch")
cfg = Config(compounds_str)
cfg.net = net
for m in cfg.net.modules():
    m.training = False

cfg.device = "cuda"

# %%
ex = gen_examples(cfg, 1, 11)
show_result(cfg, ex)

# %%
# notebook.show_mat(net, lambda mod: mod.weight.grad > 1.0)
def fn(mod: nn.Linear) -> torch.Tensor:
    return torch.abs(mod.weight) < 1

fig = plt.figure(0, (10, 5))
notebook.show_mat(net, fn, title="foo", fig=fig);
# plt.show(fig)





# %%
