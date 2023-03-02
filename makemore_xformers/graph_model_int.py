# %%
from pathlib import Path
import torch
from torch import Tensor
import torch.utils.tensorboard as tboard

# path = Path(sys.argv[1])
filename = "runs/mm-ss4tut-karpathy-v2c-seqlen 256, wordlen 1, nhead 6, nlayers 6, hidlen 1536, emblen 384, optim adamw, startlr 1.0E-03, endlr 1.0E-04, sched StepLR, batch 128, minicnt 2, epochs 500.torch"
# filename = "/home/tim/devel/nanogpt/out-shakespeare-char/ckpt.pt"
path = Path(filename)

# batch, seqlen

inputs = torch.zeros((1, 10), dtype=torch.long, device="cuda")
inputs.requires_grad_(False)

dirname = f"runs/graph-{path.name}"
writer = tboard.SummaryWriter(log_dir=dirname)
model = torch.load(path)

n_params = sum(p.numel() for p in model.parameters())

print(model)
print(f"{n_params/1e6=}")

with torch.no_grad():
    model.eval()
    torch.inference_mode(True)
    inputs.requires_grad_(False)
    writer.add_graph(model, inputs)

# %%
