from typing import List, Tuple
import torch
from torch import nn
import datetime

def array_str(array: torch.Tensor) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def train(network: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
          inputs: torch.Tensor, expected: torch.Tensor, steps: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss_all = torch.zeros((steps, ))
    outputs_all: List[torch.Tensor] = list()

    first_print = last_print = datetime.datetime.now()
    last_step = 0
    for step in range(steps):
        now = datetime.datetime.now()
        outputs = network(inputs)

        loss = loss_fn(outputs, expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all[step] = loss
        outputs_all.append(outputs)

        delta_last = now - last_print
        if delta_last >= datetime.timedelta(seconds=1):
            delta_first = now - first_print
            persec_first = step / delta_first.total_seconds()
            persec_last = (step - last_step) / delta_last.total_seconds()
            last_print = now
            last_step = step
            print(f"step {step}/{steps} | loss {loss:.4f} | {persec_last:.4f}, {persec_first:.4f} overall")

    return loss_all, outputs_all

