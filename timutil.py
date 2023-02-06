from typing import List, Tuple
import datetime

import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader

import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torchvision.utils as vutils

def array_str(array: torch.Tensor) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def train(network: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader, test_dataloader: DataLoader,
          epochs: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    num_batches = len(train_dataloader)
    num_data = len(train_dataloader.dataset)
    batch_size = int(num_data / num_batches)

    loss_all = torch.zeros((epochs, num_batches))
    outputs_all: List[torch.Tensor] = list()

    first_print_time = datetime.datetime.now()
    last_print_time = first_print_time
    last_print_step = 0
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch, (inputs, expected) in enumerate(train_dataloader):
            now = datetime.datetime.now()
            outputs = network(inputs)

            loss = loss_fn(outputs, expected)
            optimizer.zero_grad()
            loss.backward()
            loss = loss.item()
            optimizer.step()

            loss_all[epoch][batch] = loss
            outputs_all.append(outputs)

            epoch_acc += (outputs.argmax(1) == expected).type(torch.float).sum().item()
            epoch_loss += loss

            delta_last = now - last_print_time
            if delta_last >= datetime.timedelta(seconds=1):
                delta_first = now - first_print_time
                persec_first = global_step / delta_first.total_seconds()
                persec_last = (global_step - last_print_step) / delta_last.total_seconds()
                last_print_time = now
                last_print_step = global_step
                data_idx = batch * batch_size
                print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | loss {loss:.4f} | {persec_last:.4f} steps/sec, {persec_first:.4f} steps/sec overall")
            
            global_step += 1
        
        train_loss = epoch_loss / num_batches
        train_acc = epoch_acc / num_data
        test_loss, test_acc = test(network, loss_fn, test_dataloader)
        print(f"epoch {epoch + 1}/{epochs}:")
        print(f"    train loss = {train_loss:.4f}  acc = {train_acc:.4f}")
        print(f"     test loss = {test_loss:.4f}  acc = {test_acc:.4f}")

    return loss_all, outputs_all

def test(network: nn.Module, loss_fn: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    # returns loss, accuracy
    num_batches = len(dataloader)
    num_data = len(dataloader.dataset)

    loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for inputs, expected in dataloader:
            outputs = network(inputs)
            loss += loss_fn(outputs, expected)
            accuracy += (outputs.argmax(1) == expected).type(torch.float).sum().item()
        
        loss /= num_batches
        accuracy /= num_data
    
    return (loss, accuracy)


