from typing import List, Callable, Tuple
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class QuoteDataset(Dataset):
    inputs: torch.Tensor
    truth: torch.Tensor

    def __init__(self, inputs: torch.Tensor, truth: torch.Tensor):
        if len(inputs) != len(truth):
            raise ValueError(f"{len(inputs)=} != {len(truth)=}")
        self.inputs = inputs
        self.truth = truth

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.inputs[index], self.truth[index])

def make_net(num_quotes: int, num_hidden: int, hidden_size: int, device="cpu") -> nn.Module:
    mods: List[nn.Module] = list()

    mods.append(nn.Linear(num_quotes, hidden_size))
    for _ in range(num_hidden):
        lin = nn.Linear(hidden_size, hidden_size)
        # with torch.no_grad():
        #     lin.weight *= 10.0
        mods.append(lin)
        mods.append(nn.BatchNorm1d(hidden_size))
        mods.append(nn.LeakyReLU())
    mods.append(nn.Linear(hidden_size, 1)) # next price

    net = nn.Sequential(*mods)
    return net.to(device)

# file
def read_quotes(filename: str) -> torch.Tensor:
    quoteslist: List[float] = list()

    # Date,Time,Open,High,Low,Close,Volume,OpenInt
    # 2015-11-24,15:35:00,6.99,6.99,6.92,6.95,5971,0
    # 2015-11-24,15:40:00,6.99,7.1,6.99,7.055,3000,0
    # 2015-11-24,15:45:00,7.02,7.0699,6.96,6.97,4900,0
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            quoteslist.append(float(row['Open']))
    
    return torch.tensor(quoteslist)

def make_examples(all_quotes: torch.Tensor, net_quotes_len: int, train_split: float, device="cpu") -> Tuple[Dataset, Dataset]:
    all_quotes_len = all_quotes.shape[-1]
    train_raw_len = int(all_quotes_len * train_split)

    train_quotes = all_quotes[:train_raw_len]
    val_quotes = all_quotes[train_raw_len:]

    train_num = len(train_quotes) - net_quotes_len - 1
    val_num = len(val_quotes) - net_quotes_len - 1

    train_inputs = torch.zeros((train_num, net_quotes_len))
    train_truth = torch.zeros((train_num, 1))
    val_inputs = torch.zeros((val_num, net_quotes_len))
    val_truth = torch.zeros((val_num, 1))

    for i in range(train_num):
        train_inputs[i] = train_quotes[i:i + net_quotes_len]
        train_truth[i][0] = train_quotes[i + net_quotes_len + 1]
    
    for i in range(val_num):
        val_inputs[i] = val_quotes[i:i + net_quotes_len]
        val_truth[i][0] = val_quotes[i + net_quotes_len + 1]
    
    train_inputs = train_inputs.to(device)
    train_truth = train_truth.to(device)
    val_inputs = val_inputs.to(device)
    val_truth = val_truth.to(device)

    train_data = QuoteDataset(train_inputs, train_truth)
    val_data = QuoteDataset(val_inputs, val_truth)

    return train_data, val_data

def simulate(net: nn.Module, val_dataloader: DataLoader, num_steps: int = 0, print = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    actual_quotes = torch.cat([truth for inputs, truth in val_dataloader])
    pred_quotes = torch.zeros_like(actual_quotes)
    num_quotes = len(actual_quotes)

    with torch.no_grad():
        net.eval()
        first_batch = next(iter(val_dataloader))
        first_sample = first_batch[0]

        inputs = first_sample.clone()
        inputs_next = torch.zeros_like(inputs)

        if num_steps == 0:
            num_steps = num_quotes
        for i in range(num_steps):
            out = net(inputs)
            if print and num_steps < print:
                input_list = inputs[0].tolist()[-5:]
                input_list_str = ", ".join([format(i, ".2f") for i in input_list])
                print(f" #{i:2}   in=({input_list_str})  |  out={out[0][0]:.2f}  |  real={actual_quotes[i][0]:.2f}")

            inputs_next[0][0:-1] = inputs[0][1:]
            inputs_next[0][-1] = out[0][0]

            temp = inputs
            inputs = inputs_next
            inputs_next = temp

            pred_quotes[i] = out[0][0]
    
    net.train()
    
    return actual_quotes, pred_quotes
