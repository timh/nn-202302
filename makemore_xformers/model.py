from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

dictsize = 26 + 1

def make_data(numchar: int, device="cpu"):
    names = open("names.txt").read().splitlines()

    num_names = len(names)
    num_chars = sum(len(name) for name in names)

    inputs_res: List[torch.Tensor] = list()
    truth_res: List[torch.Tensor] = list()

    for nidx, name in enumerate(names):
        name_inputs = torch.zeros(len(name) + numchar, dictsize)
        for cidx, ch in enumerate(name):
            ch = torch.tensor(ord(ch) - ord('a'))
            name_inputs[numchar + cidx] = F.one_hot(ch, dictsize)

        for name_exidx in range(len(name) + 1):
            inputs = name_inputs[name_exidx : name_exidx + numchar]
            if name_exidx < len(name):
                truth = torch.tensor(ord(name[name_exidx]) - ord('a'))
            else:
                truth = torch.tensor(0)
            
            inputs_res.append(inputs.to(device))
            truth_res.append(truth.to(device))

    return list(zip(inputs_res, truth_res))

class SimpleEmbedding(nn.Module):
    embedding: torch.Tensor
    numchar: int
    embedding_dim: int

    def __init__(self, numchar: int, embedding_dim: int, device="cpu"):
        super().__init__()
        self.embedding = torch.randn((dictsize, embedding_dim), device=device)
        self.numchar = numchar
        self.embedding_dim = embedding_dim
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = batch * 1
        outputs = inputs @ self.embedding

        outputs = outputs.view(-1, self.numchar * self.embedding_dim)
        return outputs

    def __repr__(self):
        return f"SimpleEmbedding(numchar={self.numchar}, embedding_dim={self.embedding_dim})"

def make_net(numchar: int, embedding_dim: int, num_hidden: int, hidden_size: int, device="cpu"):
    # mods = [nn.Embedding(dictsize, embedding_dim)]
    mods = [SimpleEmbedding(numchar, embedding_dim, device)]
    for i in range(num_hidden):
        if i == 0:
            mods.append(nn.Linear(numchar * embedding_dim, hidden_size, device=device))
        else:
            mods.append(nn.Linear(hidden_size, hidden_size, device=device))
        if i != num_hidden - 1:
            mods.append(nn.ReLU())
    mods.append(nn.Linear(hidden_size, dictsize, device=device))

    return nn.Sequential(*mods)

def inference(numchar: int, embedding_dim: int, num_pred_chars: int, net: nn.Module, device="cpu") -> str:
    net.eval()

    inputs = torch.zeros((1, numchar, dictsize), device=device)

    # seed one character into the input.
    randchar = torch.randint(1, 27, (1,), device=device)
    # print(f"{randchar=}")
    inputs[0, :-1, 0] = 1.0
    inputs[0, -1, randchar] = 1.0
    # print(f"start inputs {inputs}")

    res = chr(randchar + ord('a') - 1)
    while num_pred_chars > 0:
        outputs = net.forward(inputs)
        # print(f"{outputs=}")
        chidx = torch.argmax(outputs, dim=1).to(device)
        # print(f"{chidx=}")
        if chidx != 0:
            res += (chr(chidx + ord('a') - 1))
        num_pred_chars -= 1
        nextinputs = torch.zeros_like(inputs, device=device)
        nextinputs[:-1] = inputs
        nextinputs[-1] = F.one_hot(chidx, dictsize)
        inputs = nextinputs

    return res



