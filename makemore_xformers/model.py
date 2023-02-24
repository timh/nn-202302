from typing import List, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

dictsize = 26 + 1

def make_data(numchar: int, device="cpu", dtype=torch.float, filename="names.txt"):
    names = open(filename).read().splitlines()

    num_names = len(names)
    num_chars = sum(len(name) for name in names)

    inputs_res: List[torch.Tensor] = list()
    truth_res: List[torch.Tensor] = list()

    for nidx, name in enumerate(names):
        # name_inputs is a 1d tensor. it has the index of the character in
        # each position, with (numchar *) leading 0's
        name_inputs = torch.zeros(len(name) + numchar,)
        name_inputs[:numchar] = 0
        for cidx, ch in enumerate(name):
            ch = torch.tensor(ord(ch) - ord('a') + 1)
            name_inputs[numchar + cidx] = ch

        # make an example for each letter of the name (which has 1, 2...numchar
        # 0's before it starts), then one additional one for the (numchar-1)
        # letters followed by 0.
        for exidx in range(len(name) + 1):
            inputs = name_inputs[exidx : exidx + numchar]
            if exidx == len(name):
                # example for trailing 0
                truth = torch.tensor(0)
            else:
                truth = torch.tensor(ord(name[exidx]) - ord('a') + 1)

            inputs = inputs.to(device, dtype=dtype)
            truth = truth.to(device, dtype=dtype)
            inputs_res.append(inputs)
            truth_res.append(truth)

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
        # inputs = batch X numchar X dictsize
        # outputs = batch X (numchar * embedding_size)
        outputs = inputs @ self.embedding
        outputs = outputs.view(-1, self.numchar * self.embedding_dim)
        return outputs

    def __repr__(self):
        return f"SimpleEmbedding(numchar={self.numchar}, embedding_dim={self.embedding_dim})"

class SimpleEmbedding2D(nn.Module):
    embedding: torch.Tensor
    embedding_dim: int

    def __init__(self, embedding_dim: int, device="cpu"):
        super().__init__()
        self.embedding = torch.randn((dictsize, embedding_dim), device=device)
        self.embedding_dim = embedding_dim
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = batch X numchar X dictsize
        # outputs = batch X numchar X embedding_size
        outputs = inputs @ self.embedding
        return outputs

    def __repr__(self):
        return f"SimpleEmbedding(embedding_dim={self.embedding_dim})"

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

def make_net2D(numchar: int, embedding_dim: int, num_hidden: int, hidden_size: int, device="cpu"):
    # mods = [nn.Embedding(dictsize, embedding_dim)]
    mods = [SimpleEmbedding2D(numchar, embedding_dim, device)]
    for i in range(num_hidden):
        if i == 0:
            mods.append(nn.Linear(embedding_dim, hidden_size, device=device))
        else:
            mods.append(nn.Linear(hidden_size, hidden_size, device=device))
        if i != num_hidden - 1:
            mods.append(nn.ReLU())
    mods.append(nn.Flatten(1, 2))
    mods.append(nn.Linear(hidden_size * numchar, dictsize, device=device))

    return nn.Sequential(*mods)

def predict(net: nn.Module, numchar: int, num_preds: int, device="cpu") -> str:
    net.eval()

    inputs = torch.zeros((1, numchar), device=device, dtype=torch.long)

    # seed one character into the input.
    randchar = torch.randint(0, 26, (1,), device=device)
    inputs[0][-1] = randchar + 1

    res = chr(randchar + ord('a'))
    while num_preds > 0:
        outputs, _loss = net(inputs, None)
        outputs = F.softmax(outputs, -1)
        chidx = torch.multinomial(outputs[0][-1], 1)
        if chidx != 0:
            res += (chr(chidx + ord('a') - 1))
        num_preds -= 1
        nextinputs = torch.zeros_like(inputs, device=device)
        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = chidx
        inputs = nextinputs

    return res



