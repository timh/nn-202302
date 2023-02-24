# %%
import sys
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model
sys.path.insert(0, "..")
from experiment import Experiment
import trainer

class XformersModule(nn.Module):
    def __init__(self, dictsize: int, numchar: int, embedding_dim: int, device = "cpu"):
        super().__init__()

        self.tok_embedding = nn.Embedding(dictsize, embedding_dim)       # (batch, numchar, dictsize) => (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(numchar, embedding_dim)        # (batch, numchar, embdim) => (batch, numchar, embdim)
        self.flatten = nn.Flatten(1, 2)                                  # (batch, numchar, embdim) => (batch, numchar*embdim)
        self.lm_head = nn.Linear(embedding_dim * numchar, dictsize)      # (batch, numchar*embdim) => (batch, dictsize)

        self.tok_embedding = self.tok_embedding.to(device)
        self.pos_embedding = self.pos_embedding.to(device)
        self.flatten = self.flatten.to(device)
        self.lm_head = self.lm_head.to(device)

        # self.register_buffer("device", self.device)
        self.device = device

    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        batch_size, numchar = input_tokens.shape

        tok_emb = self.tok_embedding(input_tokens)       # (batch, numchar) -> (batch, numchar, embdim)
        positions = torch.arange(0, numchar, device=self.device)
        pos_emb = self.pos_embedding(positions)          # -> (numchar, )
        emb = tok_emb + pos_emb                          # -> (batch, numchar, embdim)
        embflat = self.flatten(emb)                      # -> (batch, numchar * embdim)
        outputs = self.lm_head(embflat)                  # -> (batch, dictsize)

        loss = None
        if truth is not None:
            loss = F.cross_entropy(outputs, truth)
        
        return outputs, loss

# def make_net_xformers(numchar: int, nhead: int, embedding_dim: int, qkv_len: int, device="cpu"):
def make_net_xformers(numchar: int, embedding_dim: int, device="cpu"):
    net = XformersModule(dictsize=model.dictsize, numchar=numchar, embedding_dim=embedding_dim, device=device)
    return net


# dictsize = 27
# numchar, embdim = 5, 4
# device = "cuda"
# dataset = model.make_data(numchar, device, "names-1000.txt", use_long=True, inputs_argmax=True)
# dataloader = DataLoader(dataset[3:], batch_size=2)

# in0, truth0 = next(iter(dataloader))
# net = XformersModule(dictsize, numchar, embdim, "cuda")
# out = net.forward(in0, truth0)
# out

