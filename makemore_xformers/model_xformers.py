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
    def __init__(self, dictsize: int, numchar: int, emb_len: int, kqv_len: int, device = "cpu"):
        super().__init__()

        self.tok_embedding = nn.Embedding(dictsize, emb_len, device=device)       # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(numchar, emb_len, device=device)        # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        # self.tok_embedding = nn.Linear(dictsize, emb_len, bias=False, device=device)   # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        # self.pos_embedding = nn.Linear(numchar, emb_len, bias=False, device=device)    # (batch, numchar, emb_len) -> (batch, numchar, emb_len)

        self.query = nn.Linear(emb_len, kqv_len, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        self.key = nn.Linear(emb_len, kqv_len, bias=False, device=device)              # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        self.value = nn.Linear(emb_len, kqv_len, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)

        self.flatten = nn.Flatten(-2, -1)                                         # (batch, numchar, kqv_len) -> (batch, numchar*kqv_len)
        self.linear = nn.Linear(numchar * emb_len, dictsize, device=device)       # (batch, numchar*kqv_len) -> (batch, dictsize)

        self.device = device

    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        batch_size, numchar = input_tokens.shape

        input_emb = self.tok_embedding(input_tokens)   # (batch, numchar) -> (batch, numchar, emb_len)

        # print(f"{self.query.weight.shape=}")
        # print(f"{input_emb.shape=}")
        query = self.query(input_emb)                  # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        key = self.key(input_emb)                      # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        value = self.value(input_emb)                  # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)

        # get unnormalized scores for all queries and keys.
        scores = query @ key.transpose(-2, -1)                               # -> (batch, numchar, numchar)

        # ensure that e.g., token[0] doesn't know about token[1:]
        tril = torch.tril(torch.ones(numchar, numchar, device=self.device))  # -> (numchar, numchar)
        scores = torch.masked_fill(scores, tril == 0, float('-inf'))
        scores_norm = F.softmax(scores, dim=-1)                              # -> (batch, numchar, numchar)
        
        affinities = scores_norm @ input_emb                                    # -> (batch, numchar, emb_len)

        # positions = torch.arange(0, numchar, device=self.device)
        # pos_emb = self.pos_embedding(positions)          # -> (numchar, )
        # emb = input_emb + pos_emb                        # -> (batch, numchar, emb_len)
        # embflat = self.flatten(emb)                      # -> (batch, numchar * emb_len)
        # outputs = self.lm_head(embflat)                  # -> (batch, dictsize)

        aff_flat = self.flatten(affinities)
        outputs = self.linear(aff_flat)

        loss = None
        if truth is not None:
            # print(f"{outputs.shape=}")
            # print(f"{truth.shape=}")
            loss = F.cross_entropy(outputs, truth)
        
        return outputs, loss

# def make_net_xformers(numchar: int, nhead: int, emb_len: int, qkv_len: int, device="cpu"):
def make_net_xformers(numchar: int, emb_len: int, kqv_len: int, device="cpu"):
    net = XformersModule(dictsize=model.dictsize, numchar=numchar, emb_len=emb_len, kqv_len=kqv_len, device=device)
    return net
