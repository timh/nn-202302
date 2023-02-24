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

class SelfAttentionHead(nn.Module):
    def __init__(self, dictsize: int, numchar: int, emb_len: int, kqv_len: int, device = "cpu"):
        super().__init__()

        self.query = nn.Linear(emb_len, kqv_len, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        self.key = nn.Linear(emb_len, kqv_len, bias=False, device=device)              # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        self.value = nn.Linear(emb_len, kqv_len, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)

        self.device = device
        self.kqv_len = kqv_len
        self.kqv_len_sqrt_inv = (torch.tensor(self.kqv_len) ** -0.5).item()

    def forward(self, input_emb: Tensor) -> Tensor:
        batch_size, numchar, emb_len = input_emb.shape

        # print(f"{self.query.weight.shape=}")
        # print(f"{input_emb.shape=}")
        query = self.query(input_emb)                                        # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        key = self.key(input_emb)                                            # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)

        # get unnormalized scores for all queries and keys.
        scores = query @ key.transpose(-2, -1)                               # -> (batch, numchar, numchar)
        scores *= self.kqv_len_sqrt_inv                                      # to keep variance low and prevent softmax from getting 

        # ensure that e.g., token[0] doesn't know about token[1:]
        tril = torch.tril(torch.ones(numchar, numchar, device=self.device))  # -> (numchar, numchar)
        scores = torch.masked_fill(scores, tril == 0, float('-inf'))         #    NOTE this tril stuff makes this a decoder block.
        scores_norm = F.softmax(scores, dim=-1)                              # -> (batch, numchar, numchar)
        
        value = self.value(input_emb)                                        # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        outputs = scores_norm @ value                                        # -> (batch, numchar, kqv_len)

        return outputs

class MultiHeadSelfAttention(nn.Module):
    heads: nn.Module
    def __init__(self, nhead: int, dictsize: int, numchar: int, emb_len: int, kqv_len: int, device="cpu"):
        super().__init__()
        self.heads = [SelfAttentionHead(dictsize, numchar, emb_len, kqv_len, device) for _ in range(nhead)]
    
    def forward(self, input_emb: Tensor) -> Tensor:
        output_list = [head(input_emb) for head in self.heads]
        return torch.concat(output_list, dim=-1)

class LangModel(nn.Module):
    def __init__(self, nhead: int, dictsize: int, numchar: int, emb_len: int, kqv_len: int, device="cpu"):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(dictsize=dictsize,               # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
                                                numchar=numchar, nhead=nhead,
                                                emb_len=emb_len, 
                                                kqv_len=kqv_len // nhead, 
                                                device=device)
        self.tok_embedding = nn.Embedding(dictsize, emb_len, device=device)      # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(numchar, emb_len, device=device)       # (batch, numchar, emb_len) -> (batch, numchar, emb_len)

        self.flatten = nn.Flatten(-2, -1)                                        # (batch, numchar, kqv_len) -> (batch, numchar * kqv_len)
        self.linear = nn.Linear(numchar * kqv_len, dictsize, device=device)      # (batch, numchar * kqv_len) -> (batch, dictsize)

        self.device = device
    
    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        batch_size, numchar = input_tokens.shape
        tok_emb = self.tok_embedding(input_tokens)                           # (batch, numchar) -> (batch, numchar, emb_len)
        pos_emb = self.pos_embedding(torch.arange(0, numchar, device=self.device))
        input_emb = tok_emb + pos_emb

        sa_out = self.self_attn(input_emb)                                   # (batch, numchar, emb_len) -> (batch, numchar, kqv_len)
        sa_flat = self.flatten(sa_out)                                       # (batch, numchar, kqv_len) -> (batch, numchar * kqv_len)
        outputs = self.linear(sa_flat)                                       # (batch, numchar * kqv_len) -> (batch, dictsize)

        loss = None
        if truth is not None:
            # print(f"{outputs.shape=}")
            # print(f"{truth.shape=}")
            loss = F.cross_entropy(outputs, truth)
        
        return outputs, loss
        
def make_net_xformers(nhead: int, numchar: int, emb_len: int, kqv_len: int, device="cpu"):
    return LangModel(nhead=nhead, dictsize=model.dictsize, numchar=numchar, emb_len=emb_len, kqv_len=kqv_len, device=device)
