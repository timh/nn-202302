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
    def __init__(self, emb_len: int, head_size: int, device = "cpu"):
        super().__init__()

        self.query = nn.Linear(emb_len, head_size, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        self.key = nn.Linear(emb_len, head_size, bias=False, device=device)              # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        self.value = nn.Linear(emb_len, head_size, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, head_size)

        self.device = device
        self.head_size = head_size
        self.kqv_len_sqrt_inv = (torch.tensor(self.head_size) ** -0.5).item()

    def forward(self, input_emb: Tensor) -> Tensor:
        batch_size, numchar, emb_len = input_emb.shape

        query = self.query(input_emb)                                        # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        key = self.key(input_emb)                                            # (batch, numchar, emb_len) -> (batch, numchar, head_size)

        # get unnormalized scores for all queries and keys.
        scores = query @ key.transpose(-2, -1)                               # -> (batch, numchar, numchar)
        scores *= self.kqv_len_sqrt_inv                                      # to keep variance low and prevent softmax from getting 

        # ensure that e.g., token[0] doesn't know about token[1:]
        tril = torch.tril(torch.ones(numchar, numchar, device=self.device))  # -> (numchar, numchar)
        scores = torch.masked_fill(scores, tril == 0, float('-inf'))         #    NOTE this tril stuff makes this a decoder block.
        scores_norm = F.softmax(scores, dim=-1)                              # -> (batch, numchar, numchar)
        
        value = self.value(input_emb)                                        # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        outputs = scores_norm @ value                                        # -> (batch, numchar, head_size)

        return outputs

class MultiHeadSelfAttention(nn.Module):
    heads: nn.Module
    def __init__(self, nhead: int, emb_len: int, head_size: int, device="cpu"):
        super().__init__()
        self.heads = [SelfAttentionHead(emb_len, head_size, device) for _ in range(nhead)]
        self.project = nn.Linear(nhead * head_size, emb_len, device=device)
    
    def forward(self, input_emb: Tensor) -> Tensor:
        output_list = [head(input_emb) for head in self.heads]
        outputs_concat = torch.concat(output_list, dim=-1)
        outputs = self.project(outputs_concat)
        return outputs

class LangModel(nn.Module):
    def __init__(self, nhead: int, dictsize: int, numchar: int, emb_len: int, head_size: int, device="cpu"):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(nhead=nhead, emb_len=emb_len,    # (batch, numchar, emb_len) -> (batch, numchar, head_size)
                                                head_size=head_size // nhead, 
                                                device=device)
        self.tok_embedding = nn.Embedding(dictsize, emb_len, device=device)      # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(numchar, emb_len, device=device)       # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        self.linear = nn.Linear(emb_len, dictsize, device=device)

        self.device = device
    
    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        batch_size, numchar = input_tokens.shape
        tok_emb = self.tok_embedding(input_tokens)                           # (batch, numchar) -> (batch, numchar, emb_len)
        pos_emb = self.pos_embedding(torch.arange(0, numchar, device=self.device))
        input_emb = tok_emb + pos_emb

        sa_out = self.self_attn(input_emb)                                  # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        # outputs = self.flatten(sa_out)
        outputs = self.linear(sa_out)

        loss = None
        if truth is not None:
            # print(f"{outputs.shape=}")
            # print(f"{truth.shape=}")
            # emb_len = 
            # out_flat = outputs.view(batch_size * numchar, )
            loss = F.cross_entropy(outputs[:, -1, :], truth)
        
        return outputs, loss
        
def make_net_xformers(nhead: int, numchar: int, emb_len: int, head_size: int, device="cpu"):
    return LangModel(nhead=nhead, dictsize=model.dictsize, numchar=numchar, emb_len=emb_len, head_size=head_size, device=device)
