# %%
import sys
from typing import List, Dict, Tuple

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
        outputs_concat = torch.cat(output_list, dim=-1)
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
        tok_emb = self.tok_embedding(input_tokens)          # (batch, numchar) -> (batch, numchar, emb_len)
        pos_emb = self.pos_embedding(torch.arange(0, numchar, device=self.device))
        input_emb = tok_emb + pos_emb

        sa_out = self.self_attn(input_emb)                  # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
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

class TextEncDec:
    dictsize: int
    numchar: int
    inputs: Tensor
    truth: Tensor

    char_to_token: Dict[str, int]
    token_to_char: Dict[int, str]

    def __init__(self, numchar: int, filename: str, device="cpu", dtype=torch.float):
        text = open(filename).read()

        uniq_chars = sorted(list(set(text)))
        self.char_to_token = {ch: i for i, ch in enumerate(uniq_chars)}
        self.token_to_char = {i: ch for i, ch in enumerate(uniq_chars)}

        tokens = [self.char_to_token[ch] for ch in text]
        all_tokens = torch.tensor(tokens, dtype=dtype, device=device)

        nexamples = len(all_tokens) - numchar - 1
        self.inputs = torch.zeros((nexamples, numchar), dtype=dtype, device=device)
        self.truth = torch.zeros((nexamples,), dtype=dtype, device=device)

        for i in range(nexamples):
            self.inputs[i] = all_tokens[i:i + numchar]
            self.truth[i] = all_tokens[i + numchar]
        
        self.dictsize = len(uniq_chars)
        self.numchar = numchar
    
    def as_pairs(self, batch_size: int) -> List[Tuple[Tensor, Tensor]]:
        pairs: List[Tuple(Tensor, Tensor)] = list()
        nexamples = len(self.inputs)
        for i in range(0, nexamples, batch_size):
            start = i
            end = min(nexamples, start + batch_size)
            inputs = self.inputs[start:end]
            truth = self.truth[start:end]
            pairs.append((inputs, truth))
        return pairs
    
def make_net_xformers_big(ted: TextEncDec, nhead: int, numchar: int, emb_len: int, head_size: int, device="cpu"):
    return LangModel(nhead=nhead, dictsize=ted.dictsize, numchar=numchar, emb_len=emb_len, head_size=head_size, device=device)

def predict(ted: TextEncDec, net: nn.Module, numchar: int, num_preds: int, device="cpu") -> str:
    net.eval()

    inputs = torch.zeros((1, numchar), device=device, dtype=torch.long)

    res = ""
    while num_preds > 0:
        outputs, _loss = net(inputs, None)
        outputs = F.softmax(outputs, -1)
        chidx = torch.multinomial(outputs[0][-1], 1)
        res += ted.token_to_char[chidx]
        num_preds -= 1
        nextinputs = torch.zeros_like(inputs, device=device)
        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = chidx
        inputs = nextinputs

    return res


