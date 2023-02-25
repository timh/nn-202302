# %%
import sys
from typing import List, Dict, Tuple
import datetime

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
    def __init__(self, emb_len: int, head_size: int, dropout: float, device = "cpu"):
        super().__init__()

        self.query = nn.Linear(emb_len, head_size, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        self.key = nn.Linear(emb_len, head_size, bias=False, device=device)              # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        self.value = nn.Linear(emb_len, head_size, bias=False, device=device)            # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        self.dropout = nn.Dropout(dropout)

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
        scores_norm = self.dropout(scores_norm)

        value = self.value(input_emb)                                        # (batch, numchar, emb_len) -> (batch, numchar, head_size)
        outputs = scores_norm @ value                                        # -> (batch, numchar, head_size)

        return outputs

"""(batch, numchar, emb_len) -> (batch, numchar, emb_len)"""
class MultiHeadSelfAttention(nn.Module):
    heads: nn.Sequential
    project: nn.Linear

    def __init__(self, nhead: int, emb_len: int, head_size: int, dropout: float, device="cpu"):
        super().__init__()
        self.heads = nn.Sequential(*[
            SelfAttentionHead(emb_len=emb_len, head_size=head_size, dropout=dropout, device=device)
            for _ in range(nhead)
        ])
        self.project = nn.Linear(nhead * head_size, emb_len, device=device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_emb: Tensor) -> Tensor:
        output_list = [head(input_emb) for head in self.heads.children()]
        outputs_concat = torch.cat(output_list, dim=-1)
        outputs = self.project(outputs_concat)
        outputs = self.dropout(outputs)
        return outputs

"""(batch, numchar, emb_len) -> (batch, numchar, emb_len)"""
class FeedForward(nn.Sequential):
    def __init__(self, emb_len: int, dropout: float, device="cpu"):
        super().__init__(
            nn.Linear(emb_len, 4 * emb_len, device=device),
            nn.ReLU(),
            nn.Linear(4 * emb_len, emb_len, device=device),
            nn.Dropout(dropout)
        )

"""(batch, numchar, emb_len) -> (batch, numchar, emb_len)"""
class Block(nn.Module):
    mh_attn: MultiHeadSelfAttention
    feedforward: FeedForward
    do_residual: bool
    layer_norm1: nn.LayerNorm     # between input_emb and mh_attn
    layer_norm2: nn.LayerNorm     # between mh_atten and feedforward

    def __init__(self, nhead: int, emb_len: int, dropout: float, do_layernorm = True, do_residual = True, device="cpu"):
        super().__init__()
        head_size = emb_len // nhead
        self.mh_attn = MultiHeadSelfAttention(nhead=nhead, emb_len=emb_len, head_size=head_size, dropout=dropout, device=device)
        self.feedforward = FeedForward(emb_len, dropout=dropout, device=device)

        if do_layernorm:
            self.layer_norm1 = nn.LayerNorm(emb_len, device=device)
            self.layer_norm2 = nn.LayerNorm(emb_len, device=device)
        else:
            # set them to identity
            self.layer_norm1 = lambda x: x
            self.layer_norm2 = lambda x: x
        
        self.do_residual = do_residual
    
    def forward(self, input_emb: Tensor) -> Tensor:
        if self.do_residual:
            outputs = input_emb + self.mh_attn(self.layer_norm1(input_emb))
            outputs = outputs + self.feedforward(self.layer_norm2(outputs))
        else:
            outputs = self.mh_attn(self.layer_norm1(input_emb))
            outputs = self.feedforward(self.layer_norm2(outputs))
        return outputs

class TextEncDec: pass
class LangModel(nn.Module):
    blocks: nn.Sequential
    tok_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    encdec: TextEncDec

    def __init__(self, 
                 encdec: TextEncDec,
                 nblock: int, do_layernorm: bool, do_residual: bool,
                 nhead: int, emb_len: int,
                 dropout: float,
                 device="cpu"):
        super().__init__()

        self.tok_embedding = nn.Embedding(encdec.dictsize, emb_len, device=device)   # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(encdec.numchar, emb_len, device=device)    # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        self.blocks = nn.Sequential(*[
            Block(nhead=nhead, emb_len=emb_len, dropout=dropout,                     # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
                  do_layernorm=do_layernorm, do_residual=do_residual, 
                  device=device)
            for _ in range(nblock)])
        self.layer_norm = nn.LayerNorm(emb_len, device=device)                       # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        self.linear = nn.Linear(emb_len, encdec.dictsize, device=device)             # (batch, numchar, emb_len) -> (batch, numchar, dictsize)

        self.device = device
        self.encdec = encdec

        self.apply(self._init_weights) # from karpathy

    def _init_weights(self, module):
        # from karpathy
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        batch_size, numchar = input_tokens.shape

        pos = torch.arange(0, numchar, device=self.device)

        tok_emb = self.tok_embedding(input_tokens)          # (batch, numchar) -> (batch, numchar, emb_len)
        pos_emb = self.pos_embedding(pos)                   # -> (numchar,)
        input_emb = tok_emb + pos_emb                       # -> (batch, numchar, emb_len)

        outputs = self.blocks(input_emb)                    # (batch, numchar, emb_len) -> (batch, numchar, emb_len)
        outputs = self.layer_norm(outputs) # karpathy
        outputs = self.linear(outputs)                      # (batch, numchar, emb_len) -> (batch, numchar, dictsize)

        loss = None
        if truth is not None:
            loss = F.cross_entropy(outputs[:, -1, :], truth)
        
        return outputs, loss

    def predict(self, num_preds: int, device="cpu") -> str:
        self.eval()

        inputs = torch.zeros((1, self.encdec.numchar), device=device, dtype=torch.long)

        res = ""
        for _ in range(num_preds):
            outputs, _loss = self(inputs, None)
            outputs = F.softmax(outputs, -1)
            chidx = torch.multinomial(outputs[0][-1], 1).item()
            res += self.encdec.token_to_char[chidx]
            nextinputs = torch.zeros_like(inputs, device=device)
            nextinputs[0, :-1] = inputs[0, 1:]
            nextinputs[0, -1] = chidx
            inputs = nextinputs

        return res

class TextEncDec:
    dictsize: int
    numchar: int
    inputs: List[Tensor]
    truth: List[Tensor]

    char_to_token: Dict[str, int]
    token_to_char: Dict[int, str]

    def __init__(self, numchar: int, filename: str, device="cpu", dtype=torch.float):
        text = open(filename).read()

        print(f"start: {datetime.datetime.now()}")
        uniq_chars = sorted(list(set(text)))
        uniq_str = "".join(uniq_chars)
        self.char_to_token = {ch: i for i, ch in enumerate(uniq_chars)}
        self.token_to_char = {i: ch for i, ch in enumerate(uniq_chars)}

        print(f"gen tensor: {datetime.datetime.now()}")
        tokens = [self.char_to_token[ch] for ch in text]
        all_tokens = torch.tensor(tokens, dtype=dtype, device=device)

        nexamples = len(all_tokens) - numchar - 1
        self.inputs = list()
        self.truth = list()

        print(f"gen result: {datetime.datetime.now()}")
        for i in range(nexamples):
            self.inputs.append(all_tokens[i:i + numchar])
            self.truth.append(all_tokens[i + numchar])

        print(f"done: {datetime.datetime.now()}")
        
        self.dictsize = len(uniq_chars)
        self.numchar = numchar
    
    def as_pairs(self) -> List[Tuple[Tensor, Tensor]]:
        return list(zip(self.inputs, self.truth))
    
