from typing import List, Dict, Tuple, Literal, Set, Union, Callable
import re
import math
from pathlib import Path
import csv

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import tokens

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_len: int, max_len: int = 5000, device="cpu"):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_len, 2) * (-math.log(10000.0) / emb_len))
        pos_enc = torch.zeros(1, max_len, emb_len)
        # print(f"{pos_enc.shape=} {torch.sin(position * div_term).shape=}")
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc.to(device))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: Tensor, shape [batch_size, seq_len, emb_len]
        """
        out = inputs + self.pos_enc[0, :inputs.shape[1]]
        return out
    
def predict(net: nn.Module, 
            seq_len: int, num_preds: int, 
            tokenizer: tokens.Tokenizer, dictionary: tokens.Dictionary, 
            start_text = "", device="cpu"):
    net.eval()

    if start_text:
        inputs = tokenizer.tokenize(start_text)
        inputs = dictionary.words_to_tensors(inputs, device=device)
        inputs = torch.unsqueeze(inputs, 0)
    else:
        inputs = torch.zeros((1, 1), device=device, dtype=torch.long)

    nextinputs = torch.zeros_like(inputs, device=device)

    res = start_text
    for i in range(len(start_text), num_preds + len(start_text)):
        outputs = net(inputs[:i + 1])
        outputs = F.softmax(outputs, -1)
        word_idx = torch.multinomial(outputs[0, -1], 1).item()

        res += dictionary.tokens_to_str([word_idx])

        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = word_idx
        inputs, nextinputs = nextinputs, inputs

    return res

def loss_fn(seq_len: int, vocab_len: int) -> Callable[[Tensor, Tensor], Tensor]:
    def ce(outputs: Tensor, truth: Tensor) -> Tensor:
        batch_size = outputs.shape[0]
        outflat = outputs.view(batch_size * seq_len, vocab_len)
        truthflat = truth.view(batch_size * seq_len)
        return F.cross_entropy(outflat, truthflat)

    return ce

