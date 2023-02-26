from typing import List, Dict, Tuple
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_len: int, dropout: float, max_len: int = 5000, device="cpu"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

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
            inputs: Tensor, shape [batch_size, numchar, emb_len]
        """
        inputs = inputs + self.pos_enc[0, :inputs.shape[1]]
        return self.dropout(inputs)
    
class TextMapper:
    dictsize: int
    numchar: int
    inputs: List[Tensor]
    truth: List[Tensor]

    char_to_token: Dict[str, int]
    token_to_char: Dict[int, str]

    def __init__(self, numchar: int, filename: str, device="cpu", dtype=torch.float):
        text = open(filename).read()

        uniq_chars = sorted(list(set(text)))
        uniq_str = "".join(uniq_chars)
        self.char_to_token = {ch: i for i, ch in enumerate(uniq_chars)}
        self.token_to_char = {i: ch for i, ch in enumerate(uniq_chars)}

        tokens = [self.char_to_token[ch] for ch in text]
        all_tokens = torch.tensor(tokens, dtype=dtype, device=device)
        all_tokens.requires_grad_(False)

        nexamples = len(all_tokens) - numchar - 1
        self.inputs = list()
        self.truth = list()

        for i in range(nexamples):
            self.inputs.append(all_tokens[i:i + numchar])
            self.truth.append(all_tokens[i + 1:i + numchar + 1])

        self.dictsize = len(uniq_chars)
        self.numchar = numchar
    
    def as_pairs(self) -> List[Tuple[Tensor, Tensor]]:
        return list(zip(self.inputs, self.truth))
    

def predict(net: nn.Module, textmap: TextMapper, num_preds: int, device="cpu"):
    net.eval()

    inputs = torch.zeros((1, textmap.numchar), device=device, dtype=torch.long)

    res = ""
    for _ in range(num_preds):
        outputs = net(inputs)
        outputs = F.softmax(outputs, -1)
        chidx = torch.multinomial(outputs[0][-1], 1).item()
        res += textmap.token_to_char[chidx]
        nextinputs = torch.zeros_like(inputs, device=device)
        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = chidx
        inputs = nextinputs

    return res

