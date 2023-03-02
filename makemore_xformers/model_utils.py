from typing import List, Dict, Tuple, Literal, Set, Union, Callable
import re
import math
from pathlib import Path
import csv

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

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
    
RE_WORD = re.compile(r"([^\w]*)(\w*)([^\w]*)")
class TextMapper:
    vocab_len: int
    seq_len: int
    inputs: List[Tensor]
    truth: List[Tensor]

    vocab_to_token: Dict[str, int]
    token_to_vocab: Dict[int, str]

    def __init__(self, seq_len: int, filename: str, limit_uppercase=True, wordmaxlen=0, device="cpu", dtype=torch.float):
        all_words: List[str] = list()

        # read all text
        with open(filename, "r") as file:
            text = file.read()

        # go through each line
        for line in text.split("\n"):
            # go through each part of the line, separated by a space.
            for raw_word in line.split(" "):
                match = RE_WORD.match(raw_word)
                if not match:
                    all_words.append("\n")
                    continue

                leading, word, trailing = [g.strip() for g in match.groups()]
                for part in [leading, word, trailing]:
                    if not part:
                        continue

                    if limit_uppercase and all(ch.isupper() for ch in part):
                        # all upper case = name. do them separately
                        # print(f"add {part=}")
                        all_words.extend(part)
                    elif wordmaxlen > 0:
                        while True:
                            pcur, pnext = part[:wordmaxlen], part[wordmaxlen:]
                            # print(f"{pcur=} {pnext=}")
                            all_words.append(pcur)
                            if not pnext:
                                break
                            part = pnext
                    else:
                        all_words.append(part)
                
                all_words.append(" ")

            all_words.append("\n")
        
        uniq_strs = sorted(list(set(all_words)))
        # print(f"{uniq_strs=}")

        self.vocab_to_token = {word: i for i, word in enumerate(uniq_strs)}
        self.token_to_vocab = {i: word for i, word in enumerate(uniq_strs)}

        all_tokens_list = [self.vocab_to_token[str] for str in all_words]
        all_tokens = torch.tensor(all_tokens_list, dtype=dtype, device=device)

        nexamples = len(all_tokens) - seq_len - 1
        self.inputs = list()
        self.truth = list()

        for i in range(nexamples):
            self.inputs.append(all_tokens[i:i + seq_len])
            self.truth.append(all_tokens[i + 1:i + seq_len + 1])

        self.vocab_len = len(uniq_strs)
        self.seq_len = seq_len

        print(f"TextMapper: {seq_len=} {self.vocab_len=}")
    
    def as_pairs(self) -> List[Tuple[Tensor, Tensor]]:
        return list(zip(self.inputs, self.truth))
    
    def to_str_list(self, input_list: Union[List[Tensor], Tensor]) -> str:
        if isinstance(input_list, Tensor):
            input_list = [input_list]

        res: List[str] = list()
        for input_tensor in input_list:
            if len(input_tensor) > 1:
                input_tensor = input_tensor.flatten()

            for t in input_tensor:
                res.append(self.token_to_vocab[t.item()])
        return res

    def to_str(self, input_list: Union[List[Tensor], Tensor]) -> str:
        return "".join(self.to_str_list(input_list))
    

def predict(net: nn.Module, textmap: TextMapper, seq_len: int, num_preds: int, device="cpu"):
    net.eval()

    inputs = torch.zeros((1, 1), device=device, dtype=torch.long)

    res = ""
    for i in range(num_preds):
        # print(f"{inputs.shape=} {seq_len=}")
        outputs = net(inputs)
        outputs = F.softmax(outputs, -1)
        word_idx = torch.multinomial(outputs[0, -1], 1).item()
        res += textmap.token_to_vocab[word_idx]
        if i < seq_len - 1:
            nextinputs = torch.zeros((inputs.shape[0], inputs.shape[1] + 1), device=inputs.device, dtype=inputs.dtype)
            nextinputs[0, :-1] = inputs
        else:
            nextinputs = torch.zeros_like(inputs)
            nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = word_idx
        inputs = nextinputs

    return res

def loss_fn(seq_len: int, vocab_len: int) -> Callable[[Tensor, Tensor], Tensor]:
    def ce(outputs: Tensor, truth: Tensor) -> Tensor:
        batch_size = outputs.shape[0]
        outflat = outputs.view(batch_size * seq_len, vocab_len)
        truthflat = truth.view(batch_size * seq_len)
        return F.cross_entropy(outflat, truthflat)

    return ce

