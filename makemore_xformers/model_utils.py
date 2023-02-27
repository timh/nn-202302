from typing import List, Dict, Tuple, Literal
import re
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
            inputs: Tensor, shape [batch_size, seq_len, emb_len]
        """
        inputs = inputs + self.pos_enc[0, :inputs.shape[1]]
        return self.dropout(inputs)
    
class TextMapper:
    vocab_len: int
    seq_len: int
    inputs: List[Tensor]
    truth: List[Tensor]

    vocab_to_token: Dict[str, int]
    token_to_vocab: Dict[int, str]

    def __init__(self, seq_len: int, filename: str, words_or_chars: Literal["words", "chars"] = "chars", wordmaxlen=0, device="cpu", dtype=torch.float):
        text = open(filename).read()

        all_strs: List[str] = list()
        if words_or_chars == "chars":
            all_strs = [ch for ch in text]
        else:
            RE_WORD = re.compile(r"([^\w]*)(\w*)([^\w]*)")
            for line in text.split("\n"):
                for rawword in line.split(" "):
                    match = RE_WORD.match(rawword)
                    if not match:
                        all_strs.append("\n")
                        continue
                    leading, word, trailing = [g.strip() for g in match.groups()]
                    # print(f"{leading=}")
                    # print(f"{word=}")
                    # print(f"{trailing=}")
                    for part in [leading, word, trailing]:
                        if part:
                            if all(ch.isupper() for ch in part):
                                # all upper case = name. do them separately
                                # print(f"add {part=}")
                                all_strs.extend(part)
                            elif wordmaxlen > 0:
                                while True:
                                    pcur, pnext = part[:wordmaxlen], part[wordmaxlen:]
                                    # print(f"{pcur=} {pnext=}")
                                    all_strs.append(pcur)
                                    if not pnext:
                                        break
                                    part = pnext
                            else:
                                all_strs.append(pcur)

                    all_strs.append(" ")
                all_strs.append("\n")
        
        uniq_strs = sorted(list(set(all_strs)))
        # print(f"{uniq_strs=}")

        self.vocab_to_token = {word: i for i, word in enumerate(uniq_strs)}
        self.token_to_vocab = {i: word for i, word in enumerate(uniq_strs)}

        all_tokens_list = [self.vocab_to_token[str] for str in all_strs]
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
    

def predict(net: nn.Module, textmap: TextMapper, num_preds: int, device="cpu"):
    net.eval()

    inputs = torch.zeros((1, textmap.seq_len), device=device, dtype=torch.long)

    res = ""
    for _ in range(num_preds):
        outputs = net(inputs)
        outputs = F.softmax(outputs, -1)
        chidx = torch.multinomial(outputs[0][-1], 1).item()
        res += textmap.token_to_vocab[chidx]
        nextinputs = torch.zeros_like(inputs, device=device)
        nextinputs[0, :-1] = inputs[0, 1:]
        nextinputs[0, -1] = chidx
        inputs = nextinputs

    return res

