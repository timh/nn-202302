import math
from typing import Callable, Tuple, Dict, Union, List
from dataclasses import dataclass
import re
import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tokens import WordTextReader, TextReader
from torch.utils.data import DataLoader

sys.path.insert(0, "..")
from experiment import Experiment


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
inputs: (batch, seqlen, emblen)
return: (batch, seqlen, emblen)
"""
class PositionalEncoding(nn.Module):
    def __init__(self, emblen: int, max_len: int = 5000, device="cpu"):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emblen, 2) * (-math.log(10000.0) / emblen))
        pos_enc = torch.zeros(1, max_len, emblen)
        # print(f"{pos_enc.shape=} {torch.sin(position * div_term).shape=}")
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc.to(device))

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs + self.pos_enc[0, :inputs.shape[1]]
        return out


"""
Multi Head Attention. All heads are computed in batch, at once.

        inputs: (batch, seqlen, emblen)
    input_mask: (seq_len, )
    
        return: (batch, seqlen, emblen)
                (batch, nhead, seqlen, seqlen)    if return_weights == True
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, emblen: int, nhead: int, dropout: float, use_flash = False, device = "cpu"):
        super().__init__()
        self.attn_combined = nn.Linear(emblen, emblen * 3, bias=False, device=device)
        self.dropout_score = nn.Dropout(dropout)
        self.project_out = nn.Linear(emblen, emblen, bias=False, device=device)
        self.dropout_out = nn.Dropout(dropout)

        self.use_flash = use_flash
        if self.use_flash and not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("use_flash set, but scaled_dot_product_attention not available; need pytorch nightly!")

        self.nhead = nhead
        self.headsize = emblen // nhead
        self.dropout = dropout

    def forward(self, inputs: Tensor, input_mask: Tensor = None, return_weights = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        batch, seqlen, emblen = inputs.shape
        device = inputs.device

        # calc query, key, value all at the same time.
        query, key, value = self.attn_combined(inputs).split(emblen, dim=-1)

        # change from
        # key/query/value = (batch, seqlen, emblen)
        #          .view -> (batch, seqlen, nhead, headsize)
        #      .tranpose -> (batch, nhead, seqlen, headsize)
        query = query.view(batch, seqlen, self.nhead, self.headsize).transpose(1, 2)
        key = key.view(batch, seqlen, self.nhead, self.headsize).transpose(1, 2)
        value = value.view(batch, seqlen, self.nhead, self.headsize).transpose(1, 2)

        if self.use_flash:
            # -> (batch, nhead, seqlen, headsize)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=input_mask, dropout_p=self.dropout)
        else:
            #    (batch, nhead, seqlen, headsize) @ (batch, nhead, headsize, seqlen)
            # -> (batch, nhead, seqlen, seqlen)
            out_weights = query @ key.transpose(-1, -2) * (1.0 / math.sqrt(self.headsize))
            out = out_weights + input_mask
            out = F.softmax(out, dim=-1)

            #    (batch, nhead, seqlen, seqlen) 
            #  @ (batch, nhead, seqlen, headsize)
            # -> (batch, nhead, seqlen, headsize)
            out = self.dropout_score(out) @ value

        # view the combined heads output.
        #              (batch, nhead, seqlen, headsize)
        # .tranpose -> (batch, seqlen, nhead, headsize)   NOTE: emblen == nhead * headsize
        #     .view -> (batch, seqlen, emblen)
        out = out.transpose(1, 2).contiguous().view(batch, seqlen, emblen)

        # (batch, seqlen, emblen) -> (batch, seqlen, emblen)
        out = self.project_out(out)
        out = self.dropout_out(out)

        if return_weights:
            return out, out_weights
        return out

"""
    inputs: (batch, seqlen, emblen)

    return: (batch, seqlen, emblen)
"""
class MLP(nn.Module):
    def __init__(self, emblen: int, hidlen: int, dropout: float, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(emblen, hidlen, bias=False, device=device)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidlen, emblen, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.layer1(inputs)        # (batch, seqlen, emblen) -> (batch, seqlen, hidlen)
        out = self.relu(out)
        out = self.layer2(out)           # (batch, seqlen, hidlen) -> (batch, seqlen, emblen)
        out = self.dropout(out)
        return out

"""
    inputs: (batch, seqlen, emblen)

    return: (batch, seqlen, emblen)
            (batch, nhead, seqlen, seqlen)   - if return_weights == True
"""
class Block(nn.Module):
    def __init__(self, emblen: int, nhead: int, hidlen: int, dropout: float, device="cpu"):
        super().__init__()
        self.norm_in = nn.LayerNorm(emblen, device=device)
        self.attn = MultiHeadAttention(emblen=emblen, nhead=nhead, dropout=dropout, device=device)
        self.norm_attn = nn.LayerNorm(emblen, device=device)
        self.mlp = MLP(emblen=emblen, hidlen=hidlen, dropout=dropout, device=device)
    
    def forward(self, inputs: Tensor, input_mask: Tensor = None, return_weights = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        out = self.norm_in(inputs)
        if return_weights:
            out_attn, out_weights = self.attn(out, input_mask, True)
            out = out + out_attn
        else:
            out = out + self.attn(out, input_mask)
        out = self.norm_attn(out)
        out = out + self.mlp(out)

        if return_weights:
            return out, out_weights
        return out

"""
        inputs: (batch, seqlen)   - int tokens from [0, vocablen)
    input_mask: (seqlen, )        - dtype=bool

        return: (batch, seqlen)                       - but only the last result counts!
                [ (batch, nhead, seqlen, seqlen) ..]  - if return_weights == True
"""
class TransformerModel(nn.Module):
    def __init__(self, vocablen: int, 
                 emblen: int, nhead: int, 
                 nlayers: int, hidlen: int, dropout: float, 
                 device="cpu"):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emblen, device=device)
        self.tok_encoder = nn.Embedding(vocablen, emblen, device=device)
        self.drop_in = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(emblen=emblen, nhead=nhead, hidlen=hidlen, dropout=dropout, device=device)
            for _ in range(nlayers)
        ])
        self.norm_blocks = nn.LayerNorm(emblen, device=device)
        self.tok_decoder = nn.Linear(emblen, vocablen, bias=False, device=device)

        self.emblen = emblen

        # init weights like nanogpt
        nn.init.normal_(self.tok_decoder.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tok_encoder.weight, mean=0.0, std=0.02)

        for block in self.blocks:
            torch.nn.init.normal_(block.attn.project_out.weight, std=0.02/math.sqrt(2 * nlayers))
    
    def forward(self, inputs: Tensor, input_mask: Tensor = None, return_weights = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        batch, seqlen = inputs.shape
        device = inputs.device

        if return_weights:
            out_weights: List[Tensor] = list()

        if input_mask is None:
            input_mask = generate_square_subsequent_mask(seqlen, device)

        inputs_emb = self.tok_encoder(inputs)               # (batch, seqlen) -> (batch, seqlen, emblen)
        inputs_emb = self.pos_encoder(inputs_emb)           # (batch, seqlen, emblen) -> (batch, seqlen, emblen)
        inputs_emb = self.drop_in(inputs_emb)

        out = inputs_emb
        for block in self.blocks:
            if return_weights:
                # (batch, seqlen, emblen) -> [ (batch, seqlen, emblen),
                #                              (batch, nhead, seqlen, seqlen) ]
                out, out_weights_one = block(out, input_mask, True)
                out_weights.append(out_weights_one)
            else:
                out = block(out, input_mask)
        out = self.norm_blocks(out)

        # (batch, seqlen, emblen) -> (batch, seqlen, vocablen)
        out_tok = self.tok_decoder(out)
        if return_weights:
            return out_tok, out_weights
        return out_tok

"""
seqlen: int
return: (seqlen, seqlen, dtype=bool)
"""
def generate_square_subsequent_mask(seqlen: int, device="cpu", dtype=torch.float) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(seqlen, seqlen, device=device) * float('-inf'), diagonal=1).to(dtype=dtype)

"""Text Prediction loss function generator.
returns loss function which takes:
    inputs: (batch, seqlen, vocablen)
    truth: (batch, seqlen, vocablen)
"""
def loss_fn(seqlen: int, vocablen: int) -> Callable[[Tensor, Tensor], Tensor]:
    def cross_ent(outputs: Tensor, truth: Tensor) -> Tensor:
        # print(f"{outputs.shape=} {truth.shape=}")
        batch_size = outputs.shape[0]
        outflat = outputs.view(batch_size * seqlen, vocablen)
        truthflat = truth.view(batch_size * seqlen)
        return F.cross_entropy(outflat, truthflat)

    return cross_ent


RE_AFTER_BASENAME = re.compile(r"[\w\d-]+-(.*)\.torch")
def _parse_model_filename(model_filename: str) -> Dict[str, str]:
    if model_filename.startswith("runs/"):
        model_filename = model_filename[5:]
    match = RE_AFTER_BASENAME.match(model_filename)
    fields_str = match.group(1)

    fields_list = fields_str.split(", ")
    fields: Dict[str, str] = dict()
    fields["filename"] = model_filename
    for field_str in fields_list:
        key, value = field_str.split(" ")
        # if key in FIELDS_LONG:
        #     key = FIELDS_LONG[key]
        fields[key] = value
    
    return fields

def load_model_and_reader(model_filename: str, text_filename: str) -> Tuple[TransformerModel, TextReader]:
    model_fields = _parse_model_filename(model_filename)
    model: TransformerModel = torch.load(model_filename)

    seq_len = int(model_fields["seqlen"])
    wordlen = int(model_fields["wordlen"])
    treader = WordTextReader(seq_len=seq_len, wordlen=wordlen, include_special=True, filename=text_filename, device="cuda")

    return model, treader

@dataclass(kw_only=True)
class TextExperiment(Experiment):
    seqlen: int
    wordlen: int
    vocablen: int = 0
    nhead: int
    nlayers: int
    emblen: int
    hidlen: int

    optim_type: str
    sched_type: str
    startlr: float
    endlr: float
    dropout: float

    batch: int
    minicnt: int
    epochs: int

    # from parent
    label: str = None
    net: nn.Module = None
    loss_fn: Callable[[Tensor, Tensor], Tensor] = None
    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None


    def to_dict(self) -> Dict[str, str]:
        fields = ("seqlen wordlen nhead nlayers emblen hidlen "
                  "optim_type sched_type startlr endlr "
                  "batch minicnt epochs").split(" ")

        res: Dict[str, str] = dict()
        for field in fields:
            value = getattr(self, field)

            if field in ["startlr", "endlr"]:
                value = format(value, ".2E")
            elif isinstance(field, float):
                value = format(value, ".4f")
            else:
                value = str(value) 
            res[field] = value
        return res

def from_experiment(exp: TextExperiment, device = "cpu") -> TransformerModel:
    return TransformerModel(vocablen=exp.vocablen, emblen=exp.emblen, 
                            nhead=exp.nhead, nlayers=exp.nlayers, hidlen=exp.hidlen, 
                            dropout=exp.dropout,
                            device=device)
