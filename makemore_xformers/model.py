# %%
import math
from typing import Callable, Tuple, Dict, Union, List, Optional
import dataclasses
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
import tokens


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
    def __init__(self, emblen: int, nhead: int, hidlen: int, dropout: float, use_flash = False, device = "cpu"):
        super().__init__()
        self.norm_in = nn.LayerNorm(emblen, device=device)
        self.attn = MultiHeadAttention(emblen=emblen, nhead=nhead, dropout=dropout, use_flash=use_flash, device=device)
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
                 use_flash = False,
                 device = "cpu"):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emblen, device=device)
        self.tok_encoder = nn.Embedding(vocablen, emblen, device=device)
        self.drop_in = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(emblen=emblen, nhead=nhead, hidlen=hidlen, dropout=dropout, use_flash=use_flash, device=device)
            for _ in range(nlayers)
        ])
        self.norm_blocks = nn.LayerNorm(emblen, device=device)
        self.tok_decoder = nn.Linear(emblen, vocablen, bias=False, device=device)

        self.vocablen = vocablen
        self.emblen = emblen
        self.nhead = nhead
        self.nlayers = nlayers
        self.hidlen = hidlen
        self.dropout = dropout

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

@dataclass(kw_only=True)
class TextExperiment(Experiment):
    seqlen: int
    wordlen: int
    vocablen: int = 0
    use_flash: bool
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

    seed: Optional[int] = None

    tokenizer: tokens.Tokenizer = None
    dictionary: tokens.Dictionary = None

    # from parent
    label: str = None
    net: nn.Module = None
    loss_fn: Callable[[Tensor, Tensor], Tensor] = None
    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None

    """dict for saving with torch.save"""
    def state_dict(self) -> Dict[str, any]:
        fields = [field for field in dir(self)
                  if not field.startswith("_") 
                  and not field.startswith("last")
                  and not field.startswith("total")
                  and type(getattr(self, field)) in [int, str, float, bool, Tensor]]
        res = {field: getattr(self, field) for field in fields}

        res["net"] = self.net.state_dict()
        res["optim"] = self.optim.state_dict()
        res["scheduler"] = self.scheduler.state_dict()
        res["tokenizer"] = self.tokenizer
        res["dictionary"] = self.dictionary

        return res

    """descriptive, all string dictionary, for filename generation"""
    def to_dict(self) -> Dict[str, any]:
        fields = ("seqlen wordlen nhead nlayers emblen hidlen "
                  "optim_type sched_type startlr endlr "
                  "batch minicnt epochs use_flash").split(" ")

        res: Dict[str, any] = dict()
        for field in fields:
            value = getattr(self, field)

            if field in ["startlr", "endlr"]:
                value = format(value, ".2E")
            elif isinstance(field, float):
                value = format(value, ".4f")
            else:
                value = str(value) 
            res[field] = value
        
        if self.seed is not None:
            res["seed"] = self.seed
        
        return res

def from_experiment(exp: TextExperiment, device = "cpu") -> TransformerModel:
    return TransformerModel(vocablen=exp.vocablen, emblen=exp.emblen, 
                            nhead=exp.nhead, nlayers=exp.nlayers, hidlen=exp.hidlen, 
                            dropout=exp.dropout,
                            use_flash=exp.use_flash,
                            device=device)

# %%
def load_experiment(state_dict: Dict[str, any], device = "cpu") -> TextExperiment:
    state_dict = state_dict.copy()
    unwanted = "_orig_mod."
    for field, value in list(state_dict["net"].items()):
        if field.startswith(unwanted):
            state_dict["net"][field[len(unwanted):]] = state_dict["net"].pop(field)

    net_fields = "vocablen emblen nhead nlayers hidlen dropout".split(" ")
    net_args = {field: state_dict[field] for field in net_fields}

    net = TransformerModel(device=device, **net_args)
    net.load_state_dict(state_dict["net"])
    state_dict["net"] = net

    nparams = sum(p.numel() for p in net.parameters())
    descr = ", ".join(f"{field} {value}" for field, value in net_args.items())
    print(f"TransformerModel ({descr}) has {nparams / 1e6:.2f}M params")

    optimizer_dict = state_dict["optim"]
    if state_dict["optim_type"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters())
    elif state_dict["optim_type"] == "adamw":
        optimizer = torch.optim.AdamW(net.parameters())
    else:
        raise ValueError(f"unknown {state_dict['optim_type']=}")
    optimizer.load_state_dict(optimizer_dict)

    scheduler_dict = state_dict["scheduler"]
    if state_dict["sched_type"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_dict["step_size"])
    else:
        raise Exception(f"unknown {state_dict['sched_type']=}")

    state_dict["optim"] = optimizer
    state_dict["scheduler"] = scheduler
    
    exp = TextExperiment(**state_dict)
    exp.loss_fn = loss_fn(exp.seqlen, exp.vocablen)
    return exp

def load_model_and_reader(model_filename: str, text_filename: str, device = "cpu") -> Tuple[TransformerModel, TextReader]:
    state_dict = torch.load(model_filename)
    exp = load_experiment(state_dict, device=device)

    seqlen, wordlen = exp.seqlen, exp.wordlen
    treader = WordTextReader(seq_len=seqlen, wordlen=wordlen, include_special=True, filename=text_filename, device=device)

    return exp.net, treader


if __name__ == "__main__":
    state_dict = torch.load("runs/python-100000_10-seqlen 128, wordlen 2, nhead 4, nlayers 4, emblen 384, hidlen 1536, optim_type adamw, sched_type StepLR, startlr 1.00E-03, endlr 1.00E-04, batch 128, minicnt 2, epochs 10, elapsed 2.85s, vloss 4.878.ckpt")
    exp = load_experiment(state_dict)
    print(f"{exp.optim=}")
    print(f"{exp.scheduler=}")
    print(f"{exp.train_loss_hist[-1]=}")

    net = exp.net
    print("\n".join(state_dict["net"].keys()))
    # print(state_dict["net"].keys())
    # print(list(net.parameters()))


# %%
