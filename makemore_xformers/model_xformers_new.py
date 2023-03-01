import math
from typing import Callable, Tuple, Dict, Union, List
import re

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_utils import TextMapper, PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, emblen: int, nhead: int, dropout: float, device="cpu"):
        super().__init__()
        self.attn_combined = nn.Linear(emblen, emblen * 3, bias=False, device=device)
        self.dropout_score = nn.Dropout(dropout)
        self.project_out = nn.Linear(emblen, emblen, bias=False, device=device)
        self.dropout_out = nn.Dropout(dropout)

        self.nhead = nhead
        self.headsize = emblen // nhead
        self.dropout = dropout

    """
            inputs: (batch, seqlen, emblen)
        input_mask: (seq_len, )
        
            return: (batch, seqlen, emblen)
                    (batch, nhead, seqlen, seqlen)    if return_weights == True
    """
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

        if False:
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

class MLP(nn.Module):
    def __init__(self, emblen: int, hidlen: int, dropout: float, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(emblen, hidlen, bias=False, device=device)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidlen, emblen, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)
    
    """
        inputs: (batch, seqlen, emblen)

        return: (batch, seqlen, emblen)
    """
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.layer1(inputs)
        out = self.relu(out) 
        out = self.layer2(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, emblen: int, nhead: int, hidlen: int, dropout: float, device="cpu"):
        super().__init__()
        self.norm_in = nn.LayerNorm(emblen, device=device)
        self.attn = MultiHeadAttention(emblen=emblen, nhead=nhead, dropout=dropout, device=device)
        self.norm_attn = nn.LayerNorm(emblen, device=device)
        self.mlp = MLP(emblen=emblen, hidlen=hidlen, dropout=dropout, device=device)
    
    """
        inputs: (batch, seqlen, emblen)

        return: (batch, seqlen, emblen)
    """
    def forward(self, inputs: Tensor, input_mask: Tensor = None, return_weights = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # TODO: forgot to add residual here.
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

class TransformerModel2(nn.Module):
    def __init__(self, vocab_len: int, 
                 emblen: int, nhead: int, 
                 nlayers: int, hidlen: int, dropout: float, 
                 device="cpu"):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emblen, dropout, device=device)
        self.tok_encoder = nn.Embedding(vocab_len, emblen, device=device)
        self.drop_in = nn.Dropout(dropout)

        self.blocks = [Block(emblen=emblen, nhead=nhead, hidlen=hidlen, dropout=dropout, device=device)
                        for _ in range(nlayers)]
        self.norm_blocks = nn.LayerNorm(emblen, device=device)
        self.tok_decoder = nn.Linear(emblen, vocab_len, bias=False, device=device)

        self.emblen = emblen

        # init weights a la nanogpt
        nn.init.normal_(self.tok_decoder.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tok_encoder.weight, mean=0.0, std=0.02)

    """
            inputs: (batch, seqlen)   - tokens
        input_mask: (seqlen, )        - dtype=bool

            return: (batch, seqlen)   - but only the last result counts!
    """
    def forward(self, inputs: Tensor, input_mask: Tensor = None, return_weights = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Args:
            inputs: Tensor, shape [batch_size, seq_len]
            input_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_len]
        """
        batch, seqlen = inputs.shape
        device = inputs.device

        if return_weights:
            out_weights: List[Tensor] = list()

        if input_mask is None:
            input_mask = generate_square_subsequent_mask(seqlen, device)

        inputs_emb = self.tok_encoder(inputs)
        inputs_emb = self.pos_encoder(inputs_emb)
        inputs_emb = self.drop_in(inputs_emb)

        out = inputs_emb
        for block in self.blocks:
            if return_weights:
                out, out_weights_one = block(out, input_mask, True)
                out_weights.append(out_weights_one)
            else:
                out = block(out, input_mask)
        out = self.norm_blocks(out)

        out_tok = self.tok_decoder(out)
        if return_weights:
            return out_tok, out_weights
        return out_tok

def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
