import math
from typing import Callable, Tuple, Dict, Union, List
import re

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_utils import TextMapper, PositionalEncoding

# CausalSelfAttention:
# takes:
#   dropout
#   n_emb
#   n_head
#   linear layer len = n_emb * 4
#
# # tensors:
#   c_attn = key, query, value
#   c_proj = n_emb -> n_emb (output)
#   attn_dropout = after softmax
#   resid_dropout = after c_proj

# our model:
#   TransformerEncoder(Layer) does feed forward, 
class MultiHeadAttention(nn.Module):
    def __init__(self, emblen: int, nhead: int, dropout: float, device="cpu"):
        super().__init__()
        self.attn_combined = nn.Linear(emblen, emblen * 3, bias=False, device=device)
        self.dropout_score = nn.Dropout(dropout)
        self.project_out = nn.Linear(emblen, emblen, bias=False, device=device)
        self.dropout_out = nn.Dropout(dropout)
        self.nhead = nhead
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
        query = query.view(batch, seqlen, self.nhead, emblen // self.nhead).transpose(1, 2)
        key = key.view(batch, seqlen, self.nhead, emblen // self.nhead).transpose(1, 2)
        value = value.view(batch, seqlen, self.nhead, emblen // self.nhead).transpose(1, 2)

        if False:
            # -> (batch, nhead, seqlen, headsize)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=input_mask, dropout_p=self.dropout)
        else:
            #    (batch, nhead, seqlen, headsize) @ (batch, nhead, headsize, seqlen)
            # -> (batch, nhead, seqlen, seqlen)
            out_weights = query @ key.transpose(-1, -2) * (1.0 / math.sqrt(key.shape[-1]))
            out = out_weights + input_mask
            out = F.softmax(out, dim=-1)

            #    (batch, nhead, seqlen, seqlen) 
            #  @ (batch, nhead, seqlen, headsize)
            # -> (batch, nhead, seqlen, headsize)
            out = self.dropout_score(out) @ value

        # combine the outputs for combined heads.
        #              (batch, nhead, seqlen, headsize)
        # .tranpose -> (batch, seqlen, nhead, headsize)
        #     .view -> (batch, seqlen, emblen)
        out = out.transpose(1, 2).contiguous().view(batch, seqlen, emblen)

        # (batch, seqlen, emblen) -> (batch, seqlen, emblen)
        out = self.project_out(out)
        out = self.dropout_out(out)

        if return_weights:
            print(f"{out.shape=}, {out_weights.shape=}")
            return out, out_weights
        return out

class MLP(nn.Module):
    def __init__(self, emblen: int, hidlen: int, dropout: float, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(emblen, hidlen, bias=False, device=device)
        self.layer2 = nn.Linear(hidlen, emblen, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)
    
    """
        inputs: (batch, seqlen, emblen)

        return: (batch, seqlen, emblen)
    """
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.dropout(outputs)
        return outputs

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

        self.blocks = [Block(emblen=emblen, nhead=nhead, hidlen=hidlen, dropout=dropout, device=device)
                        for _ in range(nlayers)]
        self.norm_blocks = nn.LayerNorm(emblen, device=device)
        self.tok_decoder = nn.Linear(emblen, vocab_len, bias=False, device=device)

        self.emblen = emblen

        self._init_weights() # TODO

    def _init_weights(self) -> None:
        # TODO update
        initrange = 0.1
        self.tok_encoder.weight.data.uniform_(-initrange, initrange)
        # self.tok_encoder.bias.data.zero_()
        self.tok_decoder.weight.data.uniform_(-initrange, initrange)

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

        # TODO: karpathy has dropout here

        outputs = inputs_emb
        for block in self.blocks:
            if return_weights:
                outputs, out_weights_one = block(outputs, input_mask, True)
                out_weights.append(out_weights_one)
            else:
                outputs = block(outputs, input_mask)
        outputs = self.norm_blocks(outputs)

        outputs_tok = self.tok_decoder(outputs)
        if return_weights:
            return outputs_tok, out_weights
        return outputs_tok

def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
