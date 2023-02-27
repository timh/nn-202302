import math
from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_xformers import PositionalEncoding
from model_utils import TextMapper

# ntokens = len(vocab)  # size of vocabulary
# emsize = 200  # embedding dimension
hidden_len = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2  # number of heads in nn.MultiheadAttention
# dropout = 0.2  # dropout probability

class TransformerModel(nn.Module):

    def __init__(self, vocab_len: int, emb_len: int, nhead: int, 
                 nlayers: int, hidden_len: int, dropout: float, device="cpu"):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_len, dropout, device=device)
        encoder_layers = nn.TransformerEncoderLayer(emb_len, nhead, hidden_len, dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers).to(device)
        self.encoder = nn.Embedding(vocab_len, emb_len, device=device)
        self.emb_len = emb_len
        self.decoder = nn.Linear(emb_len, vocab_len, device=device)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_len]
        """
        seq_len = src.shape[-1]
        src_mask = generate_square_subsequent_mask(seq_len, device=src.device)
        src = self.encoder(src) * math.sqrt(self.emb_len)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_mask)
        output = output.transpose(1, 0)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

def loss_fn(seq_len: int, vocab_len: int) -> Callable[[Tensor, Tensor], Tensor]:
    def ce(outputs: Tensor, truth: Tensor) -> Tensor:
        batch_size = outputs.shape[0]
        outflat = outputs.view(batch_size * seq_len, vocab_len)
        truthflat = truth.view(batch_size * seq_len)
        return F.cross_entropy(outflat, truthflat)

    return ce
