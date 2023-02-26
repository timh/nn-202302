import math

import torch
from torch import nn, Tensor
from model_xformers import PositionalEncoding
from model_utils import TextMapper

# ntokens = len(vocab)  # size of vocabulary
# emsize = 200  # embedding dimension
hidden_len = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2  # number of heads in nn.MultiheadAttention
# dropout = 0.2  # dropout probability

class TransformerModel(nn.Module):

    def __init__(self, dictsize: int, emb_len: int, nhead: int, 
                 nlayers: int, hidden_len: int, dropout: float, device="cpu"):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_len, dropout, device=device)
        encoder_layers = nn.TransformerEncoderLayer(emb_len, nhead, hidden_len, dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers).to(device)
        self.encoder = nn.Embedding(dictsize, emb_len, device=device)
        self.emb_len = emb_len
        self.decoder = nn.Linear(emb_len, dictsize, device=device)

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
            src: Tensor, shape [batch_size, numchar]
            src_mask: Tensor, shape [numchar, numchar]

        Returns:
            output Tensor of shape [batch_size, numchar, dictsize]
        """
        numchar = src.shape[-1]
        src_mask = generate_square_subsequent_mask(numchar, device=src.device)
        src = self.encoder(src) * math.sqrt(self.emb_len)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)
