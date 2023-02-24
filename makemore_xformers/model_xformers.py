# %%
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model

class Model(nn.Module):
    def __init__(self, dictsize: int, numchar: int, embedding_dim: int, device = "cpu"):
        super().__init__()

        self.tok_embedding = nn.Embedding(dictsize, embedding_dim)       # (batch, numchar, dictsize) => (batch, numchar, embdim)
        self.pos_embedding = nn.Embedding(numchar, embedding_dim)        # (batch, numchar, embdim) => (batch, numchar, embdim)
        self.flatten = nn.Flatten(1, 2)                                  # (batch, numchar, embdim) => (batch, numchar*embdim)
        self.lm_head = nn.Linear(embedding_dim * numchar, dictsize)      # (batch, numchar*embdim) => (batch, dictsize)

        self.tok_embedding = self.tok_embedding.to(device)
        self.pos_embedding = self.pos_embedding.to(device)
        self.lm_head = self.lm_head.to(device)

        self.numchar = numchar
        self.dictsize = dictsize
        self.device = device
    
    def forward(self, input_tokens: Tensor, truth: Tensor = None) -> Tensor:
        #    inputs = (batch, numchar, dictsize)
        # embedding = (batch, numchar, embdim)
        #   embflat = (batch, numchar*embdim)
        #    output = (batch, dictsize)
        print(f"{input_tokens.shape=}")
        tok_emb = self.tok_embedding(input_tokens)  # (batch, numchar, dictsize) -> (batch, numchar, embdim)
        print(f"{tok_emb.shape=}")
        positions = torch.arange(0, self.numchar, device=self.device)
        print(f"{positions.shape=}")
        pos_emb = self.pos_embedding(positions)     # (numchar, )
        print(f"{pos_emb.shape=}")
        emb = tok_emb + pos_emb                     # -> (batch, numchar, embdim)
        print(f"{emb.shape=}")
        embflat = self.flatten(emb)
        print(f"{embflat=}")
        outputs = self.lm_head(embflat)             # -> (batch, numchar, dictsize)
        print(f"{outputs.shape=}")

        loss = None
        if truth is not None:
            print(f"{truth=}")
            loss = F.cross_entropy(outputs, truth)
            print(f"{loss=}")
        
        return outputs, loss

def make_net_xformers(numchar: int, nhead: int, embedding_dim: int, qkv_len: int, device="cpu"):
    pass


dictsize = 27
numchar, embdim = 5, 4
device = "cuda"
dataset = model.make_data(numchar, device, "names-1000.txt")
dataloader = DataLoader(dataset[3:], batch_size=2)

in0, truth0 = next(iter(dataloader))
in0 = in0.argmax(2)
m = Model(dictsize, numchar, embdim, "cuda")
out = m.forward(in0, truth0)
out

# %%
