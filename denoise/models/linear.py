# %%
import sys
from typing import Dict, List, Literal

import torch
from torch import Tensor
from torch import nn

sys.path.append("..")
import base_model
from . import denoise

NLType = Literal['silu', 'relu', 'gelu', 'sigmoid']
def nonlinearity(nltype: NLType) -> nn.Module:
    if nltype == 'gelu':
        return nn.GELU()
    if nltype == 'relu':
        return nn.ReLU(True)
    if nltype == 'sigmoid':
        return nn.Sigmoid()
    if nltype == 'silu':
        return nn.SiLU(True)
    raise ValueError(f"unknown nonlinearity {nltype}")

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = "in_size in_chan nlayers".split()
class DenoiseLinear(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS + ['latent_dim']
    _model_fields = BASE_FIELDS

    latent_dim: List[int]
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 nlayers: int,
                #  inner_nl
                 ):
        super().__init__()

        flat_size = in_chan * in_size * in_size

        # (in_chan, in_size, in_size) -> (in_chan * in_size * in_size)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.encoder = nn.Sequential()

        self.time_linear = nn.Sequential(
            nn.Linear(in_features=flat_size, out_features=flat_size),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=flat_size, out_features=flat_size),
            nn.SiLU(inplace=True),
        )

        self.decoder = nn.Sequential()
        self.decoder_nl = nn.Sigmoid()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=[in_chan, in_size, in_size])

        for _ in range(nlayers):
            enc_layer = nn.Sequential()
            enc_layer.append(nn.Linear(in_features=flat_size, out_features=flat_size))
            enc_layer.append(nn.LayerNorm(normalized_shape=[flat_size]))
            enc_layer.append(nn.SiLU(inplace=True))
            self.encoder.append(enc_layer)

            dec_layer = nn.Sequential()
            dec_layer.append(nn.Linear(in_features=flat_size, out_features=flat_size))
            dec_layer.append(nn.LayerNorm(normalized_shape=[flat_size]))

            dec_layer.append(nn.SiLU(inplace=True))
            self.decoder.append(dec_layer)
        
        # save hyperparameters
        self.in_size = in_size
        self.in_chan = in_chan
        self.nlayers = nlayers
        self.latent_dim = [flat_size]

    def encode(self, inputs: Tensor) -> Tensor:
        out = self.flatten(inputs)
        return self.encoder(out)
    
    def decode(self, inputs: Tensor, time: Tensor) -> Tensor:
        batch, flat_size = inputs.shape
        if time is None:
            time = torch.zeros((batch, ), device=inputs.device)

        time_embed = denoise.get_timestep_embedding(timesteps=time, emblen=flat_size)
        time_embed_out = self.time_linear(time_embed)

        out = inputs + time_embed_out
        out = self.decoder(out)
        out = self.decoder_nl(out)
        out = self.unflatten(out)

        return out

    def forward(self, inputs: Tensor, time: Tensor) -> Tensor:
        enc_out = self.encode(inputs)
        dec_out = self.decode(enc_out, time)
        return dec_out
