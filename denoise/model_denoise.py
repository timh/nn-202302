from typing import List

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

import base_model
from convolutions import DownStack, UpStack
from conv_types import ConvConfig

import image_latents

class DenoiseModel(base_model.BaseModel):
    _metadata_fields = []
    _model_fields = []

    def __init__(self, latent_dim: List[int], cfg: ConvConfig):
        super().__init__()
        chan, size, _height = latent_dim

        self.downstack = DownStack(image_size=size, nchannels=chan, cfg=cfg)
        self.upstack = UpStack(image_size=size, nchannels=chan, cfg=cfg)
        self.latent_dim = latent_dim
    
    def forward(self, latent_in: Tensor):
        out = self.downstack(latent_in)
        out = self.upstack(out)
        return out

