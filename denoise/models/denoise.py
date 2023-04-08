from typing import List, Dict, Literal
from functools import reduce
import math
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import einops

import base_model
import convolutions
import conv_types
from noisegen import NoiseWithAmountFn
from models import vae

"""
(1,) -> (embedding_dim,)
"""
def get_timestep_embedding(timesteps: Tensor, emblen: int) -> Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    if len(timesteps.shape) == 2 and timesteps.shape[-1] == 1:
        timesteps = timesteps.view((timesteps.shape[0],))
    assert len(timesteps.shape) == 1

    half_dim = emblen // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if emblen % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def create_time_mlp(channels: int, cfg: conv_types.ConvConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features=channels, out_features=channels),
        cfg.linear_nl.create(),
        nn.Linear(in_features=channels, out_features=channels),
        cfg.linear_nl.create(),
    )

class ScaleByTime(nn.Sequential):
    def __init__(self, dir: conv_types.Direction, layer: conv_types.ConvLayer, cfg: conv_types.ConvConfig):
        super().__init__()

        in_chan = layer.in_chan(dir=dir)
        self.time_mlp = create_time_mlp(in_chan * 2, cfg)

        if dir == 'down':
            mods = cfg.create_down(layer=layer)
        else:
            mods = cfg.create_up(layer=layer)
        self.conv = nn.Sequential(*mods)
    
    def forward(self, inputs: Tensor, time: Tensor) -> Tensor:
        _batch, chan, _height, _width = inputs.shape
        time_embed = get_timestep_embedding(timesteps=time, emblen=chan * 2)
        time_embed_out = self.time_mlp(time_embed)
        time_embed_out = einops.rearrange(time_embed, "b c -> b c 1 1")
        scale, shift = time_embed_out.chunk(2, dim=1)

        out = inputs * (scale + 1) + shift
        out = self.conv(out)
        return out

class DenoiseEncoder(nn.Sequential):
    def __init__(self, *, cfg: conv_types.ConvConfig):
        super().__init__()
        for i, layer in enumerate(cfg.layers):
            down_layer = ScaleByTime(dir='down', layer=layer, cfg=cfg)
            self.append(down_layer)

        self.out_dim = cfg.get_out_dim('down')
    
    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        
        out = inputs
        for layer in self:
            out = layer(out, time)
        return out

class DenoiseDecoder(nn.Sequential):
    def __init__(self, *, cfg: conv_types.ConvConfig):
        super().__init__()

        for i, layer in enumerate(reversed(cfg.layers)):
            up_layer = ScaleByTime(dir='up', layer=layer, cfg=cfg)
            self.append(up_layer)

        self.out_dim = cfg.get_out_dim('up')
    
    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        
        out = inputs
        for layer in self:
            out = layer(out, time)
        return out

class DenoiseModel(base_model.BaseModel):
    _model_fields = 'in_size in_chan'.split(' ')
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    conv_cfg: conv_types.ConvConfig

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 cfg: conv_types.ConvConfig):
        super().__init__()

        self.encoder = DenoiseEncoder(cfg=cfg)
        self.decoder = DenoiseDecoder(cfg=cfg)

        self.in_size = in_size
        self.in_chan = in_chan
        self.in_dim = [in_chan, in_size, in_size]
        self.latent_dim = self.encoder.out_dim
        self.conv_cfg = cfg
    
    def encode(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        return self.encoder(inputs, time)
    
    def decode(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        return self.decoder(inputs, time)

    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        out = self.encode(inputs, time)
        out = self.decode(out, time)
        return out

    def metadata_dict(self) -> Dict[str, any]:
        res = super().metadata_dict()
        res.update(self.conv_cfg.metadata_dict())
        return res

    def model_dict(self, *args, **kwargs) -> Dict[str, any]:
        res = super().model_dict(*args, **kwargs)
        res.update(self.conv_cfg.metadata_dict())
        return res
