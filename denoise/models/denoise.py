from typing import List, Dict, Tuple, Callable
from functools import reduce
import operator
import math
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import base_model
from convolutions import DownStack, UpStack
from conv_types import ConvConfig
import noised_data
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

class DenoiseModel(base_model.BaseModel):
    _model_fields = 'in_latent_dim emblen nlinear use_timestep'.split(' ')
    _metadata_fields = _model_fields + ['bottleneck_dim']

    mid_in: nn.Sequential
    mid_time_embed: nn.Sequential
    mid_out: nn.Sequential

    in_latent_dim: List[int]
    bottleneck_dim: List[int]
    emblen: int
    nlinear: int
    use_timestep: bool
    conv_cfg: ConvConfig

    def __init__(self, *,
                 in_latent_dim: List[int], 
                 cfg: ConvConfig,
                 emblen: int = 0,
                 nlinear: int = 0,
                 use_timestep: bool = True):
        super().__init__()

        chan, size, _height = in_latent_dim
    
        self.downstack = DownStack(image_size=size, nchannels=chan, cfg=cfg)
        out_dim = self.downstack.out_dim
        
        if emblen and not nlinear:
            raise ValueError(f"{emblen=} but nlinear=0!")

        if emblen:
            out_flat = reduce(operator.mul, out_dim, 1)

            self.mid_in = nn.Sequential()
            self.mid_in.append(nn.Flatten(start_dim=1, end_dim=-1))
            for i in range(nlinear):
                in_features = out_flat if i == 0 else emblen
                self.mid_in.append(nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=emblen),
                    cfg.create_inner_norm(out_shape=(emblen,)),
                    cfg.create_linear_nl(),
                ))

            self.mid_out = nn.Sequential()
            for i in range(nlinear):
                out_features = out_flat if i == nlinear - 1 else emblen
                self.mid_out.append(nn.Sequential(
                    nn.Linear(in_features=emblen, out_features=out_features),
                    cfg.create_inner_norm(out_shape=(out_features,)),
                    cfg.create_linear_nl()
                ))
            self.mid_out.append(nn.Unflatten(dim=1, unflattened_size=out_dim))

        if use_timestep:
            if emblen:
                timelen = emblen
            else:
                timelen = out_dim[0]

            self.mid_time_embed = nn.Sequential(
                nn.Linear(in_features=timelen, out_features=timelen),
                cfg.create_linear_nl(),
                nn.Linear(in_features=timelen, out_features=timelen),
                cfg.create_linear_nl(),
            )

        self.upstack = UpStack(image_size=size, nchannels=chan, cfg=cfg)

        self.bottleneck_dim = out_dim
        self.in_latent_dim = in_latent_dim
        self.emblen = emblen
        self.nlinear = nlinear
        self.use_timestep = use_timestep
        self.conv_cfg = cfg
    
    def encode(self, latent_in, timesteps: Tensor = None) -> Tensor:
        out = self.downstack(latent_in)
        return out
    
    def decode(self, out: Tensor, timesteps: Tensor = None) -> Tensor:
        if timesteps is None:
            timesteps = torch.zeros((out.shape[0],), device=out.device)

        if self.emblen:
            out = self.mid_in(out)
            if self.use_timestep:
                time_embed = get_timestep_embedding(timesteps=timesteps, emblen=self.emblen)
                out = self.mid_time_embed(time_embed + out)
            out = self.mid_out(out)
        elif self.use_timestep:
            _batch, chan, _width, _height = out.shape
            time_embed = get_timestep_embedding(timesteps=timesteps, emblen=chan)
            time_embed_out = self.mid_time_embed(time_embed)[:, :, None, None]
            out = out + time_embed_out

        out = self.upstack(out)
        return out

    def forward(self, latent_in: Tensor, timesteps: Tensor = None) -> Tensor:
        out = self.encode(latent_in)
        return self.decode(out, timesteps)

    def metadata_dict(self) -> Dict[str, any]:
        res = super().metadata_dict()
        res.update(self.conv_cfg.metadata_dict())
        return res

    def model_dict(self, *args, **kwargs) -> Dict[str, any]:
        res = super().model_dict(*args, **kwargs)
        res.update(self.conv_cfg.metadata_dict())
        return res
