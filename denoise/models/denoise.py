from typing import List, Dict
import math
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import einops

import base_model
import conv_types
from models import vae

"""
(1,) -> (emblen,)
"""
class SinPositionEmbedding(nn.Module):
    def __init__(self, emblen: int):
        super().__init__()
        self.emblen = emblen
    
    def forward(self, time: Tensor) -> Tensor:
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        if len(time.shape) == 2 and time.shape[-1] == 1:
            time = time.view((time.shape[0],))
        assert len(time.shape) == 1

        half_dim = self.emblen // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=time.device)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.emblen % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0,1,0,0))
        return emb

class Block(nn.Sequential):
    def __init__(self, 
                 dir: conv_types.Direction, layer: conv_types.ConvLayer, cfg: conv_types.ConvConfig,
                 time_emblen: int):
        super().__init__()

        in_chan = layer.in_chan(dir=dir)
        if time_emblen:
            self.time_mean_std = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emblen, in_chan * 2),
            )
        else:
            self.time_mean_std = None

        if dir == 'down':
            mods = cfg.create_down(layer=layer)
        else:
            mods = cfg.create_up(layer=layer)
        self.conv = nn.Sequential(*mods)
    
    def forward(self, inputs: Tensor, time_emb: Tensor) -> Tensor:
        if self.time_mean_std is not None:
            _batch, chan, _height, _width = inputs.shape
            time_mean_std = self.time_mean_std(time_emb)
            time_mean_std = einops.rearrange(time_mean_std, "b c -> b c 1 1")
            mean, std = time_mean_std.chunk(2, dim=1)

            out = inputs * (std + 1) + mean
        else:
            out = inputs
        out = self.conv(out)
        return out

class DownBlock(Block):
    def __init__(self, layer: conv_types.ConvLayer, cfg: conv_types.ConvConfig, time_emblen: int):
        super().__init__('down', layer, cfg, time_emblen)

class UpBlock(Block):
    def __init__(self, layer: conv_types.ConvLayer, cfg: conv_types.ConvConfig, time_emblen: int):
        super().__init__('up', layer, cfg, time_emblen)

class DenoiseEncoder(nn.Sequential):
    def __init__(self, cfg: conv_types.ConvConfig, time_emblen: int):
        super().__init__()

        last_out_chan: int = None
        for i, layer in enumerate(cfg.layers):
            out_chan = layer.out_chan(dir='down')
            stride = layer.max_pool_kern or layer.stride

            if last_out_chan != out_chan and stride == 1:
                block_time_emblen = time_emblen
            else:
                block_time_emblen = 0
            self.append(DownBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen))
            last_out_chan = out_chan

        self.out_dim = cfg.get_out_dim('down')
    
    def forward(self, inputs: Tensor, time_emb: Tensor = None) -> Tensor:
        out = inputs
        
        for layer in self:
            out = layer(out, time_emb)
        return out

class DenoiseDecoder(nn.Sequential):
    def __init__(self, cfg: conv_types.ConvConfig, time_emblen: int):
        super().__init__()

        last_out_chan: int = None
        for layer in reversed(cfg.layers):
            out_chan = layer.out_chan(dir='up')
            stride = layer.max_pool_kern or layer.stride

            if last_out_chan != out_chan and stride == 1:
                block_time_emblen = time_emblen
            else:
                block_time_emblen = 0
            self.append(UpBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen))
            last_out_chan = out_chan

        self.out_dim = cfg.get_out_dim('up')
    
    def forward(self, inputs: Tensor, time_emb: Tensor) -> Tensor:
        out = inputs
        for layer in self:
            out = layer(out, time_emb)
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

        time_emblen = max([layer.out_chan('down') for layer in cfg.layers])

        self.time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.encoder = DenoiseEncoder(cfg=cfg, time_emblen=time_emblen)
        self.decoder = DenoiseDecoder(cfg=cfg, time_emblen=time_emblen)

        self.in_size = in_size
        self.in_chan = in_chan
        self.in_dim = [in_chan, in_size, in_size]
        self.latent_dim = self.encoder.out_dim
        self.conv_cfg = cfg
    
    def _get_time_emb(self, inputs: Tensor, time: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb is not None:
            return time_emb

        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        return self.time_emb(time)

    def encode(self, inputs: Tensor, time: Tensor = None, time_emb: Tensor = None) -> Tensor:
        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=time_emb)
        return self.encoder(inputs, time_emb)
    
    def decode(self, inputs: Tensor, time: Tensor = None, time_emb: Tensor = None) -> Tensor:
        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=time_emb)
        return self.decoder(inputs, time_emb)

    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)
        out = self.encode(inputs, time_emb=time_emb)
        out = self.decode(out, time_emb=time_emb)
        return out

    def metadata_dict(self) -> Dict[str, any]:
        res = super().metadata_dict()
        res.update(self.conv_cfg.metadata_dict())
        return res

    def model_dict(self, *args, **kwargs) -> Dict[str, any]:
        res = super().model_dict(*args, **kwargs)
        res.update(self.conv_cfg.metadata_dict())
        return res

    @property
    def layers_str(self) -> str:
        return self.conv_cfg.layers_str()
