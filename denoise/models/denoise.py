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

class SelfAttention(nn.Module):
    def __init__(self, in_chan: int, nheads: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.headlen = in_chan // nheads
        self.in_chan = in_chan
        self.nheads = nheads
        self.attn_combined = nn.Conv2d(in_chan, 3 * nheads * self.headlen, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=in_chan, num_channels=in_chan)

        nn.init.normal_(self.attn_combined.weight, mean=0.0, std=0.0)
    
    def forward(self, inputs: Tensor, time_emb: Tensor) -> Tensor:
        batch, chan, height, width = inputs.shape

        # calc query, key, value all at the same time.
        qkv = self.attn_combined(inputs).chunk(3, dim=1)

        #         key = (batch, chan, nheads * headlen)
        #      .view -> (batch, chan, nheads, headlen)
        # .transpose -> (batch, nheads, chan, headlen)
        query, key, value = map(
            lambda t: t.view(batch, self.nheads, self.headlen, height, width).transpose(-1, -2),
            qkv
        )

        out_weights = query @ key.transpose(-1, -2) * self.scale
        out = out_weights @ value
        out = out.view(batch, chan, height, width)

        out = self.norm(out)

        return out

class DenoiseStack(nn.Sequential):
    def __init__(self, dir: conv_types.Direction, cfg: conv_types.ConvConfig, 
                 time_emblen: int, sa_nheads: int):
        super().__init__()

        layers = list(cfg.layers)
        if dir == 'up':
            layers = list(reversed(layers))

        out_chan_list = [layer.out_chan(dir=dir) for layer in layers]
        for i, layer in enumerate(layers):
            out_chan = layer.out_chan(dir=dir)
            stride = layer.max_pool_kern or layer.stride

            last_out_chan = out_chan_list[i - 1] if i > 0 else None
            next_out_chan = out_chan_list[i + 1] if i < len(out_chan_list) - 1 else None

            if last_out_chan != out_chan and stride == 1:
                block_time_emblen = time_emblen
            else:
                block_time_emblen = 0
            
            if dir == 'down':
                mod = DownBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen)
            else:
                mod = UpBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen)
            self.append(mod)

            if sa_nheads and out_chan != next_out_chan and next_out_chan:
                self.append(SelfAttention(out_chan, nheads=sa_nheads))

        self.out_dim = cfg.get_out_dim(dir)
    
    def forward(self, inputs: Tensor, time_emb: Tensor = None) -> Tensor:
        out = inputs
        
        for layer in self:
            out = layer(out, time_emb)
        return out

class DenoiseModel(base_model.BaseModel):
    _model_fields = 'in_size in_chan do_residual sa_nheads'.split(' ')
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    conv_cfg: conv_types.ConvConfig
    do_residual: bool
    sa_nheads: int

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 cfg: conv_types.ConvConfig,
                 do_residual: bool = False,
                 sa_nheads: int = 0):
        super().__init__()

        time_emblen = max([layer.out_chan('down') for layer in cfg.layers])

        self.time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down_stack = DenoiseStack(dir='down', cfg=cfg, time_emblen=time_emblen, sa_nheads=sa_nheads)
        self.up_stack = DenoiseStack(dir='up', cfg=cfg, time_emblen=time_emblen, sa_nheads=sa_nheads)

        self.sa_nheads = sa_nheads
        self.do_residual = do_residual
        self.in_size = in_size
        self.in_chan = in_chan
        self.in_dim = [in_chan, in_size, in_size]
        self.latent_dim = self.down_stack.out_dim
        self.conv_cfg = cfg
    
    def _get_time_emb(self, inputs: Tensor, time: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb is not None:
            return time_emb

        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)
        return self.time_emb(time)

    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)

        down_outputs: List[Tensor] = list()
        out = inputs
        for down_mod in self.down_stack:
            out = down_mod(out, time_emb)
            if not isinstance(down_mod, SelfAttention):
                down_outputs.append(out)
        
        for up_mod in self.up_stack:
            if self.do_residual and not isinstance(up_mod, SelfAttention):
                last_down = down_outputs.pop()
                out = up_mod(out + last_down, time_emb)
            else:
                out = up_mod(out, time_emb)

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
