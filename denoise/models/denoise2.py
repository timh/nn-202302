from typing import List, Dict
import math
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import einops
import torch.nn.functional as F

import base_model
import conv_types
from models import vae

# TODO: fix the unprocessed borders around output - probably need to run @ higher res?

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
            emb = F.pad(emb, (0,1,0,0))
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
    def __init__(self, in_chan: int, out_chan: int, nheads: int, kernel_size: int = 3):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.scale = nheads ** -0.5
        self.nheads = nheads
        self.attn_combined = nn.Conv2d(in_chan, out_chan * 3, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)

        # nn.init.normal_(self.attn_combined.weight, mean=0, std=0.02)
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch, chan, height, width = inputs.shape

        # calc query, key, value all at the same time.
        qkv = self.attn_combined(inputs).chunk(3, dim=1)

        # note: nheads * headlen = in_chan
        #     (batch, nheads * headlen, height, width)
        #  -> (batch, nheads, headlen, height * width)
        query, key, value = map(
            lambda t: einops.rearrange(t, "b (nh hl) y x -> b nh hl (y x)", nh=self.nheads),
            qkv
        )

        out = F.scaled_dot_product_attention(query, key, value)
        out = einops.rearrange(out, "b nh hl (y x) -> b (nh hl) y x", y=height)

        out = self.norm(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, in_size: int, nheads: int, clip_emblen: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.nheads = nheads

        self.clip_emblen = clip_emblen
        self.kv_linear = nn.Linear(clip_emblen, nheads * in_size * in_size)
        self.kv_unflatten = nn.Unflatten(1, (nheads, in_size, in_size))
        self.attn_kv = nn.Conv2d(nheads, nheads * 2, kernel_size=3, padding=1, bias=False)

        self.attn_q = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)
    
    def forward(self, inputs: Tensor, clip_embed: Tensor, clip_scale: float = 1.0) -> Tensor:
        batch, _chan, height, _width = inputs.shape
        if clip_embed is None:
            clip_embed = torch.zeros((batch, self.clip_emblen), device=inputs.device)
        if clip_scale is None:
            clip_scale = 1.0

        # calculate key, value on the clip embedding
        clip_flat = self.kv_linear(clip_embed)
        clip_out = self.kv_unflatten(clip_flat)
        key, value = self.attn_kv(clip_out).chunk(2, dim=1)

        # calculate query on the inputs
        query = self.attn_q(inputs)

        # note: nheads * headlen = in_chan
        #     (batch, nheads * headlen, height, width)
        #  -> (batch, nheads, headlen, height * width)
        query, key, value = map(
            lambda t: einops.rearrange(t, "b (nh hl) y x -> b nh hl (y x)", nh=self.nheads),
            [query, key, value]
        )

        out = F.scaled_dot_product_attention(query, key, value)
        out = einops.rearrange(out, "b nh hl (y x) -> b (nh hl) y x", y=height)

        out = self.norm(out)
        out = out * clip_scale
        return out

class DenoiseStack(nn.Sequential):
    def __init__(self, dir: conv_types.Direction, cfg: conv_types.ConvConfig, 
                 do_residual: bool, time_emblen: int, clip_emblen: int):
        super().__init__()

        layers = list(cfg.layers)
        if dir == 'up':
            layers = list(reversed(layers))
        
            if do_residual:
                new_layers: List[conv_types.ConvLayer] = list()
                for i, layer in enumerate(layers):
                    new_layer = layer.copy()
                    new_layer._out_chan = layer._out_chan * 2
                    new_layers.append(new_layer)
                layers = new_layers
        
        out_chan_list = [layer.out_chan(dir=dir) for layer in layers]
        for i, layer in enumerate(layers):
            in_chan = layer.in_chan(dir=dir)
            out_chan = layer.out_chan(dir=dir)
            in_size = layer.in_size(dir=dir)
            
            if layer.sa_nheads:
                self.append(SelfAttention(in_chan=in_chan, out_chan=out_chan, nheads=layer.sa_nheads))
                continue
            if layer.ca_nheads:
                self.append(CrossAttention(in_chan=in_chan, in_size=in_size, out_chan=out_chan, nheads=layer.ca_nheads, clip_emblen=clip_emblen))
                continue

            out_chan = layer.out_chan(dir=dir)
            stride = layer.max_pool_kern or layer.stride

            last_out_chan = out_chan_list[i - 1] if i > 0 else None
            if layer.time_emb:
                block_time_emblen = time_emblen
            else:
                block_time_emblen = 0
            
            if dir == 'down':
                mod = DownBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen)
            else:
                mod = UpBlock(layer=layer, cfg=cfg, time_emblen=block_time_emblen)
            self.append(mod)

        self.out_dim = cfg.get_out_dim(dir)


class DenoiseModel2(base_model.BaseModel):
    _model_fields = 'in_size in_chan do_residual clip_emblen'.split(' ')
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    conv_cfg: conv_types.ConvConfig
    do_residual: bool
    clip_emblen: int

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 cfg: conv_types.ConvConfig,
                 do_residual: bool = False,
                 clip_emblen: int = None):
        super().__init__()

        time_emblen = max([layer.out_chan('down') for layer in cfg.layers])

        self.time_emb = nn.Sequential(
            SinPositionEmbedding(emblen=time_emblen),
            nn.Linear(time_emblen, time_emblen),
            nn.GELU(),
            nn.Linear(time_emblen, time_emblen),
        )
        self.down_stack = DenoiseStack(dir='down', cfg=cfg, 
                                       do_residual=do_residual,
                                       time_emblen=time_emblen, clip_emblen=clip_emblen)
        self.up_stack = DenoiseStack(dir='up', cfg=cfg, 
                                     do_residual=do_residual,
                                     time_emblen=time_emblen, clip_emblen=clip_emblen)

        self.do_residual = do_residual
        self.clip_emblen = clip_emblen
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

    def forward(self, inputs: Tensor, time: Tensor = None, clip_embed: Tensor = None, clip_scale: float = 1.0, return_attn: bool = False) -> Tensor:
        time_emb = self._get_time_emb(inputs=inputs, time=time, time_emb=None)

        down_attn: List[Tensor] = list()

        down_outputs: List[Tensor] = list()
        out = inputs
        for down_mod in self.down_stack:
            if isinstance(down_mod, SelfAttention):
                out = down_mod(out)
                down_attn.append(out)

            elif isinstance(down_mod, CrossAttention):
                out = down_mod.forward(out, clip_embed, clip_scale=clip_scale)
                down_attn.append(out)

            else:
                out = down_mod(out, time_emb)

            down_outputs.append(out)

        up_attn: List[Tensor] = list()
        up_weights: List[Tensor] = list()
        for up_mod in self.up_stack:
            if self.do_residual:
                last_down = down_outputs.pop()
                out = torch.cat([out, last_down], dim=1)
            
            if isinstance(up_mod, SelfAttention):
                out = up_mod(out)
                up_attn.append(out)

            elif isinstance(up_mod, CrossAttention):
                out = up_mod.forward(out, clip_embed, clip_scale=clip_scale)
                up_attn.append(out)

            else:
                out = up_mod(out, time_emb)

        if return_attn:
            return out, down_attn, up_attn
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
