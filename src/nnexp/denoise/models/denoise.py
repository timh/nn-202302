from typing import List, Dict

import torch
from torch import nn, Tensor
import einops

from nnexp import base_model
from nnexp.images import conv_types
from .model_shared import SinPositionEmbedding, SelfAttention, CrossAttention

# TODO: fix the unprocessed borders around output - probably need to run @ higher res?

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


class DenoiseModel(base_model.BaseModel):
    _model_fields = 'in_size in_chan do_residual clip_emblen clip_scale_default'.split(' ')
    _metadata_fields = _model_fields + ['in_dim', 'latent_dim']

    in_dim: List[int]
    latent_dim: List[int]
    conv_cfg: conv_types.ConvConfig
    do_residual: bool
    clip_emblen: int
    clip_scale_default: float

    def __init__(self, *,
                 in_chan: int, in_size: int,
                 cfg: conv_types.ConvConfig,
                 do_residual: bool = False,
                 clip_emblen: int = None,
                 clip_scale_default: float = 1.0):
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
        self.clip_scale_default = clip_scale_default
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

    def forward(self, inputs: Tensor, time: Tensor = None, clip_embed: Tensor = None, clip_scale: float = None, return_attn: bool = False) -> Tensor:
        clip_scale = clip_scale or self.clip_scale_default
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
