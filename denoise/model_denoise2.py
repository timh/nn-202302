from typing import List, Dict, Tuple, Callable
from functools import reduce
import operator
import math
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import base_model
from conv_types import ConvConfig, ConvLayer
import noised_data
import image_latents

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

class Up(nn.Module):
    conv: nn.ConvTranspose2d
    upsample: nn.MaxPool2d
    norm: nn.Module
    nonlinearity: nn.Module
    in_chan: int

    use_timestep: bool
    do_residual: bool

    def __init__(self, *,
                 in_chan: int, in_size: int, 
                 out_chan: int, out_size: int,
                 layer: ConvLayer, cfg: ConvConfig,
                 use_timestep: bool,
                 do_residual: bool):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                       kernel_size=layer.kernel_size, stride=layer.stride, 
                                       padding=layer.up_padding,
                                       output_padding=layer.up_output_padding)
        if layer.max_pool_kern:
            self.upsample = nn.Upsample(scale_factor=layer.max_pool_kern)
        else:
            self.upsample = None
        
        self.norm = cfg.create_inner_norm(out_shape=(out_chan, out_size, out_size))
        self.nonlinearity = cfg.create_inner_nl()

        if use_timestep:
            self.temb_linear = nn.Sequential(
                nn.Linear(in_features=in_chan, out_features=out_chan),
                cfg.create_linear_nl(),
                nn.Linear(in_features=out_chan, out_features=out_chan),
                cfg.create_linear_nl(),
            )

        self.in_chan, self.out_chan = in_chan, out_chan
        self.in_size, self.out_size = in_size, out_size
        self.use_timestep = use_timestep
        self.do_residual = do_residual
    
    def forward(self, latent_in: Tensor, timesteps: Tensor = None):
        if self.use_timestep:
            time_embed = get_timestep_embedding(timesteps=timesteps, emblen=self.in_chan)
            out = self.temb_linear(time_embed)[:, :, None, None]
        else:
            out = 0
        
        out = out + self.conv(latent_in)
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.norm(out)
        out = self.nonlinearity(out)

        if self.do_residual:
            return latent_in + out

        return out

class Down(nn.Module):
    conv: nn.Conv2d
    max_pool: nn.MaxPool2d
    norm: nn.Module
    nonlinearity: nn.Module
    in_chan: int

    use_timestep: bool
    do_residual: bool

    def __init__(self, *,
                 in_chan: int, in_size: int, 
                 out_chan: int, out_size: int,
                 layer: ConvLayer, cfg: ConvConfig,
                 use_timestep: bool,
                 do_residual: bool):
        super().__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, 
                              kernel_size=layer.kernel_size, stride=layer.stride, 
                              padding=layer.down_padding)
        if layer.max_pool_kern:
            self.max_pool = nn.MaxPool2d(kernel_size=layer.max_pool_kern)
        else:
            self.max_pool = None
        
        self.norm = cfg.create_inner_norm(out_shape=(out_chan, out_size, out_size))
        self.nonlinearity = cfg.create_inner_nl()

        if use_timestep:
            self.temb_linear = nn.Sequential(
                nn.Linear(in_features=in_chan, out_features=out_chan),
                cfg.create_linear_nl(),
                nn.Linear(in_features=out_chan, out_features=out_chan),
                cfg.create_linear_nl(),
            )

        self.in_chan, self.out_chan = in_chan, out_chan
        self.in_size, self.out_size = in_size, out_size
        self.use_timestep = use_timestep
        self.do_residual = do_residual
    
    def forward(self, latent_in: Tensor, timesteps: Tensor = None):
        if self.use_timestep:
            time_embed = get_timestep_embedding(timesteps=timesteps, emblen=self.in_chan)
            out = self.temb_linear(time_embed)[:, :, None, None]
        else:
            out = 0
        
        out = out + self.conv(latent_in)
        if self.max_pool is not None:
            out = self.max_pool(out)
        out = self.norm(out)
        out = self.nonlinearity(out)

        if self.do_residual:
            return latent_in + out

        return out

class DownStack(nn.Sequential):
    latent_dim: List[int]
    def __init__(self, *,
                 in_latent_dim: List[int], 
                 use_timestep: bool, 
                 do_residual: bool, 
                 cfg: ConvConfig):
        super().__init__()

        lat_chan, lat_size, _height = in_latent_dim
        channels = cfg.get_channels_down(lat_chan)
        sizes = cfg.get_sizes_down_actual(lat_size)

        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i : i + 2]
            in_size, out_size = sizes[i : i + 2]
            down = Down(in_chan=in_chan, in_size=in_size, 
                        out_chan=out_chan, out_size=out_size,
                        layer=layer, cfg=cfg, 
                        use_timestep=False, do_residual=do_residual)
            self.append(down)
            self.out_dim = [out_chan, out_size, out_size]
    
    def forward(self, latent_in: Tensor, timestep: Tensor = None) -> Tensor:
        out = latent_in
        for mod in self:
            out = mod(out, timestep)
        return out


class UpStack(nn.Sequential):
    def __init__(self, *,
                 latent_dim: List[int],
                 use_timestep: bool, 
                 do_residual: bool, 
                 cfg: ConvConfig):
        super().__init__()

        lat_chan, lat_size, _ = latent_dim
        channels = cfg.get_channels_up(lat_chan)
        down_out_size = cfg.get_sizes_down_actual(lat_size)[-1]
        sizes = cfg.get_sizes_up_actual(down_out_size)

        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i : i + 2]
            in_size, out_size = sizes[i : i + 2]
            up = Up(in_chan=in_chan, in_size=in_size, 
                    out_chan=out_chan, out_size=out_size,
                    layer=layer, cfg=cfg, 
                    use_timestep=use_timestep, do_residual=do_residual)
            self.append(up)
            self.out_dim = [out_chan, out_size, out_size]

    def forward(self, latent_in: Tensor, timestep: Tensor = None) -> Tensor:
        out = latent_in
        for mod in self:
            out = mod(out, timestep)
        return out


class DenoiseModel2(base_model.BaseModel):
    _metadata_fields = 'in_latent_dim use_timestep do_residual'.split(' ')
    _model_fields = _metadata_fields

    in_latent_dim: List[int]
    use_timestep: bool
    do_residual: bool
    conv_cfg: ConvConfig

    def __init__(self, *,
                 in_latent_dim: List[int], 
                 cfg: ConvConfig,
                 use_timestep: bool,
                 do_residual: bool):
        super().__init__()

        chan, size, _height = in_latent_dim

        if do_residual:
            raise NotImplemented("do_residual not implemented")

        self.downstack = DownStack(in_latent_dim=in_latent_dim,
                                   use_timestep=use_timestep,
                                   do_residual=do_residual,
                                   cfg=cfg)
        self.upstack = UpStack(latent_dim=in_latent_dim,
                               use_timestep=use_timestep,
                               do_residual=do_residual,
                               cfg=cfg)
        self.in_latent_dim = in_latent_dim
        self.use_timestep = use_timestep
        self.do_residual = do_residual
        self.conv_cfg = cfg
    
    def forward(self, latent_in: Tensor, timesteps: Tensor = None):
        out = self.downstack(latent_in, timesteps)
        out = self.upstack(out, timesteps)
        return out

    def metadata_dict(self) -> Dict[str, any]:
        res = super().metadata_dict()
        res.update(self.conv_cfg.metadata_dict())
        return res

    def model_dict(self, *args, **kwargs) -> Dict[str, any]:
        res = super().model_dict(*args, **kwargs)
        res.update(self.conv_cfg.metadata_dict())
        return res

# def get_dataloaders(*,
#                     vae_net: model_new.VarEncDec,
#                     vae_net_path: Path,
#                     src_train_dl: DataLoader,
#                     src_val_dl: DataLoader,
#                     batch_size: int,
#                     use_timestep: bool,
#                     noise_fn: Callable[[Tuple], Tensor],
#                     amount_fn: Callable[[], Tensor],
#                     device: str) -> Tuple[DataLoader, DataLoader]:
    
#     encds_args = dict(net=vae_net, net_path=vae_net_path, enc_batch_size=batch_size, device=device)
#     train_ds = image_latents.EncoderDataset(dataloader=src_train_dl, **encds_args)
#     val_ds = image_latents.EncoderDataset(dataloader=src_val_dl, **encds_args)

#     noiseds_args = dict(use_timestep=use_timestep, noise_fn=noise_fn, amount_fn=amount_fn)
#     train_ds = noised_data.NoisedDataset(base_dataset=train_ds, **noiseds_args)
#     val_ds = noised_data.NoisedDataset(base_dataset=val_ds, **noiseds_args)

#     train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)
#     val_dl = DataLoader(dataset=val_ds, shuffle=True, batch_size=batch_size)
    
#     return train_dl, val_dl
