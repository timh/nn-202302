from typing import List, Tuple, Callable
from functools import reduce
import operator
import math

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import base_model
from convolutions import DownStack, UpStack
from conv_types import ConvConfig
import model_new
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

class DenoiseModel(base_model.BaseModel):
    _metadata_fields = ['emblen', 'use_timestep']
    _model_fields = ['emblen', 'use_timestep']

    mid_in: nn.Sequential
    mid_time_embed: nn.Sequential
    mid_out: nn.Sequential

    def __init__(self, *,
                 in_latent_dim: List[int], 
                 cfg: ConvConfig,
                 emblen: int = 0,
                 use_timestep: bool = False):
        super().__init__()

        chan, size, _height = in_latent_dim
    
        self.downstack = DownStack(image_size=size, nchannels=chan, cfg=cfg)
        if use_timestep and not emblen:
            raise ValueError("use_timestep set, but emblen=0")

        if emblen:
            out_dim = self.downstack.out_dim
            out_flat = reduce(operator.mul, out_dim, 1)

            self.mid_in = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=out_flat, out_features=emblen),
                cfg.create_inner_norm(out_shape=(emblen,)),
                cfg.create_linear_nl(),
            )
            print("make mid_in:", self.mid_in)

            if use_timestep:
                self.mid_time_embed = nn.Sequential(
                    nn.Linear(in_features=emblen, out_features=emblen),
                    cfg.create_linear_nl(),
                    nn.Linear(in_features=emblen, out_features=emblen),
                )

            self.mid_out = nn.Sequential(
                nn.Linear(in_features=emblen, out_features=out_flat),
                cfg.create_inner_norm(out_shape=(out_flat,)),
                cfg.create_linear_nl(),
                nn.Unflatten(dim=1, unflattened_size=out_dim)
            )

        self.upstack = UpStack(image_size=size, nchannels=chan, cfg=cfg)

        self.in_latent_dim = in_latent_dim
        self.emblen = emblen
        self.use_timestep = use_timestep
    
    def forward(self, latent_in: Tensor, timesteps: Tensor = None):
        out = self.downstack(latent_in)
        if self.emblen:
            out = self.mid_in(out)
            if self.use_timestep:
                time_embed = get_timestep_embedding(timesteps=timesteps, emblen=self.emblen)
                time_out = self.mid_time_emb(time_embed)
                out = out @ time_out.T
            out = self.mid_out(out)

        out = self.upstack(out)
        return out

def get_dataloaders(vae_net: model_new.VarEncDec,
                    src_train_dl: DataLoader,
                    src_val_dl: DataLoader,
                    batch_size: int,
                    use_timestep: bool,
                    noise_fn: Callable[[Tuple], Tensor],
                    amount_fn: Callable[[], Tensor],
                    device: str) -> Tuple[DataLoader, DataLoader]:
    
    encds_args = dict(net=vae_net, enc_batch_size=batch_size, device=device)
    train_ds = image_latents.EncoderDataset(dataloader=src_train_dl, **encds_args)
    val_ds = image_latents.EncoderDataset(dataloader=src_val_dl, **encds_args)

    noiseds_args = dict(use_timestep=use_timestep, noise_fn=noise_fn, amount_fn=amount_fn)
    train_ds = noised_data.NoisedDataset(base_dataset=train_ds, **noiseds_args)
    val_ds = noised_data.NoisedDataset(base_dataset=val_ds, **noiseds_args)

    train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, batch_size=batch_size)
    
    return train_dl, val_dl
