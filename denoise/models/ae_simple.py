# %%
import sys
from typing import Dict, List

import torch
from torch import Tensor
from torch import nn

sys.path.append("..")
import conv_types
import convolutions as conv
import base_model
from . import denoise

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = "image_size nchannels".split()
class Autoencoder(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS + ['latent_dim']
    _model_fields = BASE_FIELDS

    latent_dim: List[int]
    encoder: conv.DownStack
    decoder: conv.UpStack

    def __init__(self, *,
                 image_size: int, nchannels = 3, 
                 cfg: conv_types.ConvConfig):
        super().__init__()

        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        self.encoder = conv.DownStack(image_size=image_size, nchannels=nchannels, cfg=cfg)
        self.latent_dim = self.encoder.out_dim

        self.decoder = conv.UpStack(image_size=image_size, nchannels=nchannels, cfg=cfg)

        # save hyperparameters
        self.image_size = image_size
        self.nchannels = nchannels
        self.conv_cfg = cfg

    """
          (batch, nchannels, image_size, image_size)
       -> (batch, emblen)
    """
    def encode(self, inputs: Tensor) -> Tensor:
        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        return self.encoder(inputs)
    
    """
          (batch, emblen)
       -> (batch, nchannels, image_size, image_size)
    """
    def decode(self, inputs: Tensor) -> Tensor:
        #    (batch, emblen)
        # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        return self.decoder(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        enc_out = self.encode(inputs)
        dec_out = self.decode(enc_out)
        return dec_out
    
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

class AEDenoise(Autoencoder):
    def __init__(self, *, image_size: int, nchannels=3, cfg: conv_types.ConvConfig):
        super().__init__(image_size=image_size, nchannels=nchannels, cfg=cfg)

        out_chan, _out_size, _ = self.latent_dim
        self.time_linear = nn.Sequential(
            nn.Linear(in_features=out_chan, out_features=out_chan),
            cfg.create_linear_nl(),
            nn.Linear(in_features=out_chan, out_features=out_chan),
            cfg.create_linear_nl(),
        )

    def decode(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        if time is None:
            time = torch.zeros((inputs.shape[0],), device=inputs.device)

        _batch, chan, _height, _width = inputs.shape
        time_embed = denoise.get_timestep_embedding(timesteps=time, emblen=chan)
        time_embed_out = self.time_linear(time_embed)[:, :, None, None]

        out = inputs + time_embed_out
        out = super().decode(out)

        return out

    def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
        enc_out = self.encode(inputs)
        dec_out = self.decode(enc_out, time=time)
        return dec_out
    
