# %%
import sys
from typing import List, Union, Tuple, Callable, Dict, Literal
from dataclasses import dataclass
from functools import reduce
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import conv_types

# TODO: make top level a package
# sys.path.append("..")
# from .. import base_model
import base_model
from experiment import Experiment

# contributing sites:
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
#
# variational code is influenced by/modified from
#   https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
#   https://avandekleut.github.io/vae/
#   https://github.com/pytorch/examples/blob/main/vae/main.py

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, emblen)
"""
class Encoder(nn.Module):
    conv_seq: nn.Sequential

    """side dimension of image after convolutions are processed"""
    out_size: int

    def __init__(self, *, image_size: int, nchannels: int, 
                 cfg: conv_types.ConvConfig,
                 emblen: int, do_variational: bool):
                 
        super().__init__()

        channels = cfg.get_channels_down(nchannels)
        sizes = cfg.get_sizes_down_actual(image_size)

        self.conv_seq = nn.Sequential()
        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i:i + 2]
            in_size, out_size = sizes[i:i + 2]

            conv = nn.Conv2d(in_chan, out_chan, 
                             kernel_size=layer.kernel_size, stride=layer.stride, 
                             padding=layer.down_padding)
            print(f"down: add {in_chan=} {out_chan=} {in_size=} {out_size=}")

            layer = nn.Sequential()
            layer.append(conv)
            layer.append(cfg.create_norm(out_chan=out_chan, out_size=out_size))
            layer.append(cfg.create_inner_nl())
            self.conv_seq.append(nn.Sequential(layer))

        flat_size = channels[-1] * sizes[-1] * sizes[-1]
        if emblen:
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.linear = nn.Linear(flat_size, emblen)
        
            if do_variational:
                self.mean = nn.Linear(emblen, emblen)
                self.logvar = nn.Linear(emblen, emblen)
                self.kld_loss = 0.0

        elif do_variational:
            inchan = channels[-1]
            self.mean = nn.Conv2d(inchan, inchan, kernel_size=3, padding=1)
            self.logvar = nn.Conv2d(inchan, inchan, kernel_size=3, padding=1)
            self.kld_loss = 0.0

        # if do_variational:
        #     val = 1e-5
        #     nn.init.normal_(self.mean.weight.data, 0.0, val)
        #     nn.init.normal_(self.mean.bias.data, 0.0, val)
        #     nn.init.normal_(self.logvar.weight.data, 0.0, val)
        #     nn.init.normal_(self.logvar.bias.data, 0.0, val)

        self.out_size = out_size
        self.emblen = emblen
        self.do_variational = do_variational

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv_seq(inputs)
        if self.emblen:
            out = self.flatten(out)
            out = self.linear(out)
        
        if self.do_variational:
            mean = self.mean(out)
            logvar = self.logvar(out)

            # reparameterize
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            out = mean + epsilon * std

            self.kld_loss = -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp())

        return out

    def extra_repr(self) -> str:
        extras: List[str] = []
        for k in "image_size out_size emblen nchannels use_bias".split(" "):
            extras.append(f"{k}={getattr(self, k, None)}")
        return ", ".join(extras)

"""
inputs: (batch, emblen)
return: (batch, nchannels, image_size, image_size)
"""
class Decoder(nn.Module):
    conv_seq: nn.Sequential

    def __init__(self, *, image_size: int, nchannels: int, 
                 cfg: conv_types.ConvConfig,
                 encoder_out_size: int, emblen: int):
        super().__init__()

        channels = cfg.get_channels_up(nchannels)
        sizes = cfg.get_sizes_up_actual(image_size)

        if emblen:
            in_chan = channels[0]
            self.linear = nn.Sequential(
                nn.Linear(emblen, in_chan * encoder_out_size * encoder_out_size),
                cfg,
                nn.ReLU(True)
            )
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(in_chan, encoder_out_size, encoder_out_size))

        self.conv_seq = nn.Sequential()
        for i, layer in enumerate(cfg.layers):
            in_chan, out_chan = channels[i:i + 2]
            in_size, out_size = sizes[i:i + 2]

            # if out_size < desired_out_size:
            #     output_padding = 1
            #     new_out_size = d.get_up_size(in_size=in_size, padding=padding, output_padding=output_padding)
            #     print(f"up: {output_padding=}: {out_size=} {new_out_size=}")
            #     out_size = new_out_size

            print(f"up: {in_chan=} {out_chan=} {in_size=} {out_size=}")

            conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                      kernel_size=layer.kernel_size, stride=layer.stride, 
                                      padding=layer.up_padding, output_padding=layer.up_output_padding)
            
            layer = nn.Sequential()
            layer.append(conv)
            layer.append(cfg.create_norm(out_chan=out_chan, out_size=out_size))
            if i < len(cfg.layers) - 1:
                layer.append(cfg.create_inner_nl())
            else:
                layer.append(cfg.create_final_nl())

            self.conv_seq.append(layer)
            self.out_size = out_size

        self.image_size = image_size
        self.encoder_out_size = encoder_out_size
        self.emblen = emblen

    def forward(self, inputs: Tensor) -> Tensor:
        if self.emblen:
            out = self.linear(inputs)
            out = self.unflatten(out)
        else:
            out = inputs
        out = self.conv_seq(out)

        return out

    def extra_repr(self) -> str:
        extras: List[str] = []
        for k in "image_size encoder_out_size emblen nchannels use_bias".split(" "):
            extras.append(f"{k}={getattr(self, k, None)}")
        return ", ".join(extras)

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = ("image_size nchannels conv_cfg "
               "emblen nlinear hidlen do_variational").split()
class ConvEncDec(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS + ['latent_dim', 'conv_cfg_metadata']
    _model_fields = BASE_FIELDS + ['conv_cfg']

    encoder: Encoder
    linear_layers: nn.Sequential
    decoder: Decoder

    def __init__(self, *,
                 image_size: int, nchannels = 3, 
                 emblen: int, nlinear: int = 0, hidlen: int = 0,
                 do_variational: bool,
                 cfg: conv_types.ConvConfig):
        super().__init__()

        self.encoder = Encoder(image_size=image_size, nchannels=nchannels, cfg=cfg,
                               emblen=emblen, do_variational=do_variational)
        if emblen:
            self.linear_layers = nn.Sequential()
            for i in range(nlinear):
                in_features = emblen if i == 0 else hidlen
                out_features = hidlen if i < nlinear - 1 else emblen
                self.linear_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
                self.linear_layers.append(nn.ReLU(True))

        out_size = self.encoder.out_size
        self.decoder = Decoder(image_size=image_size, nchannels=nchannels, cfg=cfg,
                               emblen=emblen, encoder_out_size=out_size)
        
        self.image_size = image_size
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.do_variational = do_variational
        if emblen == 0:
            self.latent_dim = [cfg.layers[-1].out_chan, out_size, out_size]
        else:
            self.latent_dim = [emblen]
        self.nchannels = nchannels
        self.conv_cfg = cfg
        self.conv_cfg_metadata = cfg.metadata_dict()
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.encoder(inputs)
        if self.emblen:
            out = self.linear_layers(out)
        out = self.decoder(out)
        return out

    def extra_repr(self) -> str:
        extras: List[str] = []
        for field in self._metadata_fields:
            val = getattr(self, field, None)
            extras.append(f"{field}={val}")
        return ", ".join(extras)


def get_kld_loss_fn(exp: Experiment, kld_weight: float, 
                    backing_loss_fn: Callable[[Tensor, Tensor], Tensor],
                    kld_warmup_epochs: int = 0, clamp_kld_loss = 100.0) -> Callable[[Tensor, Tensor], Tensor]:
    def fn(inputs: Tensor, truth: Tensor) -> Tensor:
        net: ConvEncDec = exp.net
        backing_loss = backing_loss_fn(inputs, truth)

        use_weight = kld_weight
        if kld_warmup_epochs and exp.nepochs < kld_warmup_epochs:
            # use_weight = 1.0 / torch.exp(torch.tensor(kld_warmup_epochs - exp.nepochs - 1)) * kld_weight
            # use_weight = torch.lerp(0.0, kld_weight, (exp.nepochs + 1) / kld_warmup_epochs)
            use_weight = kld_weight * (exp.nepochs + 1) / kld_warmup_epochs
            # print(f"warmup: use_weight = {use_weight:.2E}")

        kld_loss = use_weight * net.encoder.kld_loss
        loss = kld_loss + backing_loss
        # print(f"backing_loss={backing_loss:.3f} + kld_weight={kld_weight:.1E} * kld_loss={net.encoder.kld_loss:.3f} = {loss:.3f}")
        return loss
    return fn

