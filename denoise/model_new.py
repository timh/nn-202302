# %%
import sys
from typing import List, Union, Tuple, Callable, Dict, Literal
from dataclasses import dataclass
from functools import reduce
import operator
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# from .. import conv_types

# TODO: make top level a package
sys.path.append("..")
import conv_types
import convolutions as conv

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
class VarEncoder(nn.Module):
    def __init__(self, *, in_size: int, emblen: int):
        super().__init__()

        self.linear = nn.Linear(in_size, emblen)
        self.mean = nn.Linear(emblen, emblen)
        self.logvar = nn.Linear(emblen, emblen)
        self.kld_loss = torch.tensor(0.0)

        self.out_size = emblen
        self.out_dim = [emblen]

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.linear(inputs)
        mean = self.mean(out)
        logvar = self.logvar(out)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        out = mean + epsilon * std

        self.kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0)

        return out

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = ("image_size nchannels emblen nlinear hidlen").split()
class VarEncDec(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS # + ['class']
    _model_fields = BASE_FIELDS # + ['class'] # + ['conv_cfg']

    encoder_conv: conv.DownStack
    encoder_flatten: nn.Flatten
    encoder: VarEncoder

    decoder_linear: nn.Module
    decoder_unflatten: nn.Unflatten
    decoder_conv: conv.UpStack

    def __init__(self, *,
                 image_size: int, nchannels = 3, 
                 emblen: int, nlinear: int = 0, hidlen: int = 0,
                 cfg: conv_types.ConvConfig):
        super().__init__()

        # NOTE: enc.outsize =~ image_size // (2 ** len(layers))
        #       or, to be precise, cfg.get_sizes_down_actual(image_size)

        ######
        # encoder side
        ######

        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        self.encoder_conv = conv.DownStack(image_size=image_size, nchannels=nchannels, cfg=cfg)

        #    (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        self.encoder_flatten = nn.Flatten(start_dim=1, end_dim=-1)

        #    (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        # -> (batch, emblen)
        flat_size = reduce(operator.mul, self.encoder_conv.out_dim, 1)
        self.encoder = VarEncoder(in_size=flat_size, emblen=emblen)


        ######
        # decoder side
        ######
        self.decoder_linear = nn.Sequential()
        if nlinear:
            for lidx in range(nlinear):
                feat_in = emblen if lidx == 0 else hidlen
                feat_out = hidlen if lidx < nlinear - 1 else emblen
                self.decoder_linear.append(nn.Linear(feat_in, feat_out))
                self.decoder_linear.append(cfg.create_inner_norm(out_shape=(feat_out,)))
                self.decoder_linear.append(cfg.create_linear_nl())
            
        self.decoder_linear.append(nn.Linear(emblen, flat_size))
        self.decoder_linear.append(cfg.create_inner_norm(out_shape=(flat_size,)))
        self.decoder_linear.append(cfg.create_linear_nl())

        self.decoder_unflatten = nn.Unflatten(dim=1, unflattened_size=self.encoder_conv.out_dim)
        self.decoder_conv = conv.UpStack(image_size=image_size, nchannels=nchannels, cfg=cfg)

        # save hyperparameters
        self.image_size = image_size
        self.nchannels = nchannels
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen

        self.conv_cfg = cfg
        self.conv_cfg_metadata = cfg.metadata_dict()

    """
          (batch, nchannels, image_size, image_size)
       -> (batch, emblen)
    """
    def encode(self, inputs: Tensor) -> Tensor:
        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        conv_out = self.encoder_conv(inputs)
        self.enc_conv_out = conv_out

        #    (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        flat_out = self.encoder_flatten(conv_out)

        #    (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        # -> (batch, emblen)
        out = self.encoder(flat_out)

        return out
    
    """
          (batch, emblen)
       -> (batch, nchannels, image_size, image_size)
    """
    def decode(self, inputs: Tensor) -> Tensor:
        #    (batch, emblen)
        # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        lin_out = self.decoder_linear(inputs)
        unflat_out = self.decoder_unflatten(lin_out)
        out = self.decoder_conv(unflat_out)
        return out

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

def get_kld_loss_fn(exp: Experiment, kld_weight: float, 
                    backing_loss_fn: Callable[[Tensor, Tensor], Tensor],
                    dirname: str,
                    kld_warmup_epochs: int = 0, 
                    clamp_kld_loss = 100.0) -> Callable[[Tensor, Tensor], Tensor]:
    import torch.utils.tensorboard as tboard
    writer = tboard.SummaryWriter(log_dir=dirname)

    def fn(net_out: Tensor, truth: Tensor) -> Tensor:
        net: VarEncDec = exp.net
        backing_loss = backing_loss_fn(net_out, truth)

        kld_warmup_batches = len(exp.train_dataloader) * kld_warmup_epochs

        # 'true' losses - those that don't pay attention to the kl stuff.
        dec_out_true = net.decoder_conv(net.enc_conv_out)
        backing_loss_true = backing_loss_fn(dec_out_true, truth)

        use_weight = kld_weight
        if exp.nepochs < kld_warmup_epochs:
            use_weight = (kld_weight * (exp.nbatches + 1) / kld_warmup_batches)

        writer.add_scalars("batch/bl"       , {exp.label: backing_loss}        , global_step=exp.nbatches)
        writer.add_scalars("batch/kl"       , {exp.label: net.encoder.kld_loss}, global_step=exp.nbatches)
        writer.add_scalars("batch/kl_weight", {exp.label: use_weight}          , global_step=exp.nbatches)
        writer.add_scalars("batch/bl_true"  , {exp.label: backing_loss_true}   , global_step=exp.nbatches)

        exp.lastepoch_kl_loss = net.encoder.kld_loss.item()
        exp.lastepoch_bl_loss = backing_loss.item()
        exp.lastepoch_bl_loss_true = backing_loss_true.item()

        kld_loss = use_weight * net.encoder.kld_loss
        loss = kld_loss + backing_loss
        # print(f"backing_loss={backing_loss:.3f} + kld_weight={kld_weight:.1E} * kld_loss={net.encoder.kld_loss:.3f} = {loss:.3f}")
        return loss
    return fn
