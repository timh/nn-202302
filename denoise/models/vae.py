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

# TODO: make this output work through trainer
class VarEncoderOutput:
    mean: Tensor     # result from mean(out)
    logvar: Tensor   # result from logvar(out)

    def __init__(self, mean: Tensor, logvar: Tensor = None, std: Tensor = None):
        if logvar is None and std is None:
            raise ValueError("either logvar or std must be set")
        
        if logvar is None:
            logvar = torch.log(std) * 2.0
        
        self.mean = mean
        self.logvar = logvar
    
    def copy(self, mean: Tensor = None, logvar: Tensor = None, std: Tensor = None):
        if mean is None:
            mean = self.mean

        if std is not None:
            logvar = torch.log(std) * 2.0
        elif logvar is None:
            logvar = self.logvar
        
        return VarEncoderOutput(mean=mean, logvar=logvar)

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)

    @property    
    def kl_loss(self) -> Tensor:
        return torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mean ** 2 - self.logvar.exp(), dim=1), dim=0)
    
    def sample(self, std: Tensor = None, mean: Tensor = None, epsilon: Tensor = None) -> Tensor:
        epsilon = epsilon or torch.randn_like(self.std)
        mean = mean or self.mean
        std = std or self.std
        return mean + epsilon * std
    
    def detach(self) -> 'VarEncoderOutput':
        self.mean = self.mean.detach()
        self.logvar.detach_()
        return self

    def cpu(self) -> 'VarEncoderOutput':
        self.mean = self.mean.cpu()
        self.logvar = self.logvar.cpu()
        return self

    def len(self) -> int:
        return len(self.mean)
    
    """
    turn this (batched) VarEncoderOutput into one that is separated by its contained
    images.
    """
    def to_list(self) -> List['VarEncoderOutput']:
        if len(self.mean.shape) != 4:
            raise ValueError(f"can only call on (batch, chan, height, width): {self.mean.shape=}")

        res: List[VarEncoderOutput] = list()
        for mean, logvar in zip(self.mean, self.logvar):
            res.append(VarEncoderOutput(mean=mean, logvar=logvar))
        return res
    
    def __mul__(self, other: Union[Tensor, 'VarEncoderOutput']) -> 'VarEncoderOutput':
        if isinstance(other, VarEncoderOutput):
            other_mean = other.mean
            other_logvar = other.logvar
        else:
            other_mean = other
            other_logvar = other
        
        return VarEncoderOutput(mean=self.mean * other_mean, logvar=self.logvar * other_logvar)

    def __add__(self, other: Tensor) -> 'VarEncoderOutput':
        if isinstance(other, VarEncoderOutput):
            other_mean = other.mean
            other_logvar = other.logvar
        else:
            other_mean = other
            other_logvar = other
        
        return VarEncoderOutput(mean=self.mean + other_mean, logvar=self.logvar + other_logvar)


"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, emblen)
"""
class VarEncoderLinear(nn.Module):
    def __init__(self, *, in_size: int, emblen: int):
        super().__init__()

        self.linear = nn.Linear(in_size, emblen)
        self.mean = nn.Linear(emblen, emblen)
        self.logvar = nn.Linear(emblen, emblen)
        self.kld_loss = torch.tensor(0.0)
        self.out_size = emblen
        self.out_dim = [emblen]

    def forward(self, inputs: Tensor, return_veo: bool = False) -> Union[Tensor, VarEncoderOutput]:
        out = self.linear(inputs)
        mean = self.mean(out)
        logvar = self.logvar(out)

        veo = VarEncoderOutput(mean=mean, logvar=logvar)
        self.kld_loss = veo.kl_loss

        if return_veo:
            return veo
        return veo.sample()

    
class VarEncoderConv2d(nn.Module):
    def __init__(self, *, in_dim: List[int], kernel_size: int):
        super().__init__()

        padding = (kernel_size - 1) // 2
        in_chan = in_dim[0]

        self.conv_out = nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, padding=padding)
        self.conv_mean = nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, padding=padding)
        self.conv_logvar = nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, padding=padding)
        self.kld_loss = torch.tensor(0.0)
        self.out_dim = in_dim

    def forward(self, inputs: Tensor, return_veo: bool = False) -> Union[Tensor, VarEncoderOutput]:
        out = self.conv_out(inputs)
        mean = self.conv_mean(out)
        logvar = self.conv_logvar(out)

        veo = VarEncoderOutput(mean=mean, logvar=logvar)
        self.kld_loss = veo.kl_loss

        if return_veo:
            return veo
        return veo.sample()

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = "image_size nchannels emblen nlinear hidlen encoder_kernel_size do_residual".split()
class VarEncDec(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS + ['latent_dim']
    _model_fields = BASE_FIELDS

    encoder_conv: conv.DownStack
    encoder_flatten: nn.Flatten
    encoder: Union[VarEncoderLinear, VarEncoderConv2d]

    decoder_linear: nn.Module
    decoder_unflatten: nn.Unflatten
    decoder_conv: conv.UpStack

    do_residual: bool
    emblen: int
    nlinear: int
    hidlen: int

    def __init__(self, *,
                 image_size: int, nchannels = 3, 
                 emblen: int, nlinear: int = 0, hidlen: int = 0,
                 encoder_kernel_size: int,
                 do_residual: bool = False,
                 cfg: conv_types.ConvConfig):
        super().__init__()

        if emblen and encoder_kernel_size:
            raise ValueError(f"{emblen=} and {encoder_kernel_size=} can't both be set")

        # NOTE: enc.outsize =~ image_size // (2 ** len(layers))
        #       or, to be precise, cfg.get_sizes_down_actual(image_size)

        ######
        # encoder side
        ######

        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        self.encoder_conv = conv.DownStack(image_size=image_size, nchannels=nchannels, cfg=cfg)

        if emblen:
            #    (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
            # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
            self.encoder_flatten = nn.Flatten(start_dim=1, end_dim=-1)

            #    (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
            # -> (batch, emblen)
            flat_size = reduce(operator.mul, self.encoder_conv.out_dim, 1)
            self.encoder = VarEncoderLinear(in_size=flat_size, emblen=emblen)
            self.encoder_out_dim = [emblen]
        else:
            self.encoder_flatten = None
            self.encoder = VarEncoderConv2d(in_dim=self.encoder_conv.out_dim, kernel_size=encoder_kernel_size)
            self.encoder_out_dim = self.encoder_conv.out_dim
        
        self.latent_dim = self.encoder_out_dim

        ######
        # decoder side
        ######
        self.decoder_linear = nn.Sequential()
        if nlinear:
            if not emblen:
                raise NotImplementedError(f"emblen=0 but {nlinear=} {hidlen=}")
            for lidx in range(nlinear):
                feat_in = emblen if lidx == 0 else hidlen
                feat_out = hidlen if lidx < nlinear - 1 else emblen
                self.decoder_linear.append(nn.Linear(feat_in, feat_out))
                self.decoder_linear.append(cfg.create_inner_norm(out_shape=(feat_out,)))
                self.decoder_linear.append(cfg.create_linear_nl())
        
        if emblen or nlinear:
            self.decoder_linear.append(nn.Linear(emblen, flat_size))
            self.decoder_linear.append(cfg.create_inner_norm(out_shape=(flat_size,)))
            self.decoder_linear.append(cfg.create_linear_nl())
            self.decoder_unflatten = nn.Unflatten(dim=1, unflattened_size=self.encoder_conv.out_dim)
        else:
            self.decoder_linear = None
            self.decoder_unflatten = None

        self.decoder_conv = conv.UpStack(image_size=image_size, nchannels=nchannels, cfg=cfg)

        if do_residual:
            for layer in [*self.decoder_conv, *self.encoder_conv]:
                if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight.data, 1.0, 0.02)
                    nn.init.constant_(layer.bias.data, 0)

        # save hyperparameters
        self.image_size = image_size
        self.nchannels = nchannels
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.do_residual = do_residual
        self.encoder_kernel_size = encoder_kernel_size

        self.conv_cfg = cfg
        self.conv_cfg_metadata = cfg.metadata_dict()

        self.enc_conv_out = None

    """
          (batch, nchannels, image_size, image_size)
       -> (batch, emblen)
    """
    def encode(self, inputs: Tensor, return_veo: bool = False) -> Union[Tensor, VarEncoderOutput]:
        #    (batch, nchannels, image_size, image_size)
        # -> (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
        out = self.encoder_conv(inputs)
        self.enc_conv_out = out

        if self.encoder_flatten is not None:
            #    (batch, layers[-1].out_chan, enc.out_size, enc.out_size)
            # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
            flat_out = self.encoder_flatten(out)

            #    (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
            # -> (batch, emblen)
            out = self.encoder(flat_out, return_veo=return_veo)
        else:
            out = self.encoder(out, return_veo=return_veo)

        return out
    
    """
          (batch, emblen)
       -> (batch, nchannels, image_size, image_size)
    """
    def decode(self, inputs: Tensor) -> Tensor:
        #    (batch, emblen)
        # -> (batch, layers[-1].out_chan * enc.out_size * enc.out_size)
        out = inputs
        if self.decoder_linear is not None:
            out = self.decoder_linear(out)
        if self.decoder_unflatten is not None:
            out = self.decoder_unflatten(out)
        
        if self.do_residual and self.training:
            nlayers = len(self.decoder_conv)
            last_dec_shape = None
            for dec_idx, dec_layer in enumerate(self.decoder_conv):
                enc_in = self.encoder_conv.layer_ins[nlayers - dec_idx - 1]
                out = dec_layer(out)
                if enc_in.shape == out.shape:
                    if last_dec_shape != out.shape:
                        # out = (out + enc_in)/2
                        out = out + enc_in
                        # print(f"{dec_idx} {out.shape}")
                        last_dec_shape = out.shape
                # out = out + enc_in
        elif self.do_residual:
            for dec_layer in self.decoder_conv:
                out = dec_layer(out)
                out = out + out
        else:
            out = self.decoder_conv(out)
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

        use_weight = kld_weight
        if exp.nepochs < kld_warmup_epochs:
            use_weight = (kld_weight * (exp.nbatches + 1) / kld_warmup_batches)

        if net.training:
            # 'true' losses - those that don't pay attention to the kl stuff.
            if net.enc_conv_out is not None:
                dec_out_true = net.decoder_conv(net.enc_conv_out)
                backing_loss_true = backing_loss_fn(dec_out_true, truth)
            else:
                dec_out_true = None
                backing_loss_true = None

            # TODO: this definitely shouldn't be here.
            writer.add_scalars("batch/bl"       , {exp.label: backing_loss}        , global_step=exp.nbatches)
            writer.add_scalars("batch/kl"       , {exp.label: net.encoder.kld_loss}, global_step=exp.nbatches)
            writer.add_scalars("batch/kl_weight", {exp.label: use_weight}          , global_step=exp.nbatches)

            exp.lastepoch_kl_loss = net.encoder.kld_loss.item()
            exp.lastepoch_bl_loss = backing_loss.item()

            if backing_loss_true is not None:
                writer.add_scalars("batch/bl_true"  , {exp.label: backing_loss_true}   , global_step=exp.nbatches)
                exp.lastepoch_bl_loss_true = backing_loss_true.item()

        kld_loss = use_weight * net.encoder.kld_loss
        loss = kld_loss + backing_loss
        # print(f"backing_loss={backing_loss:.3f} + kld_weight={kld_weight:.1E} * kld_loss={net.encoder.kld_loss:.3f} = {loss:.3f}")
        return loss
    return fn