# %%
import sys
from typing import List, Union, Tuple, Callable, Dict, Literal
from dataclasses import dataclass
from functools import reduce
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

sys.path.append("..")
import base_model
from experiment import Experiment
import trainer

# contributing sites:
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
#
# variational code is influenced by/modified from
#   https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
#   https://avandekleut.github.io/vae/
#   https://github.com/pytorch/examples/blob/main/vae/main.py
@dataclass
class ConvDesc:
    channels: int
    kernel_size: int
    stride: int
    max_pool_kern: int = 0

    def get_down_size(self, in_size: int, padding: int) -> int:
        if self.max_pool_kern:
            return in_size // self.max_pool_kern

        out_size = (in_size + 2 * padding - self.kernel_size) // self.stride + 1
        return out_size

    def get_up_size(self, in_size: int, padding: int, output_padding = 0) -> int:
        if self.max_pool_kern:
            return in_size * self.max_pool_kern

        out_size = (in_size - 1) * self.stride - 2 * padding + self.kernel_size + output_padding
        return out_size

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, emblen)
"""
class Encoder(nn.Module):
    conv_seq: nn.Sequential

    """side dimension of image after convolutions are processed"""
    out_size: int

    def __init__(self, *, image_size: int, nchannels: int, 
                 emblen: int, descs: List[ConvDesc], do_variational: bool,
                 use_bias: bool, norm_fn: Callable[[int, int], nn.Module]):
                 
        super().__init__()

        channels = [nchannels] + [d.channels for d in descs]
        in_size = image_size
        self.conv_seq = nn.Sequential()
        for i, d in enumerate(descs):
            in_chan, out_chan = channels[i:i+2]
            padding = (d.kernel_size - 1) // 2
            out_size = d.get_down_size(in_size, padding=padding)

            conv = nn.Conv2d(in_chan, out_chan, 
                             kernel_size=d.kernel_size, stride=d.stride, 
                             padding=padding)
            print(f"down: add {in_chan=} {out_chan=} {in_size=} {out_size=}")

            layer = nn.Sequential()
            layer.append(conv)
            layer.append(norm_fn(out_channels=out_chan, out_size=out_size))
            layer.append(nn.ReLU(True))
            self.conv_seq.append(nn.Sequential(layer))

            in_size = out_size

        flat_size = descs[-1].channels * out_size * out_size
        if emblen:
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.linear = nn.Linear(flat_size, emblen)
        
            if do_variational:
                self.mean = nn.Linear(emblen, emblen)
                self.logvar = nn.Linear(emblen, emblen)
                self.kld_loss = 0.0

        elif do_variational:
            inchan = descs[-1].channels
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

    def __init__(self, *, image_size: int, nchannels: int, descs: List[ConvDesc],
                 encoder_out_size: int, emblen: int, 
                 use_bias: bool, 
                 norm_fn: Callable[[int, int], nn.Module], 
                 last_nonlinearity: Literal['relu', 'sigmoid']):
        super().__init__()

        firstchan = descs[0].channels
        if emblen:
            self.linear = nn.Sequential(
                nn.Linear(emblen, firstchan * encoder_out_size * encoder_out_size),
                nn.ReLU(True)
            )
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(firstchan, encoder_out_size, encoder_out_size))

        channels = [d.channels for d in descs] + [nchannels]
        in_size = encoder_out_size
        self.conv_seq = nn.Sequential()
        for i, d in enumerate(descs):
            in_chan, out_chan = channels[i:i + 2]
            padding = (d.kernel_size - 1) // 2
            output_padding = 0
            out_size = d.get_up_size(in_size, padding=padding, output_padding=output_padding)
            desired_out_size = in_size * d.stride

            if out_size < desired_out_size:
                output_padding = 1
                new_out_size = d.get_up_size(in_size=in_size, padding=padding, output_padding=output_padding)
                print(f"up: {output_padding=}: {out_size=} {new_out_size=}")
                out_size = new_out_size

            print(f"up: {in_chan=} {out_chan=} {in_size=} {out_size=}")

            conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                      kernel_size=d.kernel_size, stride=d.stride, 
                                      padding=padding, output_padding=output_padding)
            
            layer = nn.Sequential()
            layer.append(conv)
            layer.append(norm_fn(out_channels=out_chan, out_size=out_size))
            if i < len(descs) - 1 or last_nonlinearity == 'relu':
                layer.append(nn.ReLU(True))
            elif last_nonlinearity == 'sigmoid':
                layer.append(nn.Sigmoid())
            else:
                raise ValueError(f"unknown {last_nonlinearity=}")

            self.conv_seq.append(layer)

            in_size = out_size
            self.out_size = out_size

        self.image_size = image_size
        self.encoder_out_size = encoder_out_size
        self.emblen = emblen
        self.use_bias = use_bias
        self.descs = descs

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

def get_norm_fn(do_batchnorm: bool, do_layernorm: bool) -> Callable[[int, int], nn.Module]:
    def layer_norm(out_channels: int, out_size: int) -> nn.Module:
        return nn.LayerNorm((out_channels, out_size, out_size))

    def batch_norm(out_channels: int, out_size: int) -> nn.Module:
        return nn.BatchNorm2d(out_channels)

    def identity(out_channels: int, out_size: int) -> nn.Module:
        return nn.Identity()
    
    if do_batchnorm:
        return batch_norm
    
    if do_layernorm:
        return layer_norm
    
    return identity

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
BASE_FIELDS = ("image_size nchannels "
               "emblen nlinear hidlen do_variational "
               "do_layernorm do_batchnorm use_bias decoder_last_nonlinearity").split()
class ConvEncDec(base_model.BaseModel):
    _metadata_fields = BASE_FIELDS + ["latent_dim"]
    _model_fields = BASE_FIELDS + ["descs"]

    encoder: Encoder
    linear_layers: nn.Sequential
    decoder: Decoder

    def __init__(self, *,
                 image_size: int, nchannels = 3, 
                 emblen: int, nlinear: int, hidlen: int, 
                 do_variational: bool,
                 descs: List[ConvDesc], 
                 do_layernorm: bool = True, do_batchnorm: bool = False, 
                 use_bias = True, decoder_last_nonlinearity: Literal['relu', 'sigmoid'] = 'relu'):
        super().__init__()

        norm_fn = get_norm_fn(do_batchnorm=do_batchnorm, do_layernorm=do_layernorm)
        self.encoder = Encoder(image_size=image_size, emblen=emblen, do_variational=do_variational,
                               use_bias=use_bias, norm_fn=norm_fn,
                               descs=descs, nchannels=nchannels)
        if emblen:
            self.linear_layers = nn.Sequential()
            for i in range(nlinear):
                in_features = emblen if i == 0 else hidlen
                out_features = hidlen if i < nlinear - 1 else emblen
                self.linear_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
                self.linear_layers.append(nn.ReLU(True))

        out_size = self.encoder.out_size
        descs_rev = list(reversed(descs))
        self.decoder = Decoder(image_size=image_size, encoder_out_size=out_size, emblen=emblen, 
                               use_bias=use_bias, norm_fn=norm_fn,
                               descs=descs_rev, nchannels=nchannels, last_nonlinearity=decoder_last_nonlinearity)
        
        self.image_size = image_size
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.do_variational = do_variational
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.use_bias = use_bias
        self.decoder_last_nonlinearity = decoder_last_nonlinearity
        self.descs = descs
        if emblen == 0:
            self.latent_dim = [descs[-1].channels, out_size, out_size]
        else:
            self.latent_dim = [emblen]
        self.nchannels = nchannels
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.encoder(inputs)
        if self.emblen:
            out = self.linear_layers(out)
        out = self.decoder(out)
        return out

    def extra_repr(self) -> str:
        extras: List[str] = []
        for k in "image_size emblen nlinear hidlen nchannels".split(" "):
            extras.append(f"{k}={getattr(self, k, None)}")
        return ", ".join(extras)

"""generate a list of ConvDescs from a string like the following:

input: "k5-s2-c16,c32,s1-c16"

where
    k = kernel_size
    s = stride
    c = channels

would return a list of 3 ConvDesc's:
    [ConvDesc(kernel_size=5, stride=2, channels=16),
     ConvDesc(kernel_size=5, stride=2, channels=32), 
     ConvDesc(kernel_size=5, stride=1, channels=16), 

This returns a ConvDesc for each comma-separated substring.

Each ConvDesc *must* have a (c)hannel set, but the (k)ernel_size and (s)adding
will carry on from block to block.
"""
def gen_descs(s: str) -> List[ConvDesc]:
    kernel_size = 0
    stride = 0

    descs: List[ConvDesc] = list()
    for onedesc_str in s.split(","):
        channels = 0
        max_pool_kern = 0
        for part in onedesc_str.split("-"):
            if part.startswith("c"):
                if channels:
                    raise ValueError(f"{channels=} already set: {onedesc_str=}")
                channels = int(part[1:])
            elif part.startswith("k"):
                kernel_size = int(part[1:])
            elif part.startswith("s"):
                stride = int(part[1:])
            elif part.startswith("mp"):
                max_pool_kern = int(part[2:])
            else:
                raise Exception(f"dunno what to do with {part=} for {s=}")

        if not kernel_size or not stride or not channels:
            raise ValueError(f"{kernel_size=} {stride=} {channels=}")

        onedesc = ConvDesc(channels=channels, kernel_size=kernel_size, stride=stride, max_pool_kern=max_pool_kern)
        descs.append(onedesc)
    
    return descs

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

if __name__ == "__main__":
    sz = 128
    emblen = 32
    descs = gen_descs("k4-s2-c64,c32,c8")

    net = ConvEncDec(image_size=sz, emblen=emblen, nlinear=0, hidlen=0, descs=descs)

    print(f"{emblen=}")
    print(f"{net.latent_dim=}")

    inputs = torch.zeros((1, 3, sz, sz))
    out = net.encoder(inputs)
    print(f"{out.shape=}")

    optim = torch.optim.SGD(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, 1)
    exp = Experiment(label='foo', net=net, sched=sched, optim=optim)
    sd = exp.model_dict()
    print("sd:")
    print(sd['net']['descs'])

    import model_util
    from pathlib import Path
    model_util.save_ckpt_and_metadata(exp, Path("foo.ckpt"), Path("foo.json"))

    with open("foo.ckpt", "rb") as cp_file:
        sdload = torch.load(cp_file)
        print("sdload:")
        print(sdload['net']['descs'])
    
    import dn_util
    net = dn_util.load_model(sdload)
    print("net:")
    print(net.descs)
