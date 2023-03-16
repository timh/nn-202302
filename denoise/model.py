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
# variational code is modified from
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
# and
# https://avandekleut.github.io/vae/
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

        if emblen:
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            in_size = descs[-1].channels * out_size * out_size

            self.linear = nn.Linear(in_size, emblen)
            in_size = emblen
        
            if do_variational:
                self.normal_dist = torch.distributions.Normal(0, 1)
                self.var_mean = nn.Linear(in_size, emblen)
                self.var_stddev = nn.Linear(in_size, emblen)
                self.kl_loss = 0.0

        elif do_variational:
            inchan = descs[-1].channels
            self.normal_dist = torch.distributions.Normal(0, 1)
            self.var_mean = nn.Conv2d(inchan, inchan, kernel_size=3, padding=1)
            self.var_stddev = nn.Conv2d(inchan, inchan, kernel_size=3, padding=1)
            self.kl_loss = 0.0

        self.out_size = out_size
        self.emblen = emblen
        self.do_variational = do_variational

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv_seq(inputs)
        if self.emblen:
            out = self.flatten(out)
            out = self.linear(out)
        
        if self.do_variational:
            mean = self.var_mean(out)
            log_stddev = self.var_stddev(out)
            stddev = torch.exp(log_stddev)

            # we sample from the standard normal a matrix of batch_size * 
            # latent_size (taking into account minibatches)
            self.normal_dist.loc = self.normal_dist.loc.to(inputs.device)
            self.normal_dist.scale = self.normal_dist.scale.to(inputs.device)

            # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
            latent = mean + stddev * self.normal_dist.sample(mean.shape)
            out = latent

            # x = torch.flatten(x, start_dim=1)
            # x = F.relu(self.linear1(x))
            # mu =  self.linear2(x)
            # sigma = torch.exp(self.linear3(x))
            # z = mu + sigma*self.N.sample(mu.shape)
            # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            # kl_loss = -0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)

            self.kl_loss = (stddev ** 2 + mean ** 2 - log_stddev - 1 / 2).mean()
            # self.kl_loss = -0.5 * (1 + log_stddev - mean**2 - stddev**2).sum()
            # print(f"kl_loss {self.kl_loss:.3f}")

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
                 use_bias: bool, norm_fn: Callable[[int, int], nn.Module]):
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
                print(f"up: {out_size=} {new_out_size=}")
                out_size = new_out_size
            print(f"up: {in_chan=} {out_chan=} {in_size=} {out_size=}")

            conv = nn.ConvTranspose2d(in_chan, out_chan, 
                                      kernel_size=d.kernel_size, stride=d.stride, 
                                      padding=padding, output_padding=output_padding)
            
            layer = nn.Sequential()
            layer.append(conv)
            layer.append(norm_fn(out_channels=out_chan, out_size=out_size))
            layer.append(nn.ReLU(True))
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
               "do_layernorm do_batchnorm use_bias").split()
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
                 use_bias = True, 
                 device = "cpu"):
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
                               descs=descs_rev, nchannels=nchannels)
        
        self.image_size = image_size
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.do_variational = do_variational
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.use_bias = use_bias
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

def get_kl_loss_fn(exp: Experiment, kl_weight: float, 
                   backing_loss_fn: Callable[[Tensor, Tensor], Tensor],
                   clamp_kl_loss = 100.0) -> Callable[[Tensor, Tensor], Tensor]:
    def fn(inputs: Tensor, truth: Tensor) -> Tensor:
        net: ConvEncDec = exp.net
        backing_loss = backing_loss_fn(inputs, truth)
        # TODO: rename to kld..
        kld_loss = kl_weight * net.encoder.kl_loss
        if clamp_kl_loss:
            kld_loss = min(clamp_kl_loss, kld_loss)
        loss = kld_loss + backing_loss
        # print(f"backing_loss={backing_loss:.3f} + kl_weight={kl_weight:.1E} * kl_loss={net.encoder.kl_loss:.3f} = {loss:.3f}")
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
