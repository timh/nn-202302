# %%
import sys
from typing import List, Union, Tuple, Callable, Dict, Literal
from dataclasses import dataclass
import math

import torch
from torch import nn, Tensor

sys.path.append("..")
from experiment import Experiment
import trainer

ENCODER_FIELDS = "image_size out_size emblen do_layernorm do_batchnorm use_bias descs nchannels".split(" ")
DECODER_FIELDS = "image_size encoder_out_size emblen do_layernorm do_batchnorm use_bias descs".split(" ")
ENCDEC_FIELDS = "image_size emblen nlinear hidlen do_layernorm do_batchnorm use_bias descs nchannels".split(" ")

# contributing sites:
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
@dataclass
class ConvDesc:
    channels: int
    kernel_size: int
    stride: int
    max_pool_kern: int = 0
    keep_size: bool = False

def get_out_size_conv2d(in_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    return out_size

def get_out_size_convtrans2d(in_size: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0):
    out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding
    return out_size

def _gen_conv_layer(in_size: int, desired_out_size: int, 
                    in_channels: int, out_channels: int, kernel_size: int, stride: int,
                    do_layernorm: bool, do_batchnorm: bool, use_bias: bool,
                    direction: Literal["up", "down", "keep"]) -> List[nn.Module]:
    res: List[nn.Module] = list()

    padding = (kernel_size - 1) // 2
    args = dict(kernel_size=kernel_size, stride=stride, padding=padding)
    if direction in ["down", "keep"]:
        out_size = get_out_size_conv2d(in_size=in_size, **args)
        conv = nn.Conv2d(in_channels, out_channels, bias=use_bias, **args)
    else:
        out_size = get_out_size_convtrans2d(in_size=in_size, **args)
        conv = nn.ConvTranspose2d(in_channels, out_channels, bias=use_bias, **args)

    res.append(conv)

    if out_size != desired_out_size:
        if out_size < desired_out_size:
            diff_size = (desired_out_size - out_size)
            topleft, botright = diff_size // 2, diff_size // 2
            if diff_size % 2 == 1:
                botright += 1
            res.append(nn.ConstantPad2d((topleft, botright, topleft, botright), value=0.5))
        else:
            raise ValueError(f"don't know how to deal with {out_size=} > {desired_out_size=}")
        out_size = desired_out_size

    if do_layernorm:
        res.append(nn.LayerNorm((out_channels, out_size, out_size)))
    if do_batchnorm:
        res.append(nn.BatchNorm2d(out_channels))
    res.append(nn.ReLU(True))

    return res

def _gen_conv_layers(in_size: int,
                     direction: Literal["up", "down"],
                     do_layernorm: bool, do_batchnorm: bool, use_bias: bool,
                     descs: List[ConvDesc], nchannels = 3) -> Tuple[List[nn.Module], int]:
    res: List[nn.Module] = list()

    if direction == "down":
        channels = [nchannels] + [d.channels for d in descs]
    else:
        channels = [d.channels for d in descs] + [nchannels]
    for i, d in enumerate(descs):
        inchan, outchan = channels[i:i + 2]

        if d.keep_size:
            one_direc = "keep"
            out_size = in_size

        elif direction == "down":
            one_direc = "down"
            do_bias = use_bias
            if d.max_pool_kern:
                in_size = in_size // d.max_pool_kern
                out_size = in_size
                res.append(nn.MaxPool2d(d.max_pool_kern))
            else:
                out_size = in_size // d.stride
            
        else:
            one_direc = "up"
            do_bias = use_bias or i == len(descs) - 1
            # print(f"{i=} {do_bias=} {use_bias=}")
            if d.max_pool_kern:
                out_size = in_size * d.max_pool_kern
                in_size = out_size
                res.append(nn.Upsample(scale_factor=d.max_pool_kern))
            else:
                out_size = in_size * d.stride
            
        do_norm = i > 0 and i < len(descs) - 1
        one_layers = _gen_conv_layer(in_size, out_size, in_channels=inchan, out_channels=outchan,
                                     kernel_size=d.kernel_size, stride=d.stride, 
                                     do_layernorm=do_layernorm and do_norm,
                                     do_batchnorm=do_batchnorm and do_norm,
                                     use_bias=do_bias,
                                     direction=one_direc)
        res.extend(one_layers)
        in_size = out_size

    return res, out_size

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, emblen)
"""
class Encoder(nn.Module):
    conv_seq: nn.Sequential

    """side dimension of image after convolutions are processed"""
    out_size: int

    def __init__(self, image_size: int, emblen: int,
                 do_layernorm: bool, do_batchnorm: bool, use_bias: bool,
                 descs: List[ConvDesc], nchannels = 3):
        super().__init__()
        layers, out_size = _gen_conv_layers(
            image_size,
            "down", 
            do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, use_bias=use_bias,
            descs=descs, nchannels=nchannels
        )
        self.conv_seq = nn.Sequential()
        self.conv_seq.extend(layers)

        if emblen:
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.linear = nn.Sequential(
                nn.Linear(descs[-1].channels * out_size * out_size, emblen),
            )

        self.image_size = image_size
        self.out_size = out_size
        self.emblen = emblen
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.descs = descs
        self.nchannels = nchannels

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv_seq(inputs)
        if self.emblen:
            out = self.flatten(out)
            out = self.linear(out)

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

    def __init__(self, image_size: int, encoder_out_size: int, emblen: int, 
                 do_layernorm: bool, do_batchnorm: bool, use_bias: bool,
                 descs: List[ConvDesc], nchannels = 3):
        super().__init__()

        firstchan = descs[0].channels
        if emblen:
            self.linear = nn.Sequential(
                nn.Linear(emblen, firstchan * encoder_out_size * encoder_out_size),
                nn.ReLU(True)
            )
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(firstchan, encoder_out_size, encoder_out_size))

        layers, out_size = _gen_conv_layers(
            encoder_out_size,
            "up", 
            do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, use_bias=use_bias,
            descs=descs, nchannels=nchannels
        )
        self.conv_seq = nn.Sequential()
        self.conv_seq.extend(layers)
        self.conv_seq.append(nn.Tanh())

        self.image_size = image_size
        self.encoder_out_size = encoder_out_size
        self.out_size = out_size
        self.emblen = emblen
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
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

"""
inputs: (batch, nchannels, image_size, image_size)
return: (batch, nchannels, image_size, image_size)
"""
class ConvEncDec(nn.Module):
    encoder: Encoder
    linear_layers: nn.Sequential
    decoder: Decoder

    def __init__(self, image_size: int, emblen: int, nlinear: int, hidlen: int, 
                 do_layernorm: bool, do_batchnorm: bool, 
                 descs: List[ConvDesc], 
                 use_bias = True,
                 nchannels = 3, device = "cpu"):
        super().__init__()

        self.encoder = Encoder(image_size=image_size, emblen=emblen, 
                               do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, use_bias=use_bias,
                               descs=descs, nchannels=nchannels) #, device=device)
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
                               do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, use_bias=use_bias,
                               descs=descs_rev, nchannels=nchannels) #, device=device)
        
        self.image_size = image_size
        self.emblen = emblen
        self.nlinear = nlinear
        self.hidlen = hidlen
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.use_bias = use_bias
        self.descs = descs
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

    def state_dict(self, *args, **kwargs) -> Dict[str, any]:
        res = super().state_dict(*args, **kwargs)
        res = res.copy()
        for k in ENCDEC_FIELDS:
            res[k] = getattr(self, k)
        return res
    
    @staticmethod
    def new_from_state_dict(state_dict: Dict[str, any]):
        state_dict = {k.replace("_orig_mod.", ""): state_dict[k] for k in state_dict.keys()}
        ctor_args = {k: state_dict.pop(k) for k in ENCDEC_FIELDS if k in state_dict}
        res = ConvEncDec(**ctor_args)
        res.load_state_dict(state_dict)
        return res

"""
generate an image from pure noise.
"""
def generate(exp: Experiment, num_steps: int, size: int, 
             truth_is_noise: bool, use_timestep: bool,
             inputs: Tensor = None,
             device = "cpu") -> Tensor:
    if inputs is None:
        inputs = torch.rand((1, 3, size, size), device=device)

    # BUG: no way to pass this in, to get consistent tensor for timestep
    if use_timestep:
        timestep = torch.zeros((1, 1), device=device)
        timestep[0, 0] = 1.0 / num_steps

    orig_input = inputs
    exp.net.eval()
    if num_steps <= 1:
        return inputs
    
    # TODO: this doesn't do the right math for use_timestep, i don't think.
    with torch.no_grad():
        for step in range(num_steps - 1):
            if use_timestep:
                net_inputs = [inputs, timestep]
            else:
                net_inputs = [inputs]
            if truth_is_noise:
                out_noise = exp.net.forward(*net_inputs)
                if use_timestep:
                    out: Tensor = inputs - out_noise
                    out.clamp_(min=0.0, max=1.0)
                else:
                    keep_noise_amount = (step + 1) / num_steps
                    out = inputs - keep_noise_amount * out_noise
                inputs = out
            else:
                if use_timestep:
                    raise ValueError("bad logic not implemented")
                out = exp.net.forward(*net_inputs)
                keep_output = (step + 1) / num_steps
                out = (out * keep_output) + (inputs * (1 - keep_output))
            inputs = out
    return out

def gen_noise(size) -> Tensor:
    # return torch.normal(mean=0, std=0.5, size=size)
    return torch.rand(size=size)

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
def gen_descs(image_size: int, s: str) -> List[ConvDesc]:
    kernel_size = 0
    stride = 0

    descs: List[ConvDesc] = list()
    for onedesc_str in s.split(","):
        channels = 0
        max_pool_kern = 0
        for part in onedesc_str.split("-"):
            if part.startswith("c"):
                channels = int(part[1:])
            elif part.startswith("k"):
                kernel_size = int(part[1:])
            elif part.startswith("s"):
                stride = int(part[1:])
            elif part.startswith("mp"):
                max_pool_kern = int(part[2:])
            else:
                raise Exception("dunno what to do with {part=}")

        if not kernel_size or not stride or not channels:
            raise ValueError(f"{kernel_size=} {stride=} {channels=}")

        onedesc = ConvDesc(channels=channels, kernel_size=kernel_size, stride=stride, max_pool_kern=max_pool_kern)
        descs.append(onedesc)
    
    return descs

if __name__ == "__main__":
    import importlib
    # descs = gen_descs("k3-s2-p1-c8,c16,c32")
    image_size = 128
    emblen = 64
    nlinear = 2
    hidlen = 64
    descs = gen_descs(image_size, "k3-s1-mp2-c8,c16,c32")

    net = ConvEncDec(image_size=image_size, emblen=emblen, nlinear=nlinear, hidlen=hidlen, descs=descs, nchannels=3,
                     do_batchnorm=False, do_layernorm=True, flatconv2d_kern=0).to("cuda")
    inputs = torch.rand((1, 3, image_size, image_size), device="cuda")
    print(net.encoder)
    print(net.decoder)
    print(f"{net.encoder.out_size=}")
    print(f"{inputs.shape=}")
    out = net(inputs)
    print(f"{out.shape=}")

