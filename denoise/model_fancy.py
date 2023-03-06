from typing import Literal
import torch
from torch import Tensor, nn

class Permute(nn.Module):
    def __init__(self, out_size: any):
        super().__init__()
        self.out_size = out_size
    
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.permute(inputs, self.out_size)

"""
Convert a batch of inputs from standard image format (batch, chan, width, height)
to a (batch * image_size, width, chan) format, so linear layers, etc, can deal
with (image_size) features instead of (image_size**2) features.

inputs: (batch, chan, width, height)
return: (batch * width, height, chan)   for "vert"
     OR (batch * height, width, chan)   for "horiz"

horiz: handle each column.
 vert: handle each row.
"""
class ImageFlatten1d(nn.Module):
    horiz_or_vert: str
    def __init__(self, image_size: int, horiz_or_vert: Literal["horiz", "vert"]):
        super().__init__()

        if horiz_or_vert == "horiz":
            # (batch, chan, width, height) -> (batch, height, width, chan)
            self.permute = Permute((0, 3, 2, 1))
        else:
            # (batch, chan, width, height) -> (batch, width, height, chan)
            self.permute = Permute((0, 2, 3, 1))
        
        # (batch, image_size, image_size, chan) -> (batch * image_size, image_size, chan)
        self.flatten = nn.Flatten(start_dim=0, end_dim=1)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.permute(inputs)
        out = self.flatten(out)
        return out

"""
Convert a batch of inputs from 1d-flattened back to normal (batch, chan, width, height)
dimensions.

inputs: (batch * width, height, chan)   for "vert"
     OR (batch * height, width, chan)   for "horiz"
return: (batch, chan, width, height)

horiz: handle each column.
 vert: handle each row.
"""
class ImageUnflatten1d(nn.Module):
    def __init__(self, image_size: int, horiz_or_vert: Literal["horiz", "vert"]):
        super().__init__()

        # (batch * image_size, image_size, chan) -> (batch, image_size, image_size, chan)
        self.unflatten = nn.Unflatten(dim=0, unflattened_size=(-1, image_size))

        # (batch * image_size, image_size, chan) -> (batch, )
        if horiz_or_vert == "horiz":
            # (batch, height, width, chan) -> (batch, chan, width, height)
            self.permute = Permute((0, 3, 2, 1))
        else:
            # (batch, width, height, chan) -> (batch, chan, width, height)
            self.permute = Permute((0, 3, 1, 2))
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.unflatten(inputs)
        out = self.permute(out)
        return out

class LinLayer(nn.Module):
    def __init__(self, image_size: int, do_layernorm_pre: bool, do_layernorm_post: bool, do_relu: bool):
        super().__init__()
        self.lnorm_pre = nn.LayerNorm(image_size) if do_layernorm_pre else nn.Identity()
        self.linear = nn.Linear(image_size, image_size)
        self.lnorm_post = nn.LayerNorm(image_size) if do_layernorm_post else nn.Identity()
        self.relu = nn.ReLU() if do_relu else nn.Identity()
    
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.lnorm_pre(inputs)
        out = self.linear(inputs)
        out = self.lnorm_post(out)
        out = self.relu(out)
        return out

class Stack(nn.Sequential):
    def __init__(self, image_size: int, nlayers: int, do_layernorm_pre: bool, do_layernorm_post: bool, do_relu: bool):
        super().__init__()
        for h in range(nlayers):
            self.append(LinLayer(image_size=image_size, do_layernorm_pre=do_layernorm_pre, do_layernorm_post=do_layernorm_post, do_relu=do_relu))

class DenoiseFancy(nn.Module):
    def __init__(self, image_size: int, nhoriz: int, nvert: int, do_layernorm_pre: bool, do_layernorm_post: bool, do_layernorm_post_top: bool, do_relu: bool):
        super().__init__()

        # standard input: (batch, chan, width, height)
        self.flatten_horiz = ImageFlatten1d(image_size, "horiz")
        self.horiz_stack = Stack(image_size=image_size, nlayers=nhoriz, do_layernorm_pre=do_layernorm_pre, do_layernorm_post=do_layernorm_post, do_relu=do_relu)
        self.unflatten_horiz = ImageUnflatten1d(image_size, "horiz")

        self.flatten_vert = ImageFlatten1d(image_size, "vert")
        self.vert_stack = Stack(image_size=image_size, nlayers=nhoriz, do_layernorm_pre=do_layernorm_pre, do_layernorm_post=do_layernorm_post, do_relu=do_relu)
        self.unflatten_vert = ImageUnflatten1d(image_size, "vert")

        self.horiz_weights = nn.Parameter(torch.rand((image_size, image_size, 3)))
        self.vert_weights = nn.Parameter(torch.rand((image_size, image_size, 3)))

        self.layernorm_post = nn.LayerNorm(3) if do_layernorm_post_top else nn.Identity()
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch, width, height, chan = inputs.shape

        out_horiz_flat = self.flatten_horiz(inputs)
        out_horiz_flat = self.horiz_stack(out_horiz_flat)
        out_horiz_flat = out_horiz_flat * self.vert_weights
        out_horiz = self.unflatten_horiz(out_horiz_flat)

        out_vert_flat = self.flatten_vert(inputs)
        out_vert_flat = self.vert_stack(out_vert_flat)
        out_vert_flat = out_vert_flat * self.vert_weights
        out_vert = self.unflatten_vert(out_vert_flat)

        out = out_horiz + out_vert
        out = self.layernorm_post(out)
        return out
