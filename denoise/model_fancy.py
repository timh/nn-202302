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
Rotate a batch. To be used when processing vertical rows.

inputs: (batch, chan, width, height)
return: (batch, chan, height, width)
"""
class ImageRotate(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()

        # (batch, chan, width, height) -> (batch, chan, height, width)
        self.permute = Permute((0, 1, 3, 2))
        
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.permute(inputs)
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
        self.horiz_stack = Stack(image_size=image_size, nlayers=nhoriz, do_layernorm_pre=do_layernorm_pre, do_layernorm_post=do_layernorm_post, do_relu=do_relu)

        self.rotate_vert = ImageRotate(image_size)
        self.vert_stack = Stack(image_size=image_size, nlayers=nhoriz, do_layernorm_pre=do_layernorm_pre, do_layernorm_post=do_layernorm_post, do_relu=do_relu)

        self.horiz_weights = nn.Parameter(torch.rand((image_size, image_size, 3)))
        self.vert_weights = nn.Parameter(torch.rand((image_size, image_size, 3)))

        self.layernorm_post = nn.LayerNorm(3) if do_layernorm_post_top else nn.Identity()
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch, width, height, chan = inputs.shape

        out_horiz = self.horiz_stack(inputs)

        out_vert_rot = self.rotate_vert(inputs)
        out_vert_rot = self.vert_stack(out_vert_rot)
        out_vert = self.rotate_vert(out_vert_rot)

        out = out_horiz + out_vert
        out = self.layernorm_post(out)
        return out
