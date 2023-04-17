from typing import List, Dict, Literal
import functools, operator

from torch import nn, Tensor

import sys
sys.path.append("..")
import base_model
import conv_types

NLType = Literal['relu', 'silu', 'gelu']

class FlattenLinear(base_model.BaseModel):
    _metadata_fields = ["in_dim", "out_len", "nonlinearity", "nlayers"]
    _model_fields = _metadata_fields

    def __init__(self, 
                 in_dim: List[int], out_len: int, 
                 nlayers: int, nonlinearity: NLType):
        super().__init__()
        self.in_dim = in_dim
        self.out_len = out_len
        self.nonlinearity = nonlinearity
        self.nlayers = nlayers

        in_flat = functools.reduce(operator.mul, in_dim, 1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()
        for i in range(nlayers):
            self.linear.append(nn.Linear(in_flat, out_len))
            if nonlinearity == 'relu':
                self.linear.append(nn.ReLU(True))
            in_flat = out_len
    
    def forward(self, inputs: Tensor, _clip_emb: Tensor = None) -> Tensor:
        out = self.flatten(inputs)
        return self.linear(out)

class FlattenConv(base_model.BaseModel):
    _metadata_fields = ["in_dim", "out_len"]
    _model_fields = _metadata_fields

    def __init__(self, 
                 in_dim: List[int], out_len: int, 
                 cfg: conv_types.ConvConfig):
        super().__init__()
        self.in_dim = in_dim
        self.out_len = out_len
        self.conv_cfg = cfg

        self.down = nn.Sequential()
        for layer in cfg.layers:
            one_down = nn.Sequential(*cfg.create_down(layer))
            self.down.append(one_down)
    
    def forward(self, inputs: Tensor, _clip_emb: Tensor = None) -> Tensor:
        batch = inputs.shape[0]
        out = self.down(inputs)
        return out.view((batch, self.out_len))

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
