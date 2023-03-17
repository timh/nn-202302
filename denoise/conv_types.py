from typing import List, Dict, Literal, Callable
from dataclasses import dataclass

from torch import nn

NONLINEARITY_TYPE = Literal['relu', 'sigmoid', 'silu']
NORM_TYPE = Literal['batch', 'layer', 'group']
DIRECTION_TYPE = Literal['down', 'up']

@dataclass
class ConvNonlinearity:
    nl_type: NONLINEARITY_TYPE

    def __post_init__(self):
        if self.nl_type not in {'relu', 'sigmoid', 'silu'}:
            raise ValueError(f"unknown {self.nl_type=}")

    def create(self):
        if self.nl_type == 'relu':
            return nn.ReLU(inplace=True)
        elif self.nl_type == 'sigmoid':
            return nn.Sigmoid()
        elif self.nl_type == 'silu':
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError(f"unhandled {self.nl_type=}")

@dataclass
class ConvNorm:
    norm_type: NORM_TYPE

    def __post_init__(self):
        if self.norm_type not in {'batch', 'layer', 'group'}:
            raise ValueError(f"unknown {self.norm_type=}")
    
    def create(self, *, out_chan: int, out_size: int):
        if self.norm_type == 'batch':
            return nn.BatchNorm2d(num_features=out_chan)
        elif self.norm_type == 'layer':
            shape = (out_chan, out_size, out_size)
            return nn.LayerNorm(normalized_shape=shape)
        else:
            raise NotImplementedError(f"unhandled {self.norm_type=}")

@dataclass(kw_only=True)
class ConvLayer:
    out_chan: int
    kernel_size: int
    stride: int
    max_pool_kern: int = 0
    down_padding: int = 0
    up_padding: int = 0
    up_output_padding: int = 0

    def get_size_down_actual(self, in_size: int) -> int:
        if self.max_pool_kern:
            return in_size // self.max_pool_kern

        out_size = (in_size + 2 * self.down_padding - self.kernel_size) // self.stride + 1
        return out_size
    
    def get_size_down_desired(self, in_size: int) -> int:
        if self.max_pool_kern:
            return in_size // self.max_pool_kern
        return in_size // self.stride

    def get_size_up_actual(self, in_size: int) -> int:
        if self.max_pool_kern:
            return in_size * self.max_pool_kern

        out_size = (in_size - 1) * self.stride - 2 * self.up_padding + self.kernel_size + self.up_output_padding
        return out_size

    def get_size_up_desired(self, in_size: int) -> int:
        if self.max_pool_kern:
            return in_size * self.max_pool_kern
        return in_size * self.stride


class ConvConfig:
    layers: List[ConvLayer]
    inner_nonlinearity: ConvNonlinearity
    linear_nonlinearity: ConvNonlinearity
    final_nonlinearity: ConvNonlinearity
    norm: ConvNorm

    inner_nonlinearity_type: str
    linear_nonlinearity_type: str
    final_nonlinearity_type: str
    norm_type: str

    def __init__(self, layers: List[ConvLayer], 
                 inner_nonlinearity_type: NONLINEARITY_TYPE = 'relu',
                 linear_nonlinearity_type: NONLINEARITY_TYPE = None,
                 final_nonlinearity_type: NONLINEARITY_TYPE = 'sigmoid',
                 norm_type: NORM_TYPE = 'layer'):
        if linear_nonlinearity_type is None:
            linear_nonlinearity_type = inner_nonlinearity_type
        self.layers = layers.copy()
        self.inner_nonlinearity = ConvNonlinearity(inner_nonlinearity_type)
        self.linear_nonlinearity = ConvNonlinearity(linear_nonlinearity_type)
        self.final_nonlinearity = ConvNonlinearity(final_nonlinearity_type)
        self.norm = ConvNorm(norm_type)

        self.inner_nonlinearity_type = inner_nonlinearity_type
        self.linear_nonlinearity_type = linear_nonlinearity_type
        self.final_nonlinearity_type = final_nonlinearity_type
        self.norm_type = norm_type

    def get_channels_down(self, nchannels: int) -> List[int]:
        return [nchannels] + [l.out_chan for l in self.layers]
        
    def get_channels_up(self, nchannels: int) -> List[int]:
        return [l.out_chan for l in reversed(self.layers)] + [nchannels]

    def _build_sizes(self, in_size: int, fn: Callable[[ConvLayer, int], int]) -> List[int]:
        sizes = [in_size]
        for l in self.layers:
            out_size = fn(l, in_size)
            sizes.append(out_size)
            in_size = out_size
        return sizes

    def get_sizes_down_actual(self, in_size: int) -> List[int]:
        return self._build_sizes(in_size, ConvLayer.get_size_down_actual)

    def get_sizes_down_desired(self, in_size: int) -> List[int]:
        return self._build_sizes(in_size, ConvLayer.get_size_down_desired)

    def get_sizes_up_actual(self, in_size: int) -> List[int]:
        return self._build_sizes(in_size, ConvLayer.get_size_up_actual)

    def get_sizes_up_desired(self, in_size: int) -> List[int]:
        return self._build_sizes(in_size, ConvLayer.get_size_up_desired)

    def create_inner_nl(self) -> nn.Module:
        return self.inner_nonlinearity.create()
    
    def create_final_nl(self) -> nn.Module:
        return self.final_nonlinearity.create()

    def create_norm(self, *, out_chan: int, out_size: int) -> nn.Module:
        return self.norm.create(out_chan=out_chan, out_size=out_size)
    
    def layers_str(self) -> str:
        fields = {
            'kernel_size': "k",
            'stride': "s",
            'max_pool_kern': "mp",
            'down_padding': "dp",
            'up_padding': "up",
            'up_output_padding': "op"
        }
        last_values = {field: 0 for field in fields.keys()}
        last_values['up_padding'] = 1
        last_values['down_padding'] = 1

        res = []
        for layer in self.layers:
            for field, field_short in fields.items():
                curval = getattr(layer, field)
                lastval = last_values[field]
                if curval != lastval:
                    res.append(f"{field_short}{curval}")
                    last_values[field] = curval
            res.append(str(layer.out_chan))

        return "-".join(res)

    def metadata_dict(self) -> Dict[str, any]:
        res = dict(
            inner_nonlinearity_type=self.inner_nonlinearity_type,
            linear_nonlinearity_type=self.linear_nonlinearity_type,
            final_nonlinearity_type=self.final_nonlinearity_type,
            norm_type=self.norm_type,
            layers_str=self.layers_str()
        )
        return res


def parse_layers(layers_str: str) -> List[ConvLayer]:
    kernel_size = 0
    stride = 0
    out_chan = 0
    max_pool_kern = 0

    down_padding = 1
    up_padding = 1
    up_output_padding = 0

    layers: List[ConvLayer] = list()
    for part in layers_str.split("-"):
        if part.startswith("k"):
            kernel_size = int(part[1:])
            continue
        elif part.startswith("s"):
            stride = int(part[1:])
            continue
        elif part.startswith("p"):
            down_padding = int(part[1:])
            up_padding = down_padding
            continue
        elif part.startswith("dp"):
            down_padding = int(part[2:])
            continue
        elif part.startswith("up"):
            up_padding = int(part[2:])
            continue
        elif part.startswith("op"):
            up_output_padding = int(part[3:])
            continue
        elif part.startswith("mp"):
            max_pool_kern = int(part[2:])
            continue

        out_chan = int(part)

        if not kernel_size or not stride:
            raise ValueError(f"{kernel_size=} {stride=} {out_chan=}")

        layer = ConvLayer(out_chan=out_chan, kernel_size=kernel_size, 
                          stride=stride, max_pool_kern=max_pool_kern,
                          down_padding=down_padding, 
                          up_padding=up_padding, up_output_padding=up_output_padding)
        layers.append(layer)
    
    return layers

def make_config(layers_str: str,
                inner_nonlinearity_type: NONLINEARITY_TYPE = 'relu',
                final_nonlinearity_type: NONLINEARITY_TYPE = 'sigmoid',
                norm_type: NORM_TYPE = 'layer') -> ConvConfig:
    layers = parse_layers(layers_str)
    return ConvConfig(layers=layers,
                      inner_nonlinearity_type=inner_nonlinearity_type,
                      final_nonlinearity_type=final_nonlinearity_type,
                      norm_type=norm_type)

