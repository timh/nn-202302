from typing import List, Dict, Literal, Callable
from collections import deque
from dataclasses import dataclass
from functools import reduce
import operator

from torch import nn

nl_type = Literal['relu', 'sigmoid', 'silu']
NORM_TYPE = Literal['batch', 'layer', 'group']
DIRECTION_TYPE = Literal['down', 'up']

@dataclass
class ConvNonlinearity:
    nl_type: nl_type

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
    num_groups: int = 32

    def __post_init__(self):
        if self.norm_type not in {'batch', 'layer', 'group'}:
            raise ValueError(f"unknown {self.norm_type=}")
    
    def create(self, *, out_shape: List[int]):
        if self.norm_type == 'batch':
            out_size = reduce(operator.mul, out_shape, 1)
            # print(f"{out_shape=} {out_size=}")
            return nn.BatchNorm2d(num_features=out_size)
        elif self.norm_type == 'layer':
            return nn.LayerNorm(normalized_shape=out_shape)
        elif self.norm_type == 'group':
            # print(f"{self.num_groups=} {out_shape[0]=}")
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=out_shape[0])
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

    def __eq__(self, other: 'ConvLayer') -> bool:
        out_chan: int
        kernel_size: int
        stride: int
        max_pool_kern: int = 0
        down_padding: int = 0
        up_padding: int = 0
        up_output_padding: int = 0

        fields = ('out_chan kernel_size stride max_pool_kern '
                  'down_padding up_padding up_output_padding').split()
        return all([getattr(self, field, None) == getattr(other, field, None)
                    for field in fields])



class ConvConfig:
    _metadata_fields = ('inner_nl_type linear_nl_type final_nl_type '
                        'inner_norm_type final_norm_type norm_num_groups '
                        'layers_str').split()

    layers: List[ConvLayer]
    inner_nl: ConvNonlinearity
    linear_nl: ConvNonlinearity
    final_nl: ConvNonlinearity
    inner_norm: ConvNorm
    final_norm: ConvNorm

    inner_nl_type: str
    linear_nl_type: str
    final_nl_type: str
    inner_norm_type: str
    final_norm_type: str

    def __init__(self, layers: List[ConvLayer], 
                 inner_nl_type: nl_type = 'relu',
                 linear_nl_type: nl_type = None,
                 final_nl_type: nl_type = 'sigmoid',
                 inner_norm_type: NORM_TYPE = 'layer',
                 final_norm_type: NORM_TYPE = 'layer',
                 norm_num_groups: int = None):
        if linear_nl_type is None:
            linear_nl_type = inner_nl_type
        self.layers = layers.copy()
        self.inner_nl = ConvNonlinearity(inner_nl_type)
        self.linear_nl = ConvNonlinearity(linear_nl_type)
        self.final_nl = ConvNonlinearity(final_nl_type)

        if norm_num_groups is None:
            min_chan = min(self.get_channels_down(3)[1:])
            norm_num_groups = min_chan
        self.inner_norm = ConvNorm(norm_type=inner_norm_type, num_groups=norm_num_groups)
        self.final_norm = ConvNorm(norm_type=final_norm_type, num_groups=norm_num_groups)

        self.inner_nl_type = inner_nl_type
        self.linear_nl_type = linear_nl_type
        self.final_nl_type = final_nl_type
        self.inner_norm_type = inner_norm_type
        self.final_norm_type = final_norm_type
        self.norm_num_groups = norm_num_groups

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
        return self.inner_nl.create()
    
    def create_linear_nl(self) -> nn.Module:
        return self.linear_nl.create()
    
    def create_final_nl(self) -> nn.Module:
        return self.final_nl.create()

    def create_inner_norm(self, *, out_shape: List[int]) -> nn.Module:
        return self.inner_norm.create(out_shape=out_shape)
    
    def create_final_norm(self, *, out_shape: List[int]) -> nn.Module:
        return self.final_norm.create(out_shape=out_shape)
    
    def layers_str(self) -> str:
        fields = {
            'kernel_size': "k",
            'stride': "s",
            'max_pool_kern': "mp",
            'down_padding': "dp",
            'up_padding': "up",
            # 'up_output_padding': "op"
        }
        last_values = {field: 0 for field in fields.keys()}
        last_values['up_padding'] = 1
        last_values['down_padding'] = 1

        res = []
        layers = deque(self.layers)
        while len(layers):
            layer = layers.popleft()

            for field, field_short in fields.items():
                curval = getattr(layer, field)
                lastval = last_values[field]
                if curval != lastval:
                    res.append(f"{field_short}{curval}")
                    last_values[field] = curval

            num_repeat = 1
            while len(layers) and layer == layers[0]:
                layers.popleft()
                num_repeat += 1
            
            if num_repeat > 1:
                res.append(f"{layer.out_chan}x{num_repeat}")
            else:
                res.append(str(layer.out_chan))

        return "-".join(res)

    def metadata_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in self._metadata_fields:
            if field == 'layers_str':
                continue
            res[field] = getattr(self, field)
        res['layers_str'] = self.layers_str()
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
        # elif part.startswith("op"):
        #     up_output_padding = int(part[2:])
        #     continue
        elif part.startswith("mp"):
            max_pool_kern = int(part[2:])
            continue

        if "x" in part:
            out_chan, repeat = map(int, part.split("x"))
        else:
            out_chan = int(part)
            repeat = 1

        if not kernel_size or not stride:
            raise ValueError(f"{kernel_size=} {stride=} {out_chan=}")

        for _ in range(repeat):
            layer = ConvLayer(out_chan=out_chan, kernel_size=kernel_size, 
                            stride=stride, max_pool_kern=max_pool_kern,
                            down_padding=down_padding, 
                            up_padding=up_padding, up_output_padding=up_output_padding)
            if up_output_padding == 0 and layer.get_size_up_actual(16) < layer.get_size_up_desired(16):
                # print(f"adding output padding")
                layer.up_output_padding = 1
            layers.append(layer)
    
    return layers

def make_config(layers_str: str, 
                inner_nl_type: nl_type = 'relu',
                linear_nl_type: nl_type = None,
                final_nl_type: nl_type = 'sigmoid',
                inner_norm_type: NORM_TYPE = 'layer',
                final_norm_type: NORM_TYPE = 'layer',
                norm_num_groups: int = None) -> ConvConfig:
    layers = parse_layers(layers_str)
    return ConvConfig(layers=layers,
                      inner_nl_type=inner_nl_type,
                      linear_nl_type=linear_nl_type,
                      final_nl_type=final_nl_type,
                      inner_norm_type=inner_norm_type,
                      final_norm_type=final_norm_type,
                      norm_num_groups=norm_num_groups)

