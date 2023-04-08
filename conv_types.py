from typing import List, Dict, Literal, Callable
from collections import deque
from dataclasses import dataclass
from functools import reduce
import operator

from torch import nn

NlType = Literal['relu', 'sigmoid', 'silu']
NormType = Literal['batch', 'layer', 'group']
Direction = Literal['down', 'up']

@dataclass
class ConvNonlinearity:
    NlType: NlType

    def __post_init__(self):
        if self.NlType not in {'relu', 'sigmoid', 'silu'}:
            raise ValueError(f"unknown {self.NlType=}")

    def create(self):
        if self.NlType == 'relu':
            return nn.ReLU(inplace=True)
        elif self.NlType == 'sigmoid':
            return nn.Sigmoid()
        elif self.NlType == 'silu':
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError(f"unhandled {self.NlType=}")

@dataclass
class ConvNorm:
    norm_type: NormType
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
    # in/out values are from the perspective of 'down' operations. they are flipped
    # when going up.
    _in_chan: int = None
    _out_chan: int = None
    _in_size: int = None

    kernel_size: int
    stride: int
    max_pool_kern: int = 0
    max_pool_padding: int = 0
    down_padding: int = 0
    up_padding: int = 0
    up_output_padding: int = 0

    def _compute_size_down(self) -> int:
        if self.max_pool_kern:
            return self._in_size // self.max_pool_kern + self.max_pool_padding
        return (self._in_size + 2 * self.down_padding - self.kernel_size) // self.stride + 1

    def _compute_size_up(self) -> int:
        stride = self.stride
        if self.max_pool_kern:
            stride = self.max_pool_kern

        in_size = self._compute_size_down()
        out_size = (in_size - 1) * stride - 2 * self.up_padding + self.kernel_size + self.up_output_padding
        return out_size
    
    def in_chan(self, dir: Direction) -> int:
        if dir == 'up':
            return self._out_chan
        return self._in_chan
    
    def out_chan(self, dir: Direction) -> int:
        if dir == 'up':
            return self._in_chan
        return self._out_chan
    
    def in_size(self, dir: Direction) -> int:
        if dir == 'up':
            return self._compute_size_down()
        return self._in_size
    
    def out_size(self, dir: Direction) -> int:
        if dir == 'up':
            return self._compute_size_up()
        return self._compute_size_down()
    
    def out_size_desired(self, dir: Direction) -> int:
        in_size = self.in_size(dir=dir)
        factor = self.max_pool_kern or self.stride
        if dir == 'up':
            return in_size * factor
        
        return in_size // factor
    
    def __eq__(self, other: 'ConvLayer') -> bool:
        fields = ('out_chan kernel_size stride max_pool_kern '
                  'down_padding up_padding up_output_padding').split()
        return all([getattr(self, field, None) == getattr(other, field, None)
                    for field in fields])


# TODO: this should just be able to make the nn.Conv objects itself.
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

    def __init__(self, 
                 in_chan: int, in_size: int,
                 layers: List[ConvLayer], 
                 inner_nl_type: NlType = 'relu',
                 linear_nl_type: NlType = None,
                 final_nl_type: NlType = 'sigmoid',
                 inner_norm_type: NormType = 'layer',
                 final_norm_type: NormType = 'layer',
                 norm_num_groups: int = None):
        if linear_nl_type is None:
            linear_nl_type = inner_nl_type
        self.inner_nl = ConvNonlinearity(inner_nl_type)
        self.linear_nl = ConvNonlinearity(linear_nl_type)
        self.final_nl = ConvNonlinearity(final_nl_type)

        self.inner_nl_type = inner_nl_type
        self.linear_nl_type = linear_nl_type
        self.final_nl_type = final_nl_type
        self.inner_norm_type = inner_norm_type
        self.final_norm_type = final_norm_type
        self.norm_num_groups = norm_num_groups

        # set the layers' in_chan/in_size based on what was passed into the config.
        self.layers = layers.copy()
        for layer in self.layers:
            layer._in_chan = in_chan
            layer._in_size = in_size
            in_chan = layer._out_chan
            in_size = layer._compute_size_down()

        if norm_num_groups is None:
            min_chan = min([layer.out_chan('down') for layer in layers[1:]])
            norm_num_groups = min_chan
        self.inner_norm = ConvNorm(norm_type=inner_norm_type, num_groups=norm_num_groups)
        self.final_norm = ConvNorm(norm_type=final_norm_type, num_groups=norm_num_groups)

    def create_down(self) -> List[List[nn.Module]]:
        downstack: List[List[nn.Module]] = list()

        for layer in self.layers:
            in_chan = layer.in_chan('down')
            out_chan = layer.out_chan('down')

            if layer.max_pool_kern:
                conv = nn.MaxPool2d(kernel_size=layer.max_pool_kern, padding=layer.max_pool_padding)
            else:
                conv = nn.Conv2d(in_chan, out_chan, 
                                 kernel_size=layer.kernel_size, stride=layer.stride, 
                                 padding=layer.down_padding)
            
            out_chan = layer.out_chan('down')
            out_size = layer.out_size('down')
            norm = self.inner_norm.create(out_shape=(out_chan, out_size, out_size))
            nonlinearity = self.inner_nl.create()

            downstack.append([conv, norm, nonlinearity])

        return downstack

    def create_up(self) -> List[List[nn.Module]]:
        upstack: List[List[nn.Module]] = list()

        layers = list(reversed(self.layers))
        for i, layer in enumerate(layers):
            in_chan = layer.in_chan('up')
            out_chan = layer.out_chan('up')
            print(f"create_up: in_chan {in_chan}, out_chan {out_chan}")
            print(f"           in_size {layer.in_size('up')}, out_size {layer.out_size('up')}")
            stride = layer.stride
            if layer.max_pool_kern:
                stride = layer.max_pool_kern
            
            conv = nn.ConvTranspose2d(in_chan, out_chan,
                                      kernel_size=layer.kernel_size, stride=stride, 
                                      padding=layer.up_padding, output_padding=layer.up_output_padding)
            
            out_chan = layer.out_chan('up')
            out_size = layer.out_size('up')
            is_final_layer = (i == len(layers) - 1)
            if is_final_layer:
                norm = self.final_norm.create(out_shape=(out_chan, out_size, out_size))
                nonlinearity = self.final_nl.create()
            else:
                norm = self.inner_norm.create(out_shape=(out_chan, out_size, out_size))
                nonlinearity = self.inner_nl.create()

            upstack.append([conv, norm, nonlinearity])

        return upstack
    
    def get_in_dim(self, dir: Direction) -> List[int]:
        """return the dimension after completing the given direction"""
        if dir == 'up':
            layer = self.layers[-1]
        else:
            layer = self.layers[0]
        chan = layer.out_chan(dir=dir)
        size = layer.out_size(dir=dir)
        return [chan, size, size]

    def get_out_dim(self, dir: Direction) -> List[int]:
        """return the dimension after completing the given direction"""
        if dir == 'up':
            layer = self.layers[0]
        else:
            layer = self.layers[-1]
        chan = layer.out_chan(dir=dir)
        size = layer.out_size(dir=dir)
        return [chan, size, size]

    
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
            
            out_chan = layer.out_chan('down')
            if num_repeat > 1:
                res.append(f"{out_chan}x{num_repeat}")
            else:
                res.append(str(out_chan))

        return "-".join(res)

    def metadata_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        for field in self._metadata_fields:
            if field == 'layers_str':
                continue
            res[field] = getattr(self, field)
        res['layers_str'] = self.layers_str()
        return res


def parse_layers(*, layers_str: str, in_chan: int, in_size: int) -> List[ConvLayer]:
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
            layer = ConvLayer(_out_chan=out_chan, kernel_size=kernel_size, 
                              stride=stride, max_pool_kern=max_pool_kern,
                              down_padding=down_padding, 
                              up_padding=up_padding, up_output_padding=up_output_padding)
            layer._in_size = in_size
            layer._in_chan = in_chan
            in_size = layer.out_size('down')
            in_chan = layer.out_chan('down')

            down_desired = layer.out_size_desired('down')
            down_actual = layer.out_size('down')
            up_desired = layer.out_size_desired('up')
            up_actual = layer.out_size('up')

            if up_actual < up_desired:
                print("- add output padding")
                layer.up_output_padding += 1
            if down_actual < down_desired:
                if layer.max_pool_kern:
                    print("- add max pool padding")
                    layer.max_pool_padding += 1
                else:
                    print("- add down padding")
                    layer.down_padding += 1
            
            layers.append(layer)
    
    return layers

def make_config(*,
                in_chan: int, in_size: int, 
                layers_str: str, 
                inner_nl_type: NlType = 'relu',
                linear_nl_type: NlType = None,
                final_nl_type: NlType = 'sigmoid',
                inner_norm_type: NormType = 'layer',
                final_norm_type: NormType = 'layer',
                norm_num_groups: int = None) -> ConvConfig:
    layers = parse_layers(layers_str=layers_str, in_chan=in_chan, in_size=in_size)
    return ConvConfig(layers=layers,
                      in_chan=in_chan, in_size=in_size,
                      inner_nl_type=inner_nl_type,
                      linear_nl_type=linear_nl_type,
                      final_nl_type=final_nl_type,
                      inner_norm_type=inner_norm_type,
                      final_norm_type=final_norm_type,
                      norm_num_groups=norm_num_groups)

