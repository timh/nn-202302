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

    kernel_size: int = 0
    stride: int = 0
    max_pool_kern: int = 0
    max_pool_padding: int = 0
    down_padding: int = 0
    up_padding: int = 0
    up_output_padding: int = 0

    sa_nheads: int = 0
    ca_nheads: int = 0
    time_emb: bool = False

    def _compute_size_down(self) -> int:
        if self.sa_nheads or self.ca_nheads:
            return self._in_size

        if self.max_pool_kern:
            return self._in_size // self.max_pool_kern + self.max_pool_padding
        return (self._in_size + 2 * self.down_padding - self.kernel_size) // self.stride + 1

    def _compute_size_up(self) -> int:
        if self.sa_nheads or self.ca_nheads:
            return self._in_size
        
        stride = self.stride
        if self.max_pool_kern:
            stride = self.max_pool_kern

        in_size = self._compute_size_down()
        out_size = (in_size - 1) * stride - 2 * self.up_padding + self.kernel_size + self.up_output_padding
        return out_size
    
    def in_chan(self, dir: Direction) -> int:
        if dir == 'down':
            return self._in_chan

        # else 'up'
        return self._out_chan
    
    def out_chan(self, dir: Direction) -> int:
        if dir == 'up':
            return self._in_chan
        # else 'down'
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
        fields = ('kernel_size stride max_pool_kern '
                  'down_padding up_padding up_output_padding '
                  'sa_nheads ca_nheads time_emb').split()
        res = all([getattr(self, field, None) == getattr(other, field, None)
                   for field in fields])
        if res:
            res = self.out_chan('down') == other.out_chan('down')
        return res
    
    def copy(self) -> 'ConvLayer':
        res = ConvLayer()
        fields = ('_in_chan _out_chan _in_size '
                  'kernel_size stride max_pool_kern '
                  'down_padding up_padding up_output_padding '
                  'sa_nheads ca_nheads time_emb').split()
        for field in fields:
            setattr(res, field, getattr(self, field))
        return res
        
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

        self.in_chan = in_chan
        self.in_size = in_size

        # set the layers' in_chan/in_size based on what was passed into the config.
        self.layers = layers.copy()
        for i, layer in enumerate(self.layers):
            layer._in_chan = in_chan
            layer._in_size = in_size
            in_chan = layer._out_chan
            in_size = layer._compute_size_down()

        if norm_num_groups is None:
            min_chan = min([layer.out_chan('down') for layer in layers])
            norm_num_groups = min_chan
        self.inner_norm = ConvNorm(norm_type=inner_norm_type, num_groups=norm_num_groups)
        self.final_norm = ConvNorm(norm_type=final_norm_type, num_groups=norm_num_groups)
        self.norm_num_groups = norm_num_groups

    def create_down(self, layer: ConvLayer) -> List[nn.Module]:
        in_chan = layer.in_chan('down')
        out_chan = layer.out_chan('down')

        if layer.sa_nheads or layer.ca_nheads:
            raise NotImplementedError("create_down not implemented for SelfAttention/CrossAttention")

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

        return [conv, norm, nonlinearity]

    def create_up(self, layer: ConvLayer) -> List[nn.Module]:
        in_chan = layer.in_chan('up')
        out_chan = layer.out_chan('up')
        out_size = layer.out_size('up')

        if layer.sa_nheads or layer.ca_nheads:
            raise NotImplementedError("create_up not implemented for SelfAttention/CrossAttention")

        stride = layer.stride
        if layer.max_pool_kern:
            stride = layer.max_pool_kern
        
        conv = nn.ConvTranspose2d(in_chan, out_chan,
                                  kernel_size=layer.kernel_size, stride=stride, 
                                  padding=layer.up_padding, output_padding=layer.up_output_padding)

        if layer.out_chan('up') == self.in_chan:
            norm = self.final_norm.create(out_shape=(out_chan, out_size, out_size))
            nonlinearity = self.final_nl.create()
        else:
            norm = self.inner_norm.create(out_shape=(out_chan, out_size, out_size))
            nonlinearity = self.inner_nl.create()
        
        return [conv, norm, nonlinearity]
    
    def create_down_all(self) -> List[List[nn.Module]]:
        return [self.create_down(layer) for layer in self.layers]

    def create_up_all(self) -> List[List[nn.Module]]:
        return [self.create_up(layer) for layer in reversed(self.layers)]
    
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
        res: List[str] = list()
        last_kern_size: int = 0

        # always start the string with the kernel size
        for layer in self.layers:
            if layer.kernel_size:
                res.append(f"k{layer.kernel_size}")
                last_kern_size = layer.kernel_size
                break

        layers = deque(self.layers)
        while len(layers):
            layer = layers.popleft()
            out_chan = layer.out_chan('down')

            if layer.max_pool_kern:
                res.append(f"mp{layer.max_pool_kern}")
                continue

            if layer.sa_nheads:
                res.append(f"sa{layer.sa_nheads}")
                continue

            if layer.ca_nheads:
                res.append(f"ca{layer.ca_nheads}")
                continue

            if layer.kernel_size != last_kern_size:
                res.append(f"k{layer.kernel_size}")
                last_kern_size = layer.kernel_size
            
            if layer.stride != 1:
                res.append(f"{out_chan}s{layer.stride}")
                continue

            if layer.time_emb:
                res.append(f"t+{out_chan}")
                continue
            
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
    kernel_size = 3
    stride = 1
    out_chan = 0
    max_pool_kern = 0

    down_padding = 1
    up_padding = 1
    up_output_padding = 0

    sa_nheads = 0
    ca_nheads = 0

    use_backcompat_stride = False
    if any([part[0] == "s" and part[1].isdigit() for part in layers_str.split("-")]):
        print(f"using backwards compatible stride parsing:\n  {layers_str}")
        use_backcompat_stride = True

    layers: List[ConvLayer] = list()
    for part in layers_str.split("-"):
        if part.startswith("k"):
            kernel_size = int(part[1:])
            continue
        if part[0] == "s" and part[1].isdigit():
            stride = int(part[1:])
            continue

        if "x" in part:
            part, repeat = part.split("x", 1)
            repeat = int(repeat)
        else:
            repeat = 1

        if part.startswith("sa"):
            rest = part[2:]
            sa_nheads = int(rest)
            layer = ConvLayer(_in_chan=in_chan, _out_chan=in_chan, sa_nheads=sa_nheads)
            layers.append(layer)
            continue

        if part.startswith("ca"):
            rest = part[2:]
            ca_nheads = int(rest)
            layer = ConvLayer(_in_chan=in_chan, _out_chan=in_chan, ca_nheads=ca_nheads)
            layers.append(layer)
            continue

        time_emb = False
        # maxpool2d
        if part.startswith("mp"):
            max_pool_kern = int(part[2:])
            out_chan = in_chan

        # else normal channel digits.
        else:
            if part.startswith("t+"):
                time_emb = True
                part = part[2:]

            if "s" in part:
                out_chan, stride = map(int, part.split("s", 1))
            else:
                out_chan = int(part)
                if not use_backcompat_stride:
                    stride = 1

        if (not kernel_size or not stride) and not max_pool_kern:
            raise ValueError(f"{kernel_size=} {stride=} {out_chan=}")

        for _ in range(repeat):
            layer = ConvLayer(_out_chan=out_chan, kernel_size=kernel_size, 
                              stride=stride, max_pool_kern=max_pool_kern,
                              down_padding=down_padding, 
                              up_padding=up_padding, up_output_padding=up_output_padding,
                              time_emb=time_emb)
            layer._in_size = in_size
            layer._in_chan = in_chan
            in_size = layer.out_size('down')
            in_chan = layer.out_chan('down')

            down_desired = layer.out_size_desired('down')
            down_actual = layer.out_size('down')
            up_desired = layer.out_size_desired('up')
            up_actual = layer.out_size('up')

            if up_actual < up_desired:
                # print("- add output padding")
                layer.up_output_padding += 1
            if down_actual < down_desired:
                if layer.max_pool_kern:
                    print("- add max pool padding")
                    layer.max_pool_padding += 1
                else:
                    print("- add down padding")
                    layer.down_padding += 1
            
            layers.append(layer)
        max_pool_kern = 0
        out_chan = 0
    
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

