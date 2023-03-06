import torch
from torch import Tensor, nn

"""
inputs: (batch, 3, image_size, image_size)
return: (batch, emblen)
"""
class Encoder(nn.Module):
    def __init__(self, image_size: int, emblen: int, hidlen: int, nlevels: int = 3, startlevel: int = 8):
        super().__init__()

        self.nlevels = nlevels
        self.startlevel = startlevel
        self.endlevel = startlevel * (2 ** (nlevels - 1))
        self.image_size = image_size
        self.image_size_shrunk = image_size // (2 ** nlevels)

        #  in: (batch,  3,    width,    height)
        # out: (batch, 32, width//8, height//8)
        self.conv = nn.Sequential()
        lastlevel = 3
        level = startlevel
        for i in range(nlevels):
            self.conv.append(nn.Conv2d(lastlevel, level, kernel_size=3, stride=2, padding=1))
            if i > 0 and i < nlevels - 1:
                self.conv.append(nn.BatchNorm2d(level))
            self.conv.append(nn.ReLU(inplace=True))

            lastlevel = level
            level = level * 2

        #  in: (batch, 32, self.image_size_shrunk, self.image_size_shrunk)
        # out: (batch, 32 * (self.image_size_shrunk ** 2))
        self.flatten = nn.Flatten(start_dim=1)

        #  in: (batch, 32 * image_size_shrunk**2)
        # out: (batch, embdim)
        self.encoder_lin1 = nn.Linear(self.endlevel * (self.image_size_shrunk ** 2), hidlen)
        self.encoder_lin2 = nn.Linear(hidlen, emblen)
    
    def forward(self, inputs: Tensor) -> Tensor:
        #    (batch, 3, image_size, image_size)
        # -> (batch, endlevel, image_size_shrunk, image_size_shrunk)
        out = self.conv(inputs)

        #    (batch, endlevel, image_size_shrunk, image_size_shrunk)
        # -> (batch, endlevel * image_size_shrunk * image_size_shrunk)
        out = self.flatten(out)

        #    (batch, endlevel * image_size_shrunk * image_size_shrunk)
        # -> (batch, hidlen)
        out = self.encoder_lin1(out)

        #    (batch, hidlen)
        # -> (batch, emblen)
        out = self.encoder_lin2(out)

        return out

"""
inputs: (batch, emblen)
return: (batch, 3, width, height)
"""
class Decoder(nn.Module):
    def __init__(self, image_size: int, emblen: int, hidlen: int, nlevels: int = 3, startlevel: int = 8):
        super().__init__()
        self.nlevels = nlevels
        self.startlevel = startlevel
        self.endlevel = startlevel * (2 ** (nlevels - 1))
        self.image_size = image_size
        self.image_size_shrunk = image_size // (2 ** nlevels)

        self.decoder_lin1 = nn.Linear(emblen, hidlen)
        self.decoder_lin2 = nn.Linear(hidlen, self.endlevel * (self.image_size_shrunk ** 2))

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.endlevel, self.image_size_shrunk, self.image_size_shrunk))

        self.conv = nn.Sequential()

        lastlevel = self.endlevel
        level = lastlevel // 2
        for i in range(nlevels):
            if i == nlevels - 1:
                level = 3
            self.conv.append(nn.ConvTranspose2d(lastlevel, level, kernel_size=3, stride=2, padding=1, output_padding=1))
            if i != nlevels - 1:
                self.conv.append(nn.BatchNorm2d(level))
                self.conv.append(nn.ReLU(inplace=True))
            
            lastlevel = level
            level = level // 2
        
    def forward(self, inputs: Tensor) -> Tensor:
        # (batch, emblen) -> (batch, hidlen)
        out = self.decoder_lin1(inputs)

        # (batch, hidlen) -> (batch, 32 * (image_size_shrunk ** 2))
        out = self.decoder_lin2(out)

        #    (batch, (endlevel * (image_size_shrunk ** 2) )
        # -> (batch, endlevel, image_size_shrunk, image_size_shrunk)
        out = self.unflatten(out)

        #    (batch, endlevel, image_size_shrunk, image_size_shrunk)
        # -> (batch, 3, image_size, image_size)
        out = self.conv(out)

        return out
