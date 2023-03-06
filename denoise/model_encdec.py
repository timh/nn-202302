import torch
from torch import Tensor, nn

"""
inputs: (batch, 3, image_size, image_size)
return: (batch, emblen)
"""
class Encoder(nn.Module):
    def __init__(self, image_size: int, emblen: int):
        super().__init__()

        #  in: (batch,  3,    width,    height)
        # out: (batch, 32, width//8, height//8)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16,),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        #  in: (batch, 32, image_size//8, image_size//8)
        # out: (batch, 32 * (image_size//8 ** 2))
        self.flatten = nn.Flatten(start_dim=1)

        #  in: (batch, 32 * (image_size // 8)**2)
        # out: (batch, embdim)
        self.encoder_lin1 = nn.Linear(32 * ((image_size // 8) ** 2), 128)
        self.encoder_lin2 = nn.Linear(128, emblen)
    
    def forward(self, inputs: Tensor) -> Tensor:
        #    (batch, 3, image_size, image_size)
        # -> (batch, 32, image_size//8, image_size//8)
        out = self.conv(inputs)

        #    (batch, 32, image_size//8, image_size//8)
        # -> (batch, 32 * image_size//8 * image_size//8)
        out = self.flatten(out)

        #    (batch, 32 * image_size//8 * image_size//8)
        # -> (batch, 128)
        out = self.encoder_lin1(out)

        #    (batch, 128)
        # -> (batch, emblen)
        out = self.encoder_lin2(out)

        return out

"""
inputs: (batch, emblen)
return: (batch, 3, width, height)
"""
class Decoder(nn.Module):
    def __init__(self, image_size: int, emblen: int):
        super().__init__()
        self.decoder_lin1 = nn.Linear(emblen, 128)
        self.decoder_lin2 = nn.Linear(128, 32 * ((image_size // 8) ** 2))

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, image_size//8, image_size//8))
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # (batch, emblen) -> (batch, 128)
        out = self.decoder_lin1(inputs)

        # (batch, 128) -> (batch, 32 * ((image_size // 8) ** 2))
        out = self.decoder_lin2(out)

        #    (batch, (32 * (image_size // 8) ** 2) 
        # -> (batch, 32, image_size // 8, image_size // 8)
        out = self.unflatten(out)

        #    (batch, 32, image_size // 8, image_size // 8)
        # -> (batch, 3, image_size, image_size)
        out = self.conv(out)

        return out
