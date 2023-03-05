import torch
from torch import Tensor, nn
import argparse

# doesn't work that great, but saving for posterity. 2023-03-05
class TheModel(nn.Module):
    def __init__(self, image_size: int, feature_len: int = 32):
        super().__init__()

        feature_len = 32
        self.shrink = nn.Sequential(
            nn.Conv2d(3, feature_len, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_len),
            nn.ReLU(),
            nn.Conv2d(feature_len, feature_len, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_len),
            nn.ReLU(),
            nn.Conv2d(feature_len, 3, kernel_size=5, stride=1, padding=2),
        )
        shrunk_len = int(3 * image_size * image_size / 16)
        self.linear_between = nn.Linear(shrunk_len, shrunk_len)
        self.embiggen = nn.Sequential(
            nn.ConvTranspose2d(3, feature_len, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_len),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_len, feature_len, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_len),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_len, 3, kernel_size=5, stride=1, padding=2),
        )

        self.straight = nn.Sequential(
            nn.ConvTranspose2d(3, feature_len, kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(feature_len, 3, kernel_size=5, stride=1, padding=2)
        )

        self.adjust_shrink_embiggen = nn.Parameter(torch.randn((3, image_size, image_size)))
        self.adjust_straight = nn.Parameter(torch.randn((3, image_size, image_size)))

    def forward(self, inputs: Tensor) -> Tensor:
        batch, chan, width, height = inputs.shape

        out_shrink_embiggen = self.shrink(inputs)
        out_shrink_embiggen = torch.flatten(out_shrink_embiggen, start_dim=1)
        out_shrink_embiggen = self.linear_between(out_shrink_embiggen)
        out_shrink_embiggen = out_shrink_embiggen.view((batch, chan, width // 4, height // 4))
        out_shrink_embiggen = self.embiggen(out_shrink_embiggen)

        out_straight = self.straight(inputs)

        out = out_shrink_embiggen * self.adjust_shrink_embiggen
        out += out_straight * self.adjust_straight

        return out


# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str

net = TheModel(cfg.image_size)
batch_size = 128
minicnt = 10

label = "shrink-lin-embiggen-lin"

# common_args = dict(loss_fn=loss_fn, train_dataloader=None, val_dataloader=None, epochs=epochs)
exp_descs = [
    dict(label=label, net=net)
]
