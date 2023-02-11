import math
import datetime
import sys
import os
import random

from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torchvision.utils as vutils
from PIL import Image

import trainer

def combine_left_right(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lbatch, lnumchan, lheight, lwidth = left.shape
    rbatch, rnumchan, rheight, rwidth = right.shape

    assert lbatch == rbatch
    assert lnumchan == rnumchan
    assert lheight == rheight

    outwidth = lwidth + rwidth
    out = torch.zeros((lbatch, lnumchan, lheight, outwidth))
    out[:, :, :, :lwidth] = left
    out[:, :, :, lwidth:] = right

    return out

def combine_batch(batched_output: torch.Tensor) -> torch.Tensor:
    batch, numchan, height, width = batched_output.shape

    out = np.transpose(batched_output, (0, 2, 3, 1))     # -> (batch, height, width, numchan)
    out = out.reshape((batch * height, width, numchan))  # -> (batch * height, width, numchan)
    out = np.transpose(out, (2, 0, 1))                   # -> (numchan, batch * height, width)

    return out

class RightChunkDataLoader:
    _chunk_size: int = 0
    _iter: any = None
    _dataloader: DataLoader = None

    _last_data: Tuple[torch.Tensor, any] = None
    _data_width: int = 0
    _data_height: int = 0
    _data_xchunks: int = 0
    _data_ychunks: int = 0
    _data_xi = 0
    _data_yi = 0

    def __init__(self, dataset: Dataset, chunk_size: int, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self._chunk_size = chunk_size
        iter(self)
        self._fetch_next()
    
    def __iter__(self):
        self._iter = iter(self._dataloader)
        return self

    def __len__(self):
        return self._data_xchunks * self._data_ychunks * len(self._dataloader)
    
    def _fetch_next(self):
        self._last_data = next(self._iter)[0]
        self._data_width = self._last_data.shape[3]
        self._data_height = self._last_data.shape[2]
        self._data_xchunks = int(self._data_width / self._chunk_size)
        self._data_ychunks = int(self._data_height / self._chunk_size)
        self._data_xi = 0
        self._data_yi = 0
    
    def __next__(self) -> any:
        ystart = self._data_yi * self._chunk_size
        yend = ystart + self._chunk_size

        xstart = self._data_xi * self._chunk_size
        xend = xstart + self._chunk_size

        inputs = self._last_data[:, :, xstart:xend, ystart:yend]
        expected = self._last_data[:, :, xend:xend + self._chunk_size, ystart:yend]

        self._data_xi += 1
        if xend + self._chunk_size >= self._data_width:
            self._data_xi = 0
            self._data_yi += 1
        if yend + self._chunk_size >= self._data_height:
            self._fetch_next()

        return inputs, expected

class OutputTrainer(trainer.ImageTrainer):
    def __init__(self, dataloader: DataLoader, chunk_size: int, device: str,
                 dirname: str, net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, 
                 grid_num_images: int, num_fake_cols = 1):
        # pass in the *real* dataloader, not the chunk one
        super().__init__(dirname, net, loss_fn, optimizer, grid_num_rows=2, grid_num_cols=4, grid_num_images=grid_num_images)
        self._dataloader = dataloader
        self._chunk_size = chunk_size
        self._device = device
        self._num_fake_cols = num_fake_cols
    
    def _setup_fig(self):
        super()._setup_fig()
        self._axes_recons_real = self._fig.add_subplot(self.grid_num_rows, self.grid_num_cols, 3)
        self._axes_recons_fake = self._fig.add_subplot(self.grid_num_rows, self.grid_num_cols, 4)
    
    def update_fig(self, epoch: int, expected: torch.Tensor, outputs: torch.Tensor):
        super().update_fig(epoch, expected, outputs)

        inputs, _expected = next(iter(self._dataloader))
        idx = torch.randint(0, len(inputs), (1,)).item()

        fullsize_img = inputs[idx]
        real_img = fullsize_img[:, :, 0 : self._chunk_size * (self._num_fake_cols + 1)]

        fake_img = real_img.clone()
        height = real_img.shape[1]
        clamped_height = int(height / self._chunk_size) * self._chunk_size
        num_batch = int(clamped_height / self._chunk_size)
        num_chan = real_img.shape[0]

        # get the left side of the real image, then divide it into a batch of chunks
        halfreal_img = real_img[:, :clamped_height, :self._chunk_size]                         # (3, height, 16)
        halfreal_img = np.transpose(halfreal_img, (1, 2, 0))                                   # (height, 16, 3)

        inputs = halfreal_img.reshape(num_batch, self._chunk_size, self._chunk_size, num_chan) # (B, 16, 16, 3)
        inputs = np.transpose(inputs, (0, 3, 1, 2))                                            # (B, 3, 16, 16)
        if self._device:
            inputs = inputs.to(self._device)

        fake_img = real_img.clone()
        for col in range(self._num_fake_cols):
            outputs = self.net(inputs)                                                         # (B, 3, 16, 16)
            inputs = outputs
            outputs = np.transpose(outputs.detach().cpu(), (0, 2, 3, 1))                       # (B, 16, 16, 3)
            outputs = outputs.reshape((clamped_height, self._chunk_size, num_chan))            # (B*16, 16, 3)
            outputs = np.transpose(outputs, (2, 0, 1))                                         # (3, B*16, 16)
            left = (col + 1) * self._chunk_size
            right = left + self._chunk_size
            fake_img[:, :, left:right] = outputs

        # now copy the output right strip into the fake img (replacing the original real
        # half)
        self._axes_recons_real.clear()
        self._axes_recons_real.set_title("real")
        self._axes_recons_real.imshow(np.transpose(real_img, (1, 2, 0)))

        self._axes_recons_fake.clear()
        self._axes_recons_fake.set_title("fake")
        self._axes_recons_fake.imshow(np.transpose(fake_img, (1, 2, 0)))

def main(dirname: str, epochs: int, do_display: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 5e-3
    # batch_size = 128
    batch_size = 2 ** 9

    chunk_size = 16
    len_feature_maps = 64
    num_channels = 3

    dataset = torchvision.datasets.ImageFolder(
        root=dirname,
        transform=transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = RightChunkDataLoader(dataset, chunk_size, batch_size, False)

    net = nn.Sequential(
        #                  in_channels,                    kernel_size,
        #                  |             out_channels,     |  stride,
        #                  |             |                 |  |  padding)
        nn.ConvTranspose2d(num_channels, len_feature_maps, 5, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 5, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 5, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 5, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, num_channels, 5, 1, 2, bias=False),
        nn.Tanh()
    ).to(device)

    dirname = "outputs/" + "-".join([dirname, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S")])
    os.makedirs(dirname, exist_ok=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    # timutil.train2(dataloader, dirname, net, loss_fn, optimizer, epochs, device, do_display)
    # t = trainer.ImageTrainer(dirname, net, loss_fn, optimizer)
    t = OutputTrainer(dataloader._dataloader, chunk_size, device,
                      dirname, net, loss_fn, optimizer,
                      grid_num_images=64, num_fake_cols=4)
    t.train(dataloader, epochs, device, True)

if __name__ == "__main__":
    main("alex-outpaint-128", 10000, True)
