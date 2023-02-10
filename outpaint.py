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

import timutil

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

def main(dirname: str, epochs: int, do_display: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 1e-3
    batch_size = 128

    chunk_size = 16
    len_feature_maps = 64
    num_channels = 3

    dataset = torchvision.datasets.ImageFolder(
        root=dirname,
        transform=transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    timutil.train2(dataloader, dirname, net, loss_fn, optimizer, epochs, device, do_display)

if __name__ == "__main__":
    main("alex-outpaint-128", 10, True)
