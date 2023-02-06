from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

import timutil


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 10
    learning_rate = 2e-4
    beta1 = 0.5
    batch_size = 1000
    disc_neurons = 512
    gen_neurons = 512

    len_latent = 10
    len_gen_feature_maps = 64
    len_disc_feature_maps = 64
    image_size = 64
    num_channels = 3

    dataset = torchvision.datasets.ImageFolder(
        root="celeba",
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = DataLoader(dataset, batch_size=batch_size)

    disc_net = nn.Sequential(
        # input is (num_channels) x 64 x 64
        nn.Conv2d(num_channels, len_disc_feature_maps, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (len_disc_feature_maps) x 32 x 32
        nn.Conv2d(len_disc_feature_maps, len_disc_feature_maps * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (len_disc_feature_maps*2) x 16 x 16
        nn.Conv2d(len_disc_feature_maps * 2, len_disc_feature_maps * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (len_disc_feature_maps*4) x 8 x 8
        nn.Conv2d(len_disc_feature_maps * 4, len_disc_feature_maps * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (len_disc_feature_maps*8) x 4 x 4
        nn.Conv2d(len_disc_feature_maps * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()        
    ).to(device)

    gen_net = nn.Sequential(
        nn.ConvTranspose2d(len_latent, len_gen_feature_maps * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(len_gen_feature_maps * 8),
        nn.ReLU(True),
        # state size. (len_gen_feature_maps * 8) x 4 x 4
        nn.ConvTranspose2d(len_gen_feature_maps * 8, len_gen_feature_maps * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_gen_feature_maps * 4),
        nn.ReLU(True),
        # state size. (len_gen_feature_maps * 4) x 8 x 8
        nn.ConvTranspose2d(len_gen_feature_maps * 4, len_gen_feature_maps * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_gen_feature_maps * 2),
        nn.ReLU(True),
        # state size. (len_gen_feature_maps * 2) x 16 x 16
        nn.ConvTranspose2d(len_gen_feature_maps * 2, len_gen_feature_maps, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_gen_feature_maps),
        nn.ReLU(True),
        # state size. (len_gen_feature_maps) x 32 x 32
        nn.ConvTranspose2d(len_gen_feature_maps, num_channels, 4, 2, 1, bias=False),
        nn.Tanh()
    ).to(device)
    
    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    gen_optim = torch.optim.Adam(gen_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    disc_loss_fn = nn.BCELoss()
    gen_loss_fn = nn.BCELoss()

    timutil.train_gan(disc_net, gen_net, disc_loss_fn, gen_loss_fn, disc_optim, gen_optim, 
                      dataloader, epochs, len_latent, device)

if __name__ == "__main__":
    main()
