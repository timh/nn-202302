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

import train_gan

# custom weights initialization called on netG and netD
def weights_init(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

def main(do_plot: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 20
    learning_rate = 2e-4
    beta1 = 0.5
    batch_size = 128
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

    disc_net.apply(weights_init)
    gen_net.apply(weights_init)


    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    gen_optim = torch.optim.Adam(gen_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    disc_loss_fn = nn.BCELoss()
    gen_loss_fn = nn.BCELoss()

    gnet = train_gan.GanNetworks(disc_net, gen_net, disc_loss_fn, gen_loss_fn, disc_optim, gen_optim, len_latent, "celeb")

    train_gan.train_gan(gnet, dataloader, epochs, device, do_plot)

if __name__ == "__main__":
    main(False)
