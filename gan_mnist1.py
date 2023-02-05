from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

import timutil

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 1
    learning_rate = 2e-4
    beta1 = 0.5
    batch_size = 200
    disc_neurons = 512
    gen_neurons = 512

    train_data = torchvision.datasets.mnist.MNIST("mnist-data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    disc_net = nn.Sequential(
        nn.Linear(28 * 28, disc_neurons),
        nn.ReLU(),
        nn.Linear(disc_neurons, disc_neurons),
        nn.ReLU(),
        nn.Linear(disc_neurons, 10),
    ).to(device)

    gen_net = nn.Sequential(
        nn.Linear(10, gen_neurons),
        nn.ReLU(),
        nn.Linear(gen_neurons, gen_neurons),
        nn.ReLU(),
        nn.Linear(gen_neurons, 28 * 28),
    ).to(device)
    
    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    gen_optim = torch.optim.Adam(gen_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    disc_loss_fn = nn.CrossEntropyLoss()
    gen_loss_fn = nn.MSELoss()

    timutil.train_gan(disc_net, gen_net, disc_loss_fn, gen_loss_fn, disc_optim, gen_optim, train_dataloader, epochs)

