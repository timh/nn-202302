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

    epochs = 5
    learning_rate = 1e-3
    batch_size = 100
    num_neurons = 512

    train_data = torchvision.datasets.mnist.MNIST("mnist-data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.mnist.MNIST("mnist-data", train=False, download=True, transform=torchvision.transforms.ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, num_neurons),
        nn.ReLU(),
        nn.Linear(num_neurons, num_neurons),
        nn.ReLU(),
        nn.Linear(num_neurons, 10),
    ).to(device)
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-2)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # train.
    loss_values, train_outputs_all = timutil.train(net, loss_fn, optimizer, train_dataloader, test_dataloader, epochs)
    train_outputs = train_outputs_all[-1]
