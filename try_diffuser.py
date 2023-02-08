import math
import datetime
import sys

from typing import List, Tuple
import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from IPython import display

import train_gan

def main(dirname: str, epochs: int, do_plot: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 1.5e-4
    beta1 = 0.5
    batch_size = 64

    len_latent = 20
    image_size = 128
    feature_len = 16
    numchan = 3

    dataset = torchvision.datasets.ImageFolder(
        root=dirname,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = nn.Sequential(
        #                  in_channels,          kernel_size,
        #                  |        out_channels,|  stride,
        #                  |        |            |  |  padding)
        nn.ConvTranspose2d(numchan, feature_len, 5, 1, 2, bias=False),
        nn.BatchNorm2d(feature_len),
        nn.ReLU(True),

        nn.ConvTranspose2d(feature_len, feature_len * 2, 5, 1, 2, bias=False),
        nn.BatchNorm2d(feature_len * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(feature_len * 2, feature_len * 4, 5, 1, 2, bias=False),
        nn.BatchNorm2d(feature_len * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d(feature_len * 4, feature_len * 8, 5, 1, 2, bias=False),
        nn.BatchNorm2d(feature_len * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(feature_len * 8, numchan, 5, 1, 2, bias=False),
        nn.Tanh()
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    loss_fn = nn.MSELoss()

    inputs = np.random.randn(1, 3, 64, 64)
    inputs = torch.tensor(inputs, device=device, dtype=torch.float32)

    train(inputs, net, loss_fn, optimizer, dataloader, device, epochs)

def train(inputs: torch.tensor, net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader, device: str, epochs: int):
    maxval = torch.tensor(1.0 - 1.0e-7, device=device)

    overall_loss = 0.0
    global_step = 0
    last_print = datetime.datetime.now()
    first_print = last_print
    total_samples_processed = 0

    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    all_loss = list()
    batch_size = int(num_samples / num_batches + 0.5)

    print_every = 10
    grid_num_images = 16
    grid_rows = int(math.sqrt(grid_num_images))

    fig = plt.figure(figsize=(15,15))
    axes_img = fig.add_subplot(2, 2, 1)
    axes_img.set_axis_off()
    axes_loss = fig.add_subplot(2, 2, (3, 4))

    def show_images():
        images = outputs.detach().cpu()
        images = vutils.make_grid(images[:grid_num_images], nrow=grid_rows, padding=2, normalize=True)
        axes_img.clear()
        axes_img.imshow(np.transpose(images, (1, 2, 0)))

        axes_loss.clear()
        axes_loss.plot(all_loss, label="loss")

        display.clear_output(wait=True)
        display.display(fig)

        persec = total_samples_processed / float((now - first_print).seconds)
        print(f"epoch {epoch}/{epochs-1}, batch {batch}/{num_batches} | loss {epoch_loss / (batch + 1):.3f} | {persec:.3f} samples/sec")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, (expected, _unused) in enumerate(dataloader):
            expected = expected.to(device, dtype=torch.float32)

            noise = torch.tensor(2.0 * np.random.randn(*(expected[0].shape)), device=device, dtype=torch.float32)
            inputs = expected + noise
            inputs = torch.clamp(inputs, max=maxval)

            outputs = net(inputs)
            loss = loss_fn(outputs, expected)
            optimizer.zero_grad()
            loss.backward()
            loss = loss.item()
            epoch_loss += loss
            all_loss.append(loss)
            optimizer.step()

            total_samples_processed += len(expected)

            now = datetime.datetime.now()
            delta = now - last_print
            if delta >= datetime.timedelta(seconds=print_every):
                show_images()
                last_print = now

        # show_images()

if __name__ == "__main__":
    dirname = "alex-diffuser-128"
    epochs = 200
    main(dirname, epochs, False)
