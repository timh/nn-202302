import math
import datetime
import sys
import os
import random

from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torchvision.utils as vutils

import train_gan

class BigSmallDataLoader(DataLoader):
    _smaller_size: int
    _iter: any

    def __init__(self, dataset, smaller_size: int, batch_size: int, shuffle: bool):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self._smaller_size = smaller_size
    
    def __iter__(self):
        self._iter = super().__iter__()
        return self
    
    def __next__(self) -> any:
        biginput, _bigexpect = self._iter.__next__()
        smallinput = transforms.Resize(self._smaller_size)(biginput)
        return smallinput, biginput

# custom weights initialization called on netG and netD
def weights_init(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

def main(dirname: str, epochs: int, do_display: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 1e-3
    # beta1 = 0.5
    batch_size = 128

    len_latent = 100
    resize_multiple = 2
    larger_image_size = 128
    smaller_image_size = int(larger_image_size / resize_multiple)
    len_feature_maps = 64
    num_channels = 3

    dataset = torchvision.datasets.ImageFolder(
        root=dirname,
        transform=transforms.Compose([
            transforms.Resize(larger_image_size),
            transforms.CenterCrop(larger_image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = BigSmallDataLoader(dataset, smaller_image_size, batch_size=batch_size, shuffle=True)

    net = nn.Sequential(
        #                  in_channels,                    kernel_size,
        #                  |             out_channels,     |  stride,
        #                  |             |                 |  |                padding)
        nn.ConvTranspose2d(num_channels, len_feature_maps, 4, resize_multiple, 1, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 4, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 4, 1, 2, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, len_feature_maps, 4, 1, 1, bias=False),
        nn.BatchNorm2d(len_feature_maps),
        nn.ReLU(True),

        nn.ConvTranspose2d(len_feature_maps, num_channels, 4, 1, 1, bias=False),
        nn.Tanh()
    ).to(device)

    net.apply(weights_init)

    dirname = "outputs/" + "-".join([dirname, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S")])
    os.makedirs(dirname, exist_ok=True)

    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train2(dataloader, dirname, net, loss_fn, optimizer, epochs, device, do_display)

def train2(dataloader: DataLoader,
           dirname: str,
           net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, epochs: int, device: str, do_display = False):
    num_batches = len(dataloader)
    num_data = len(dataloader.dataset)

    fig = plt.figure(figsize=(15,15))
    axes_real = fig.add_subplot(2, 2, 1)
    axes_real.set_axis_off()
    axes_fake = fig.add_subplot(2, 2, 2)
    axes_fake.set_axis_off()
    axes_loss = fig.add_subplot(2, 2, (3, 4))
    color_gen = "#ff0000"
    color_disc = "#000000"

    fake_images = None

    grid_num_images = 16
    grid_rows = int(np.sqrt(grid_num_images))

    loss_over_time = list()

    def show_images(epoch: int):
        real_images = vutils.make_grid(large_expected[:grid_num_images], nrow=grid_rows, padding=2, normalize=True).cpu()

        # Plot the real images
        axes_real.clear()
        axes_real.set_title("Real Images")
        axes_real.imshow(np.transpose(real_images, (1,2,0)))

        # Plot the fake images from the last epoch
        axes_fake.clear()
        axes_fake.set_title("Fake Images")
        axes_fake.imshow(np.transpose(fake_images, (1,2,0)))

        axes_loss.clear()
        axes_loss.set_title("Loss")
        axes_loss.plot(loss_over_time, label='gen', color=color_gen)

        if do_display:
            display.clear_output(wait=True)
            display.display(fig)

    first_print_time = datetime.datetime.now()
    last_print_time = first_print_time
    last_print_step = 0
    global_step = 0

    def maybe_print_status(epoch: int):
        nonlocal first_print_time, last_print_time, last_print_step, fake_images, epochs

        now = datetime.datetime.now()
        delta_last = now - last_print_time
        if delta_last >= datetime.timedelta(seconds=10) or epoch == epochs:
            delta_first = now - first_print_time
            persec_first = global_step * batch_size / delta_first.total_seconds()
            persec_last = (global_step - last_print_step) * batch_size / delta_last.total_seconds()
            last_print_time = now
            last_print_step = global_step
            data_idx = batch * batch_size

            fake_images = fake_outputs.reshape(large_expected.shape).detach().cpu()
            fake_images = vutils.make_grid(fake_images[:grid_num_images], nrow=grid_rows, padding=2, normalize=True)
            show_images(epoch)

            basename = f"{dirname}/epoch{epoch:05}-step{global_step:06}"

            print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | loss {loss:.4f} | {persec_first:.4f} samples/sec")
            img_filename = f"{basename}.png"
            net_filename = f"{basename}.torch"
            print(f"saving {img_filename}")
            fig.savefig(img_filename)
            print(f"saving {net_filename}")
            torch.save(net, net_filename)

    for epoch in range(epochs):
        for batch, (small_inputs, large_expected) in enumerate(dataloader):
            now = datetime.datetime.now()
            batch_size = len(small_inputs)

            if device != "cpu":
                small_inputs = small_inputs.to(device)
                large_expected = large_expected.to(device)

            # gen outputs
            if epoch == 0 and batch == 0:
                walk_modules(net, small_inputs)
            fake_outputs = net(small_inputs)

            # back prop
            loss = loss_fn(fake_outputs, large_expected)
            net.zero_grad()
            loss.backward()
            loss = loss.item()
            optimizer.step()

            loss_over_time.append(loss)
            maybe_print_status(epoch)

            global_step += 1

        maybe_print_status(epoch)

def walk_modules(net, out):
    for idx, (key, mod) in enumerate(net._modules.items()):
        out = mod(out)
        print(f"{idx}: {out.shape}")

if __name__ == "__main__":
    main("alex-resize-128", 10, True)
