import math
import datetime
import sys
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

def main(dirname: str, epochs: int, do_plot: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 2e-4
    beta1 = 0.5
    batch_size = 16

    len_latent = 100
    smaller_image_size = 64
    len_gen_feature_maps = 128
    len_disc_feature_maps = 64
    larger_image_size = 128
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

    disc_net = nn.Sequential(
        # input is (num_channels) x 128 x 128
        nn.Conv2d(num_channels, len_disc_feature_maps, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(len_disc_feature_maps, len_disc_feature_maps * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(len_disc_feature_maps * 2, len_disc_feature_maps * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(len_disc_feature_maps * 4, len_disc_feature_maps * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(len_disc_feature_maps * 8, len_disc_feature_maps * 16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(len_disc_feature_maps * 16),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(len_disc_feature_maps * 16, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()        

        # # state size. (len_disc_feature_maps*8) x 4 x 4
        # nn.Conv2d(len_disc_feature_maps * 8, 1, 4, 1, 0, bias=False),
        # nn.Sigmoid()        
    ).to(device)

    gen_net = nn.Sequential(
        #                  in_channels,                          kernel_size,
        #                  |             out_channels,             |  stride,
        #                  |             |                         |  |  padding)
        nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(True),

        nn.ConvTranspose2d(num_channels, num_channels, 4, 1, 1, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(True),

        nn.ConvTranspose2d(num_channels, num_channels, 4, 1, 1, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(True),

        nn.ConvTranspose2d(num_channels, num_channels, 4, 1, 2, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(True),

        nn.ConvTranspose2d(num_channels, num_channels, 4, 1, 2, bias=False),
        nn.Tanh()
    ).to(device)

    disc_net.apply(weights_init)
    gen_net.apply(weights_init)


    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    gen_optim = torch.optim.Adam(gen_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    disc_loss_fn = nn.BCELoss()
    gen_loss_fn = nn.BCELoss()

    gnet = train_gan.GanNetworks(disc_net, gen_net, disc_loss_fn, gen_loss_fn, disc_optim, gen_optim, len_latent, dirname)

    train(gnet, dataloader, epochs, device, do_plot)

def train(gnet: train_gan.GanNetworks, 
              real_dataloader: DataLoader, epochs: int,
              device: str,
              do_display = False):
    num_batches = len(real_dataloader)
    num_data = len(real_dataloader.dataset)

    report_every = 10

    real_label = 1.
    fake_label = 0.

    fig = plt.figure(figsize=(15,15))
    axes_real = fig.add_subplot(2, 2, 1)
    axes_real.set_axis_off()
    axes_fake = fig.add_subplot(2, 2, 2)
    axes_fake.set_axis_off()
    axes_loss_gen = fig.add_subplot(2, 2, (3, 4))
    axes_loss_disc = axes_loss_gen.twinx()
    color_gen = "#ff0000"
    color_disc = "#000000"

    # hfig = display.display(fig, display_id=True)

    fake_images = None

    grid_num_images = 16
    grid_rows = int(np.sqrt(grid_num_images))

    gen_loss_over_time = list()
    disc_loss_over_time = list()

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

        axes_loss_gen.clear()
        axes_loss_gen.set_title("Loss")
        axes_loss_gen.set_ylabel("gen", color=color_gen)
        axes_loss_gen.plot(gen_loss_over_time, label='gen', color=color_gen)
        axes_loss_disc.plot(disc_loss_over_time, label='disc', color=color_disc)
        axes_loss_disc.set_ylabel("disc", color=color_disc)

        if do_display:
            # fig.canvas.draw()
            # hfig.update(fig)
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

            print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | gen_loss {gen_loss:.4f}, disc_loss {disc_loss:.4f} | {persec_first:.4f} samples/sec")
            gnet.save_image(epoch, global_step, fig)
            gnet.save_models(epoch)

    for epoch in range(epochs):
        for batch, (small_inputs, large_expected) in enumerate(real_dataloader):
            now = datetime.datetime.now()
            batch_size = len(small_inputs)

            if device != "cpu":
                small_inputs = small_inputs.to(device)
                large_expected = large_expected.to(device)

            # run real outputs through D. expect ones.
            disc_outputs_4real = gnet.disc_net(large_expected)
            disc_outputs_4real = disc_outputs_4real.view(-1)
            disc_expected_4real = torch.full((batch_size,), real_label, device=device)
            disc_loss_4real = gnet.disc_loss_fn(disc_outputs_4real, disc_expected_4real)
            gnet.disc_net.zero_grad()
            disc_loss_4real.backward(retain_graph=True)
            disc_loss_4real = disc_loss_4real.item()
            gnet.disc_optim.step()

            # gen fake outputs
            # fake_outputs = gen_net(real_onehot)
            # walk_modules(gnet.gen_net, small_inputs)
            fake_outputs = gnet.gen_net(small_inputs)

            # train D: run fake outputs through D. expect zeros.
            disc_outputs_4fake = gnet.disc_net(fake_outputs)
            disc_outputs_4fake = disc_outputs_4fake.view(-1)
            disc_expected_4fake = torch.full((batch_size,), fake_label, device=device)
            disc_loss_4fake = gnet.disc_loss_fn(disc_outputs_4fake, disc_expected_4fake)
            gnet.disc_net.zero_grad()
            disc_loss_4fake.backward(retain_graph=True)
            disc_loss_4fake = disc_loss_4fake.item()
            gnet.disc_optim.step()

            disc_loss = (disc_loss_4fake + disc_loss_4real) / 2.0

            # now do backprop on generator. expected answer is that it
            # fools the discriminator and results in the real answer. 
            # 
            # regenerate the discriminator outputs for fake data, cuz we've 
            # updated weights in it.
            disc_outputs_4fake = gnet.disc_net(fake_outputs).view(-1)
            gen_expected = torch.full((batch_size,), real_label, device=device)
            gen_loss = gnet.gen_loss_fn(disc_outputs_4fake, gen_expected)
            gnet.gen_net.zero_grad()
            gen_loss.backward()
            gen_loss = gen_loss.item()
            gnet.gen_optim.step()

            gen_loss_over_time.append(gen_loss)
            disc_loss_over_time.append(disc_loss)

            maybe_print_status(epoch)

            global_step += 1

        maybe_print_status(epoch)

    show_images(epochs)

def walk_modules(net, out):
    for idx, (key, mod) in enumerate(net._modules.items()):
        out = mod(out)
        print(f"{idx}: {out.shape}")

if __name__ == "__main__":
    main("alex-resize-128", 10, True)
