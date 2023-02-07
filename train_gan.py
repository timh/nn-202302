import datetime

import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader
import os

import matplotlib.figure
import matplotlib.pyplot as plt
import IPython
from IPython import display
import numpy as np
import torchvision.utils as vutils

class GanNetworks:
    dirname: str

    disc_net: nn.Module
    gen_net: nn.Module

    disc_loss_fn: nn.Module
    gen_loss_fn: nn.Module

    disc_optim: torch.optim.Optimizer
    gen_optim: torch.optim.Optimizer

    len_latent: int

    def __init__(self,
                 disc_net: nn.Module, gen_net: nn.Module,
                 disc_loss_fn: nn.Module, gen_loss_fn: nn.Module,
                 disc_optim: torch.optim.Optimizer, gen_optim: torch.optim.Optimizer,
                 len_latent: int,
                 basename: str, dirname: str = "" 
                 ):
        self.disc_net, self.gen_net = disc_net, gen_net
        self.disc_loss_fn, self.gen_loss_fn = disc_loss_fn, gen_loss_fn
        self.disc_optim, self.gen_optim = disc_optim, gen_optim
        self.len_latent = len_latent

        if not dirname:
            dirname = "outputs/" + "-".join(["gan", basename, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S")])
            os.makedirs(dirname, exist_ok=True)

        self.dirname = dirname
    
    def save_models(self, epoch: int):
        gen_filename = f"{self.dirname}/net-gen.torch"
        disc_filename = f"{self.dirname}/net-disc.torch"

        print(f"saving {gen_filename}")
        torch.save(self.gen_net, open(gen_filename, "wb"))

        print(f"saving {disc_filename}")
        torch.save(self.disc_net, open(disc_filename, "wb"))
    
    def save_image(self, epoch: int, global_step: int, fig: matplotlib.figure.Figure):
        filename = f"{self.dirname}/epoch{epoch:02}-step{global_step:07}.png"
        fig.savefig(filename)
        print(f"saved image {filename}")

def train_gan(gnet: GanNetworks, 
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

    grid_num_images = 36
    grid_rows = int(np.sqrt(grid_num_images))

    gen_loss_over_time = list()
    disc_loss_over_time = list()

    def show_images(epoch: int):
        real_images = vutils.make_grid(real_inputs[:grid_num_images], nrow=grid_rows, padding=2, normalize=True).cpu()

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

            fake_images = fake_outputs.reshape(real_inputs.shape).detach().cpu()
            fake_images = vutils.make_grid(fake_images[:grid_num_images], nrow=grid_rows, padding=2, normalize=True)
            show_images(epoch)

            print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | gen_loss {gen_loss:.4f}, disc_loss {disc_loss:.4f} | {persec_first:.4f} samples/sec")
            gnet.save_image(epoch, global_step, fig)
            gnet.save_models(epoch)

    for epoch in range(epochs):
        for batch, (real_inputs, _real_expected) in enumerate(real_dataloader):
            now = datetime.datetime.now()
            batch_size = len(real_inputs)
            
            if device != "cpu":
                real_inputs = real_inputs.to(device)

            num_samples = len(real_inputs)

            # run real outputs through D. expect ones.
            disc_outputs_4real = gnet.disc_net(real_inputs)
            disc_outputs_4real = disc_outputs_4real.view(-1)
            disc_expected_4real = torch.full((batch_size,), real_label, device=device)
            disc_loss_4real = gnet.disc_loss_fn(disc_outputs_4real, disc_expected_4real)
            gnet.disc_net.zero_grad()
            disc_loss_4real.backward(retain_graph=True)
            disc_loss_4real = disc_loss_4real.item()
            gnet.disc_optim.step()

            # gen fake outputs
            # fake_outputs = gen_net(real_onehot)
            fake_inputs = torch.randn(batch_size, gnet.len_latent, 1, 1, device=device)
            fake_outputs = gnet.gen_net(fake_inputs)

            # train D: run fake outputs through D. expect zeros.
            disc_outputs_4fake = gnet.disc_net(fake_outputs).view(-1)
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
