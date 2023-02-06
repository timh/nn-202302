import datetime

import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.figure
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torchvision.utils as vutils

class GanNetworks:
    name: str

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
                 basename: str, name: str = "" 
                 ):
        self.disc_net, self.gen_net = disc_net, gen_net
        self.disc_loss_fn, self.gen_loss_fn = disc_loss_fn, gen_loss_fn
        self.disc_optim, self.gen_optim = disc_optim, gen_optim
        self.len_latent = len_latent

        if not name:
            name = "-".join(["gan", basename, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S")])
        self.name = name
        print(f"name {name}")
    
    def save(self, epoch: int):
        basename = f"{self.name}-epoch{epoch:02}"
        gen_filename = f"{basename}-gen.torch"
        disc_filename = f"{basename}-disc.torch"

        print(f"saving {gen_filename}")
        torch.save(self.gen_net, open(gen_filename, "wb"))

        print(f"saving {disc_filename}")
        torch.save(self.disc_net, open(disc_filename, "wb"))

def train_gan(gnet: GanNetworks, 
              real_dataloader: DataLoader, epochs: int,
              device: str,
              do_plot = False):

    num_batches = len(real_dataloader)
    num_data = len(real_dataloader.dataset)

    first_print_time = datetime.datetime.now()
    last_print_time = first_print_time
    last_print_step = 0
    global_step = 0

    real_label = 1.
    fake_label = 0.

    if do_plot:
        fig = plt.figure(figsize=(15,15))
        axes_real = plt.subplot(1,2,1)
        axes_real.set_axis_off()
        axes_real.set_title("Real Images")
        axes_fake = plt.subplot(1,2,2)
        axes_fake.set_axis_off()
        axes_fake.set_title("Fake Images")

    fake_images = None

    real_batch = next(iter(real_dataloader))
    real_images = vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu()

    def show_images(epoch: int):
        if not do_plot:
            return
        # Plot the real images
        axes_real.imshow(np.transpose(real_images, (1,2,0)))

        # Plot the fake images from the last epoch
        axes_fake.imshow(np.transpose(fake_images, (1,2,0)))

        #fig.canvas.draw()
        display.display(fig)
        display.clear_output(wait=True)

    for epoch in range(epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        
        for batch, (real_inputs, _real_expected) in enumerate(real_dataloader):
            now = datetime.datetime.now()
            batch_size = len(real_inputs)
            
            if device != "cpu":
                real_inputs = real_inputs.to(device)

            num_samples = len(real_inputs)

            # run real outputs through D. expect ones.
            disc_outputs_4real = gnet.disc_net(real_inputs).view(-1)
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

            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss

            delta_last = now - last_print_time
            if delta_last >= datetime.timedelta(seconds=30):
                delta_first = now - first_print_time
                persec_first = global_step * batch_size / delta_first.total_seconds()
                persec_last = (global_step - last_print_step) * batch_size / delta_last.total_seconds()
                last_print_time = now
                last_print_step = global_step
                data_idx = batch * batch_size
                print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | gen_loss {gen_loss:.4f}, disc_loss {disc_loss:.4f} | {persec_first:.4f} samples/sec")

                fake_images = fake_outputs.reshape(real_inputs.shape).detach().cpu()
                fake_images = vutils.make_grid(fake_images[:64], padding=2, normalize=True)

                show_images(epoch)
            
            global_step += 1

        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches

        print(f"epoch {epoch + 1}/{epochs}:")
        print(f"     gen loss = {epoch_gen_loss:.4f}")
        print(f"    disc loss = {epoch_disc_loss:.4f}")
        gnet.save(epoch)

    show_images(epochs)