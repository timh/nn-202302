import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
from IPython import display
import torchvision.utils as vutils

class Trainer:
    def __init__(self, dirname: str, 
                 net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                 grid_num_rows = 1, grid_num_cols = 1):
        self.dirname = dirname
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grid_num_rows = grid_num_rows
        self.grid_num_cols = grid_num_cols
    
    def train(self, dataloader: DataLoader, epochs: int, device: str = "", do_display = True):
        self._num_batches = len(dataloader)
        self._num_data = len(dataloader.dataset)

        fake_images = None

        self._do_display = do_display
        self._setup_fig()

        self._loss_over_time = list()

        self._first_print_time = datetime.datetime.now()
        self._last_print_time = self._first_print_time
        self._last_print_step = 0
        global_step = 0

        inputs, expected = next(iter(dataloader))
        self._batch_size = len(inputs)

        for epoch in range(epochs):
            for batch, (inputs, expected) in enumerate(dataloader):
                if device:
                    inputs = inputs.to(device)
                    expected = expected.to(device)

                # gen outputs
                if epoch == 0 and batch == 0:
                    walk_modules(self.net, inputs)
                fake_outputs = self.net(inputs)

                # back prop
                loss = self.loss_fn(fake_outputs, expected)
                self.net.zero_grad()
                loss.backward()
                loss = loss.item()
                self._loss_over_time.append(loss)
                self.optimizer.step()

                self.maybe_print_status(epoch, epochs, global_step, batch, inputs, expected, fake_outputs, loss)

                global_step += 1

    def maybe_print_status(self, epoch: int, epochs: int, global_step: int, batch: int,
                           inputs: torch.Tensor, expected: torch.Tensor,
                           outputs: torch.Tensor,
                           loss: float):
        now = datetime.datetime.now()
        delta_last = now - self._last_print_time
        if delta_last >= datetime.timedelta(seconds=10) or (epoch == epochs and batch == self._num_batches):
            delta_first = now - self._first_print_time
            persec_first = global_step * self._batch_size / delta_first.total_seconds()
            persec_last = (global_step - self._last_print_step) * self._batch_size / delta_last.total_seconds()

            self.update_fig(epoch, expected, outputs)

            if self._do_display:
                self.display_fig()

            basename = f"{self.dirname}/epoch{epoch:05}-step{global_step:06}"

            print(f"epoch {epoch + 1}/{epochs}, batch {batch}/{self._num_batches} | loss {loss:.4f} | {persec_first:.4f} samples/sec")

            img_filename = f"{basename}.png"
            print(f"saving {img_filename}")
            self._fig.savefig(img_filename)

            net_filename = f"{basename}.torch"
            print(f"saving {net_filename}")
            torch.save(self.net, net_filename)

            self._last_print_time = now
            self._last_print_step = global_step

    def _setup_fig(self):
        fig = plt.figure(figsize=(15, 15))
        axes_loss = fig.add_subplot(self.grid_num_rows, self.grid_num_cols, (self.grid_num_cols + 1, self.grid_num_cols + 2))

        self._fig = fig
        self._axes_loss = axes_loss

    def update_fig(self, epoch: int, expected: torch.Tensor, outputs: torch.Tensor):
        self._axes_loss.clear()
        self._axes_loss.set_title("Loss")
        self._axes_loss.plot(self._loss_over_time, label='gen')
    
    def display_fig(self):
        display.clear_output(wait=True)
        display.display(self._fig)

class ImageTrainer(Trainer):
    grid_num_images: int = 64

    def __init__(self, dirname: str, net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, 
                 grid_num_rows = 2, grid_num_cols = 2,
                 grid_num_images = 64):
        super().__init__(dirname, net, loss_fn, optimizer, grid_num_rows, grid_num_cols)
        self.grid_num_images = grid_num_images

    def update_fig(self, epoch: int, expected: torch.Tensor, outputs: torch.Tensor):
        super().update_fig(epoch, expected, outputs)

        real_images = vutils.make_grid(expected[:self._grid_num_images], nrow=self._grid_rows, padding=2, normalize=True).cpu()

        fake_images = outputs.reshape(expected.shape).detach().cpu()
        fake_images = vutils.make_grid(fake_images[:self._grid_num_images], nrow=self._grid_rows, padding=2, normalize=True).cpu()

        self._axes_real.clear()
        self._axes_real.set_title("Real Images")
        self._axes_real.imshow(np.transpose(real_images, (1, 2, 0)))

        # Plot the fake images from the last epoch
        self._axes_fake.clear()
        self._axes_fake.set_title("Fake Images")
        self._axes_fake.imshow(np.transpose(fake_images, (1, 2, 0)))

    def _setup_fig(self):
        super()._setup_fig()

        axes_real = self._fig.add_subplot(self.grid_num_rows, self.grid_num_cols, 1)
        axes_real.set_axis_off()
        axes_fake = self._fig.add_subplot(self.grid_num_rows, self.grid_num_cols, 2)
        axes_fake.set_axis_off()

        self._axes_real = axes_real
        self._axes_fake = axes_fake

        self._grid_num_images = self.grid_num_images
        self._grid_rows = int(np.sqrt(self.grid_num_images))


def walk_modules(net: nn.Module, outputs: torch.Tensor):
    for idx, (key, mod) in enumerate(net._modules.items()):
        outputs = mod(outputs)
        print(f"{idx}: {outputs.shape}")
