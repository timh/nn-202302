from typing import List, Tuple
import datetime

import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader

import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torchvision.utils as vutils

def array_str(array: torch.Tensor) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def train(network: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader, test_dataloader: DataLoader,
          epochs: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    num_batches = len(train_dataloader)
    num_data = len(train_dataloader.dataset)
    batch_size = int(num_data / num_batches)

    loss_all = torch.zeros((epochs, num_batches))
    outputs_all: List[torch.Tensor] = list()

    first_print_time = datetime.datetime.now()
    last_print_time = first_print_time
    last_print_step = 0
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch, (inputs, expected) in enumerate(train_dataloader):
            now = datetime.datetime.now()
            outputs = network(inputs)

            loss = loss_fn(outputs, expected)
            optimizer.zero_grad()
            loss.backward()
            loss = loss.item()
            optimizer.step()

            loss_all[epoch][batch] = loss
            outputs_all.append(outputs)

            epoch_acc += (outputs.argmax(1) == expected).type(torch.float).sum().item()
            epoch_loss += loss

            delta_last = now - last_print_time
            if delta_last >= datetime.timedelta(seconds=1):
                delta_first = now - first_print_time
                persec_first = global_step / delta_first.total_seconds()
                persec_last = (global_step - last_print_step) / delta_last.total_seconds()
                last_print_time = now
                last_print_step = global_step
                data_idx = batch * batch_size
                print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | loss {loss:.4f} | {persec_last:.4f} steps/sec, {persec_first:.4f} steps/sec overall")
            
            global_step += 1
        
        train_loss = epoch_loss / num_batches
        train_acc = epoch_acc / num_data
        test_loss, test_acc = test(network, loss_fn, test_dataloader)
        print(f"epoch {epoch + 1}/{epochs}:")
        print(f"    train loss = {train_loss:.4f}  acc = {train_acc:.4f}")
        print(f"     test loss = {test_loss:.4f}  acc = {test_acc:.4f}")

    return loss_all, outputs_all

def test(network: nn.Module, loss_fn: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    # returns loss, accuracy
    num_batches = len(dataloader)
    num_data = len(dataloader.dataset)

    loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for inputs, expected in dataloader:
            outputs = network(inputs)
            loss += loss_fn(outputs, expected)
            accuracy += (outputs.argmax(1) == expected).type(torch.float).sum().item()
        
        loss /= num_batches
        accuracy /= num_data
    
    return (loss, accuracy)

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

    fake_images = None

    grid_num_images = 64
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
        axes_loss.plot(loss_over_time, label='gen')

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

