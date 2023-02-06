from typing import List, Tuple
import torch, torch.cuda
from torch import nn
from torch.utils.data import DataLoader
import datetime

import matplotlib.pyplot as plt
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

def train_gan(disc_net: nn.Module, gen_net: nn.Module, 
              disc_loss_fn: nn.Module, gen_loss_fn: nn.Module,
              disc_optim: torch.optim.Optimizer, gen_optim: torch.optim.Optimizer, 
              real_dataloader: DataLoader, epochs: int,
              len_latent: int,
              device: str):

    num_batches = len(real_dataloader)
    num_data = len(real_dataloader.dataset)
    batch_size = 0

    first_print_time = datetime.datetime.now()
    last_print_time = first_print_time
    last_print_step = 0
    global_step = 0

    real_label = 1.
    fake_label = 0.

    img_list = list()

    for epoch in range(epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        
        for batch, (real_inputs, _real_expected) in enumerate(real_dataloader):
            now = datetime.datetime.now()
            if batch_size == 0:
                batch_size = len(real_inputs)
            
            if device != "cpu":
                real_inputs = real_inputs.to(device)

            num_samples = len(real_inputs)

            # run real outputs through D. expect ones.
            disc_outputs_4real = disc_net(real_inputs).view(-1)
            disc_expected_4real = torch.full((batch_size,), real_label, device=device)
            disc_loss_4real = disc_loss_fn(disc_outputs_4real, disc_expected_4real)
            disc_net.zero_grad()
            disc_loss_4real.backward()
            disc_loss_4real = disc_loss_4real.item()
            disc_optim.step()

            # gen fake outputs
            # fake_outputs = gen_net(real_onehot)
            fake_inputs = torch.randn(batch_size, len_latent, 1, 1, device=device)
            fake_outputs = gen_net(fake_inputs)

            # train D: run fake outputs through D. expect zeros.
            disc_outputs_4fake = disc_net(fake_outputs).view(-1)
            disc_expected_4fake = torch.full((batch_size,), fake_label, device=device)
            disc_loss_4fake = disc_loss_fn(disc_outputs_4fake, disc_expected_4fake)
            disc_net.zero_grad()
            disc_loss_4fake.backward(retain_graph=True)
            disc_loss_4fake = disc_loss_4fake.item()
            disc_optim.step()

            disc_loss = (disc_loss_4fake + disc_loss_4real) / 2.0

            # now do backprop on generator. expected answer is that it
            # fools the discriminator and results in the real answer. 
            # 
            # regenerate the discriminator outputs for fake data, cuz we've 
            # updated weights in it.
            disc_outputs_4fake = disc_net(fake_outputs).view(-1)
            gen_expected = torch.full((batch_size,), real_label, device=device)
            gen_loss = gen_loss_fn(disc_outputs_4fake, gen_expected)
            gen_net.zero_grad()
            gen_loss.backward()
            gen_loss = gen_loss.item()
            gen_optim.step()

            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss

            delta_last = now - last_print_time
            if delta_last >= datetime.timedelta(seconds=5):
                delta_first = now - first_print_time
                persec_first = global_step * batch_size / delta_first.total_seconds()
                persec_last = (global_step - last_print_step) * batch_size / delta_last.total_seconds()
                last_print_time = now
                last_print_step = global_step
                data_idx = batch * batch_size
                print(f"epoch {epoch + 1}/{epochs}, data {data_idx}/{num_data} | gen_loss {gen_loss:.4f}, disc_loss {disc_loss:.4f} | {persec_first:.4f} samples/sec")

                fake_images = fake_outputs.reshape(real_inputs.shape).detach().cpu()
                img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))
            
            global_step += 1

        epoch_gen_loss /= num_batches

        epoch_disc_loss /= num_batches

        print(f"epoch {epoch + 1}/{epochs}:")
        print(f"     gen loss = {epoch_gen_loss:.4f}")
        print(f"    disc loss = {epoch_disc_loss:.4f}")

    real_batch = next(iter(real_dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

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


