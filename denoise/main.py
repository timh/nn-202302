# %%
import sys
import importlib
import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
from experiment import Experiment
import noised_data, model
from noised_data import NoisedDataset
from model import ConvDesc, Denoiser

importlib.reload(trainer)
importlib.reload(noised_data)
importlib.reload(model)

class Logger(trainer.TrainerLogger):
    def __init__(self):
        self.axes_input = plt.subplot(1, 3, 1)
        self.axes_output = plt.subplot(1, 3, 2)
        self.axes_truth = plt.subplot(1, 3, 3)

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        super().print_status(exp, epoch, batch, batches, train_loss)
    
        input0 = exp.last_train_in[0]
        out0 = exp.last_train_out[0]
        truth0 = exp.last_train_truth[0]

        input0 = np.transpose(input0.detach().cpu(), (1, 2, 0))
        out0 = np.transpose(out0.detach().cpu(), (1, 2, 0))
        truth0 = np.transpose(truth0.detach().cpu(), (1, 2, 0))

        self.axes_input.imshow(input0)
        self.axes_output.imshow(out0)
        self.axes_truth.imshow(truth0)

        print(f"{torch.mean(out0)=}, {torch.std(out0)=}")

        display.display(plt.gcf())

if __name__ == "__main__":
    device = "cuda"
    image_size = 128
    batch_size = 4
    epochs = 1000

    base_dataset = torchvision.datasets.ImageFolder(
        root="alex-many-128",
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    noised_data = NoisedDataset(base_dataset=base_dataset)
    cutoff = int(len(noised_data) * 0.9)

    train_data = noised_data[:cutoff]
    val_data = noised_data[cutoff:]

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    #--
    if False:
        input0, truth0 = next(iter(train_dl))
        input0, truth0 = input0[0], truth0[0]
        
        # truth0 = torch.reshape(input0, (128, 128, 3))
        truth0 = np.transpose(truth0, (1, 2, 0))
        plt.imshow(truth0.detach().cpu())
        sys.exit(0)
    #--

    descs1 = [
        ConvDesc(channels=64, kernel_size=5),
        ConvDesc(channels=64, kernel_size=5),
        ConvDesc(channels=64, kernel_size=5),
    ]
    net1 = Denoiser(descs1, device=device)
    loss_fn = nn.MSELoss()

    exps = [
        Experiment(label="1", net=net1, loss_fn=loss_fn, train_dataloader=train_dl, val_dataloader=val_dl, epochs=epochs)
    ]
    tcfg = trainer.TrainerConfig(exps, len(exps), model.get_optim_fn)
    t = trainer.Trainer(logger=Logger())
    t.train(tcfg, device=device)

# %%


