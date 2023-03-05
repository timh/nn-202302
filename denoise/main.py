# %%
import sys
import importlib
import datetime
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

class Logger(trainer.TensorboardLogger):
    def __init__(self, train_data: NoisedDataset):
        super().__init__("denoise")

        self.train_data = train_data

        nrows = 2
        ncols = 5
        base_dim = 6
        plt.gcf().set_figwidth(base_dim * ncols)
        plt.gcf().set_figheight(base_dim * nrows)

        self.axes_input = plt.subplot(nrows, ncols, 1, title="input (src+noise)")
        self.axes_output = plt.subplot(nrows, ncols, 2, title="output (pred. noise)")
        self.axes_truth = plt.subplot(nrows, ncols, 3, title="truth (actual noise)")
        self.axes_src = plt.subplot(nrows, ncols, 4, title="source image")
        self.axes_in_sub_out = plt.subplot(nrows, ncols, 5, title="input - output")

        self.axes_gen1 = plt.subplot(nrows, ncols, 6, title="1 step")
        self.axes_gen2 = plt.subplot(nrows, ncols, 7, title="2 steps")
        self.axes_gen5 = plt.subplot(nrows, ncols, 8, title="5 steps")
        self.axes_gen10 = plt.subplot(nrows, ncols, 9, title="10 steps")
        self.last_val_loss = None

    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            filename = f"{self.dirname}-vloss_{val_loss:.3f}.torch"
            state_dict = exp.net.state_dict()
            with open(filename, "wb") as torchfile:
                torch.save(state_dict, torchfile)
            print(f"  saved to {filename}")

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        super().print_status(exp, epoch, batch, batches, train_loss)

        sidx = torch.randint(0, len(self.train_data), (1,))[0].item()
        input, truth = self.train_data[sidx]      # input = noised source image, truth = noise added
        src = self.train_data.dataset[sidx][0]    # src = source image
        exp.net.eval()
        with torch.no_grad():
            input = input.to(device)
            out = exp.net(input.unsqueeze(0))     # prediction of noise
            out = out[0]
        chan, width, height = input.shape

        transpose = lambda t: np.transpose(t.detach().cpu(), (1, 2, 0))

        self.axes_input.imshow(transpose(input))
        self.axes_output.imshow(transpose(out))
        self.axes_truth.imshow(transpose(truth))
        self.axes_src.imshow(transpose(src))
        self.axes_in_sub_out.imshow(transpose(input - out))

        noisein = torch.rand((1, 3, width, width), device=device)
        gen1 = model.generate(exp, 1, width, input=noisein, device=device)[0]
        gen2 = model.generate(exp, 2, width, input=noisein, device=device)[0]
        gen5 = model.generate(exp, 5, width, input=noisein, device=device)[0]
        gen10 = model.generate(exp, 10, width, input=noisein, device=device)[0]
        noisein = noisein[0]
        self.axes_gen1.imshow(transpose(noisein - gen1))
        self.axes_gen2.imshow(transpose(noisein - gen2))
        self.axes_gen5.imshow(transpose(noisein - gen5))
        self.axes_gen10.imshow(transpose(noisein - gen10))

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
    t = trainer.Trainer(logger=Logger(train_data))
    t.train(tcfg, device=device)

# %%


