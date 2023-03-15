import sys
from typing import List, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F

sys.path.append("..")
import base_model
from experiment import Experiment
import train_util

# copy pasta:
#   https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

# if hidden_dims is None:
#     hidden_dims = [32, 64, 128, 256, 512]
class VanillaVAE(base_model.BaseModel):
    _metadata_fields = {'in_channels', 'latent_dim', 'hidden_dims', 'image_size'}
    _model_fields = _metadata_fields

    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List[int]) -> None:
        super(VanillaVAE, self).__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules: List[nn.Module] = []

        # Build Encoder
        out_size = image_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            out_size //= 2

        self.out_size = out_size
        self.encoder = nn.Sequential(*modules)
        lin_size = hidden_dims[-1] * out_size * out_size
        self.fc_mu = nn.Linear(lin_size, latent_dim)
        self.fc_var = nn.Linear(lin_size, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, lin_size)

        hidden_dims = hidden_dims.copy()
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
        # custom weights initialization called on netG and netD
        def weights_init(module: nn.Module):
            classname = type(module).__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
        self.apply(weights_init)

            
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.out_size, self.out_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        self.input = input
        self.mu = mu
        self.log_var = log_var
        out = self.decode(z)
        return out

    def loss_function(self) -> Tensor:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        input = self.input
        mu = self.mu
        log_var = self.log_var

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def get_loss_fn(exp: Experiment, kld_weight: float, recons_loss_type: str) -> Callable[[Tensor, Tensor], Tensor]:
    recons_loss_fn = train_util.get_loss_fn(recons_loss_type)
    def fn(output: List[any], truth: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        net: VanillaVAE = exp.net
        kld_loss = net.loss_function()
        recons_loss = recons_loss_fn(output, truth)
        loss = recons_loss + kld_weight * kld_loss
        # print(f"loss: kld={kld_loss:.5f} {recons_loss_type}={recons_loss:.5f}: result={loss:.5f}")
        return loss
    return fn


