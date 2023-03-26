from typing import Literal, Tuple, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

BetaSchedType = Literal['cosine', 'linear', 'quadratic', 'sigmoid']
DEFAULT_TIMESTEPS = 300

# from The Annotated Diffusion Model
#  https://huggingface.co/blog/annotated-diffusion

NoiseWithAmountFn = Callable[[Tuple], Tuple[Tensor, Tensor]]

def make_noise_fn(type: BetaSchedType, timesteps: int, 
                  backing_type: Literal['rand', 'normal']) -> NoiseWithAmountFn:
    betas = make_betas(type=type, timesteps=timesteps)

    # also from annotated-diffusion blog post
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)                       # image_mult @ t
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # image_mult @ t-1
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    if backing_type == 'rand':
        backing_fn = noise_rand
    elif backing_type == 'normal':
        backing_fn = noise_normal
    else:
        raise ValueError(f"unknown {backing_type=}")

    def fn(size: Tuple) -> Tuple[Tensor, Tensor]:
        timestep = torch.randint(0, timesteps, size=(1,))[0]
        amount = sqrt_one_minus_alphas_cumprod[timestep]
        noise = backing_fn(size) * amount

        return noise, amount

    return fn

def make_betas(type: BetaSchedType,
               timesteps: int) -> Tensor:
    types = {
        'cosine': cosine_beta_schedule,
        'linear': linear_beta_schedule,
        'quadratic': quadratic_beta_schedule,
        'sigmoid': sigmoid_beta_schedule
    }
    betas = types[type](timesteps)
    return betas

# all schedules return:
#   (timesteps,)
def cosine_beta_schedule(timesteps: int, s=0.008) -> Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def noise_rand(size: Tuple) -> Tensor:
    return torch.rand(size=size)

def noise_normal(size: Tuple) -> Tensor:
    return torch.normal(mean=0, std=0.5, size=size)


