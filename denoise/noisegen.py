from typing import Literal, Tuple, Callable

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import einops

BetaSchedType = Literal['cosine', 'linear', 'quadratic', 'sigmoid']
# DEFAULT_TIMESTEPS = 300

# from The Annotated Diffusion Model
#  https://huggingface.co/blog/annotated-diffusion

NoiseWithAmountFn = Callable[[Tuple], Tuple[Tensor, Tensor]]

class NoiseSchedule:
    timesteps: int

    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas: Tensor
    posterior_variance: Tensor
    noise_fn: Callable[[Tuple], Tensor]

    def __init__(self, betas: Tensor, timesteps: int, noise_fn: Callable[[Tuple], Tensor]):
        # also from annotated-diffusion blog post
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)                       # image_mult @ t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # image_mult @ t-1
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # TODO: these all need to be renamed.
        self.betas = betas
        self.timesteps = timesteps
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = sqrt_recip_alphas
        self.posterior_variance = posterior_variance
        self.noise_fn = noise_fn
    
    def _add_dims(self, tensor_0d: Tensor) -> Tensor:
        assert len(tensor_0d.shape) == 0
        return einops.rearrange(tensor_0d, "-> 1 1 1")
    
    def noise(self, size: Tuple, timestep: int = None) -> Tuple[Tensor, Tensor]:
        if timestep is None:
            timestep = torch.randint(0, self.timesteps, size=(1,))[0]

        noise = self.noise_fn(size)
        amount = self.sqrt_one_minus_alphas_cumprod[timestep]
        amount_t = self._add_dims(amount)
        noise = noise * amount_t

        return noise, amount
    
    """
    not batch.
    """
    def add_noise(self, orig: Tensor, timestep: int = None) -> Tuple[Tensor, Tensor, Tensor]:
        if timestep is None:
            timestep = torch.randint(low=0, high=self.timesteps, size=(1,)).item()

        noise, amount = self.noise(size=orig.shape, timestep=timestep)
        sqrt_alphas_cumprod_t = self._add_dims(self.sqrt_alphas_cumprod[timestep])
        sqrt_one_minus_alphas_cumprod_t = self._add_dims(self.sqrt_one_minus_alphas_cumprod[timestep])
        noised_orig = sqrt_alphas_cumprod_t * orig + sqrt_one_minus_alphas_cumprod_t * noise

        return noised_orig, noise, amount
    
    def gen_frame(self, net: Callable[[Tensor, Tensor], Tensor], inputs: Tensor, timestep: int) -> Tensor:
        betas_t = self.betas[timestep]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[timestep]

        noise, amount = self.noise(size=inputs.shape, timestep=timestep)
        noise = noise.to(inputs.device)
        time_t = amount.unsqueeze(0).to(inputs.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        # print(f"timestep {timestep}: inputs - {betas_t=} * net() / {sqrt_one_minus_alphas_cumprod_t=}")
        model_mean = sqrt_recip_alphas_t * (
            inputs - betas_t * net(inputs, time_t) / sqrt_one_minus_alphas_cumprod_t
        )

        if timestep == 0:
            return model_mean

        # Algorithm 2 line 4:
        posterior_variance_t = self.posterior_variance[timestep]
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def gen(self, net: Callable[[Tensor, Tensor], Tensor], inputs: Tensor, steps: int, truth_is_noise: bool) -> Tensor:
        if not truth_is_noise:
            raise Exception("doesn't work")
        out = inputs
        steps = max(1, steps)
        steps = min(self.timesteps - 1, steps)

        step_list = torch.linspace(self.timesteps - 1, 0, steps)
        for step in step_list:
            step = int(step)
            out = self.gen_frame(net, inputs=out, timestep=int(step))

        # for step in reversed(range(steps)):
        #     out = self.gen_frame(net, inputs=out, timestep=int(step * steps / self.timesteps))
            
        return out

def make_noise_schedule(type: BetaSchedType, timesteps: int, noise_type: Literal['rand', 'normal']) -> NoiseSchedule:
    if noise_type == 'rand':
        noise_fn = noise_rand
    elif noise_type == 'normal':
        noise_fn = noise_normal
    else:
        raise ValueError(f"unknown {noise_type=}")

    betas = make_betas(type=type, timesteps=timesteps)
    return NoiseSchedule(betas=betas, timesteps=timesteps, noise_fn=noise_fn)

def make_betas(type: BetaSchedType, timesteps: int) -> Tensor:
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
    # return torch.normal(mean=0, std=0.5, size=size)
    return torch.randn(size=size)

