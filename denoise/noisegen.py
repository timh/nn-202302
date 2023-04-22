from typing import List, Tuple, Literal, Callable, Generator
import tqdm

import torch
from torch import Tensor, FloatTensor, IntTensor
from torch import nn
import torch.nn.functional as F
import einops

BetaSchedType = Literal['cosine', 'linear', 'quadratic', 'sigmoid']
# DEFAULT_TIMESTEPS = 300

# from The Annotated Diffusion Model
#  https://huggingface.co/blog/annotated-diffusion

NoiseWithAmountFn = Callable[[Tuple], Tuple[Tensor, Tensor]]

class NoiseSchedule:
    """
    NOTE if using deterministic, this object can only hold a cache for one size of
    noise.
    """
    deterministic: bool
    scale_noise: float
    timesteps: int
    _saved_noise: Tensor

    betas: Tensor                           # betas
    orig_amount: Tensor                     # sqrt_alphas_cumprod
    noise_amount: Tensor                    # sqrt_one_minus_alphas_cumprod
    sqrt_recip_alphas: Tensor               # sqrt_recip_alphas
    posterior_variance: Tensor              # posterior_variance
    noise_fn: Callable[[Tuple], Tensor]

    # betas                               - gen_frame in model_mean equation
    # sqrt_alphas_cumprod                 - add_noise: multiplier for original image
    # sqrt_one_minus_alphas_cumprod       - add_noise: multiplier for noise
    #                                       gen_frame: denominator in model_mean eq  
    # sqrt_recip_alphas                   - gen_frame: in numerator
    # posterior_variance

    def __init__(self, betas: Tensor, timesteps: int, noise_fn: Callable[[Tuple], Tensor],
                 deterministic: bool = False, scale_noise: float = 1.0):
        # TODO: add dimension argument to reinforce that this should only be used
        # for one shape of noise.

        # also from annotated-diffusion blog post
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        orig_amount = torch.sqrt(alphas_cumprod)
        noise_amount = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # TODO: some of these still need to be renamed.
        self.betas = betas
        self.timesteps = timesteps
        self.orig_amount = orig_amount
        self.noise_amount = noise_amount
        self.sqrt_recip_alphas = sqrt_recip_alphas
        self.posterior_variance = posterior_variance
        self.noise_fn = noise_fn

        self.deterministic = deterministic
        self.scale_noise = scale_noise
        self._saved_noise = None
    
    def _add_dims(self, tensor_0d: Tensor) -> Tensor:
        assert len(tensor_0d.shape) == 0
        return einops.rearrange(tensor_0d, "-> 1 1 1")
    
    def noise(self, size: Tuple, timestep: int = None) -> Tuple[FloatTensor, FloatTensor, IntTensor]:
        if timestep is None:
            timestep = torch.randint(0, self.timesteps, size=(1,))[0]
        
        if self.deterministic:
            # cache is stored by timestep, for the batchless dimension.
            nobatch_size = size[1:] if len(size) == 4 else size

            if self._saved_noise is None:
                save_size = (self.timesteps, *nobatch_size)
                self._saved_noise = self.noise_fn(save_size)
            elif self._saved_noise.shape[1:] != nobatch_size:
                raise ValueError(f"cache is {self._saved_noise.shape=}, but input size is {size=}")

            # add desired batch dimension back. return the same random numbers for all
            # batches
            noise = self._saved_noise[timestep]
            if nobatch_size != size:
                noise = torch.stack([noise] * size[0])

        else:
            noise = self.noise_fn(size)

        amount = self.noise_amount[timestep]
        amount_t = self._add_dims(amount)
        noise = noise * amount_t * self.scale_noise

        return noise, amount, timestep
    
    def add_noise(self, orig: Tensor, timestep: int = None) -> Tuple[FloatTensor, FloatTensor, FloatTensor, IntTensor]:
        """
        Add noise given an original image and timestep.
        
        Returns:
        noised_orig: original tensor with noise added
        noise: the noise that was added
        amount: the multiplier used against the noise - based on the timestep
        timestep: the timestep used
        """
        if timestep is None:
            timestep = torch.randint(low=0, high=self.timesteps, size=(1,)).item()

        noise, amount, timestep = self.noise(size=orig.shape, timestep=timestep)
        orig_amount_t = self._add_dims(self.orig_amount[timestep])
        noised_orig = orig_amount_t.to(orig.device) * orig + noise.to(orig.device)

        return noised_orig, noise, amount, timestep
    
    def remove_noise(self, orig: Tensor, noise: FloatTensor, timestep: int) -> FloatTensor:
        orig_amount_t = self._add_dims(self.orig_amount[timestep])
        denoised_orig = orig / orig_amount_t.to(orig.device) - noise.to(orig.device)

        return denoised_orig
    
    def steps_list(self, steps: int, max_steps: int = None) -> List[int]:
        """
        Returns the (reversed) steps list, given a desired denoise of the given steps.

        Parameters:
        steps: number of steps to denoise
        max_steps: if set, will be used instead of self.timestamps. Used to adjust the 
        (highest) timestep the denoising process starts at.
        """
        max_steps = max_steps or self.timesteps
        if steps > max_steps:
            step_list = torch.linspace(1, max_steps - 2, steps)
        else:
            step_list = range(max_steps - steps + 1, max_steps - 1)
        return list(map(int, reversed(step_list)))

    def gen_step(self, 
                 net: Callable[[Tensor, Tensor], Tensor], 
                 inputs: Tensor, timestep: int, 
                 clip_embed: Tensor = None, clip_scale: Tensor = None) -> Tensor:
        betas_t = self.betas[timestep]
        noise_amount_t = self.noise_amount[timestep]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[timestep]

        noise, amount, _timestep = self.noise(size=inputs.shape, timestep=timestep)
        noise = noise.to(inputs.device)
        time_t = amount.unsqueeze(0).to(inputs.device)

        net_args = {'inputs': inputs, 'time': time_t}
        if clip_embed is not None:
            net_args['clip_embed'] = clip_embed
            if clip_scale is not None:
                net_args['clip_scale'] = clip_scale

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            inputs - betas_t * net(**net_args) / noise_amount_t
        )

        if timestep == 0:
            return model_mean

        # Algorithm 2 line 4:
        posterior_variance_t = self.posterior_variance[timestep]
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def gen(self, net: Callable[[Tensor, Tensor], Tensor], inputs: Tensor, steps: int, 
            clip_embed: Tensor = None, clip_scale: Tensor = None, clip_guidance: Tensor = None,
            max_steps: int = None, yield_count: int = None,
            progress_bar: bool = False) -> Generator[Tensor, None, None]:
        
        max_steps = max_steps or self.timesteps
        if yield_count:
            yield_every = max_steps // yield_count
        else:
            yield_every = 0

        out = inputs
        steps_list = self.steps_list(steps)
        if progress_bar:
            steps_list = tqdm.tqdm(steps_list)

        for i, step in enumerate(steps_list):
            if clip_guidance is not None and clip_embed is not None:
                cond_out = self.gen_step(net, inputs=out, timestep=step, clip_embed=clip_embed, clip_scale=clip_scale)
                uncond_out = self.gen_step(net, inputs=out, timestep=step)
                # out = (1 + clip_guidance) * cond_out - clip_guidance * uncond_out # from paper
                out = uncond_out + clip_guidance * (cond_out - uncond_out)          # from diffusers
            else:
                out = self.gen_step(net, inputs=out, timestep=step, clip_embed=clip_embed, clip_scale=clip_scale)
            if yield_every and (i % yield_every == 0 or i == len(steps_list) - 1):
                yield out
        
        if not yield_every:
            yield out

def make_noise_schedule(type: BetaSchedType, timesteps: int, noise_type: Literal['rand', 'normal'],
                        deterministic: bool = False, scale_noise: float = 1.0) -> NoiseSchedule:
    if noise_type == 'rand':
        noise_fn = noise_rand
    elif noise_type == 'normal':
        noise_fn = noise_normal
    else:
        raise ValueError(f"unknown {noise_type=}")

    betas = make_betas(type=type, timesteps=timesteps)
    return NoiseSchedule(betas=betas, timesteps=timesteps, noise_fn=noise_fn, deterministic=deterministic, scale_noise=scale_noise)

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
