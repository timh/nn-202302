from typing import Tuple, List, Union, Callable

import torch
from torch import Tensor, FloatTensor

from nnexp.experiment import Experiment
from nnexp.denoise import dn_util
from nnexp.loggers import image_progress
from . import noisegen
from .models.mtypes import VarEncoderOutput


"""
always  noised_input    (noise + src)
always  truth_src       (truth)
 maybe  input - output  (noise + src) - (predicted noise)
 maybe  truth_noise     (just noise)
always  output          (either predicted noise or denoised src)
"""
class DenoiseProgress(image_progress.ImageProgressGenerator):
    truth_is_noise: bool
    render_noise: bool
    noise_sched: noisegen.NoiseSchedule = None
    device: str
    image_size: int
    decoder_fn: Callable[[Tensor], Tensor]
    latent_dim: List[int]

    steps: List[int]

    # save inputs and noise for each row - across experiments. 
    # these have a batch dimension, are detached, and on the CPU.
    saved_inputs_for_row: List[List[Tensor]] = None
    saved_noise_for_row: List[Tensor] = None

    gen_steps: List[int] = None

    def __init__(self, *,
                 truth_is_noise: bool, 
                 render_noise: bool = False,
                 noise_schedule: noisegen.NoiseSchedule,
                 decoder_fn: Callable[[Tensor], Tensor],
                 latent_dim: List[int],
                 gen_steps: List[int] = None,
                 device: str):
        self.truth_is_noise = truth_is_noise
        self.render_noise = render_noise
        self.noise_sched = noise_schedule
        self.device = device
        self.decoder_fn = decoder_fn
        self.latent_dim = latent_dim
        self.gen_steps = gen_steps

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [dn_util.exp_descr(exp, include_loss=False) for exp in exps]
    
    def get_fixed_labels(self) -> List[str]:
        return ["original", "orig + noise", "noise"]

    def get_fixed_images(self, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        truth_noise, truth_src, noised_input, amount, clip_embed, timestep = self._get_inputs(row)

        res: List[Tuple[Tensor, str]] = list()
        res.append(self.decode(truth_src))
        res.append(self.decode(noised_input))

        # add timestep to noise annotation
        noise_t = self.decode(truth_noise)
        noise_anno = f"time {timestep}/{self.noise_sched.timesteps}: {amount[0]:.2f}"
        res.append((noise_t, noise_anno))

        return res

    def get_exp_num_cols(self) -> int:
        return len(self.get_exp_col_labels())
        
    def get_exp_col_labels(self) -> List[str]:
        if self.truth_is_noise:
            if self.render_noise:
                res = ["output (predicted noise)", "output (in-out)"]
            else:
                res = ["output (in-out)"]
        else:
            res = ["output (denoised)"]

        if self.gen_steps:
            for steps in self.gen_steps:
                res.append(f"noise @{steps}")
        
        return res
    
    def get_exp_images(self, exp: Experiment, row: int, train_loss_epoch: float) -> List[Union[Tuple[Tensor, str], Tensor]]:
        truth_noise, _truth_src, noised_input, amount, clip_embed, timestep = self._get_inputs(row)

        # predict the noise.
        if clip_embed is not None:
            clip_embed = clip_embed.to(dtype=noised_input.dtype)
        noise_pred = exp.net(noised_input, amount, clip_embed)
        noise_pred_t = self.decode(noise_pred)
        loss_str = f"loss {train_loss_epoch:.5f}\ntloss {exp.last_train_loss:.5f}"
        if self.render_noise:
            res = [(noise_pred_t, f"{loss_str}\npredicted noise")]
        else:
            res = []

        # remove the noise from the original noised input
        if self.truth_is_noise:
            denoised = self.noise_sched.remove_noise(noised_input, noise_pred, timestep)
            denoised_t = self.decode(denoised)
            res.append((denoised_t, f"{loss_str}\ndenoised"))

        # denoise random noise
        if self.gen_steps:
            for i, steps in enumerate(self.gen_steps):
                gen_out = \
                    self.noise_sched.gen(net=exp.net, 
                                         inputs=truth_noise, steps=steps)
                gen_t = self.decode(gen_out)
                res.append((gen_t, f"noise @{steps}"))
        return res

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataloader = exp.train_dataloader

        if self.saved_inputs_for_row is None:
            self.saved_inputs_for_row = [list() for _ in range(nrows)]
        
        if self.saved_noise_for_row is None and self.gen_steps:
            latent_dim = [1, *exp.in_dim]
            self.saved_noise_for_row = list()
            for _ in range(nrows):
                noise, _amount, _timestep = self.noise_sched.noise(size=latent_dim)
                self.saved_noise_for_row.append(noise)
    
        self.image_size = exp.image_size
        # self.steps = [2, 5, 10, 20, 50]

    """
    Returns (truth_noise, truth_src, noised_input, amount, clip_embed, timestep).

    Returned WITH batch dimension.
    Returned on device if device is set.
    """
    def _get_inputs(self, row: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor, int]:
        # return memoized input for consistency across experiments.
        if not len(self.saved_inputs_for_row[row]):
            # take an input, then add random noise. limit noise to 1/2 steps so the 
            # visualization is more useful.
            inputs, truth = self.dataloader[row]
            _truth_noise, truth_src, _timestep = truth[:3]
            truth_src = truth_src[0]
            if len(inputs) == 3:
                clip_embed = inputs[2][0]
            else:
                clip_embed = None

            timestep = torch.randint(low=1, high=self.noise_sched.timesteps // 2, size=(1,)).item()
            noised_orig, truth_noise, amount, _timestep = self.noise_sched.add_noise(truth_src, timestep)
            
            # memoize the inputs for this row so we can use the same ones across
            # experiments.
            memo = [t.unsqueeze(0) if t is not None else None
                    for t in [truth_noise, truth_src, noised_orig, amount, clip_embed]]
            memo.append(timestep)
            self.saved_inputs_for_row[row] = memo

        # last is an int. can't call .to(device) on it.
        saved = self.saved_inputs_for_row[row]
        res = [t.to(self.device) if t is not None else None 
               for t in saved[:-1]]
        res.append(saved[-1])
        return res

    def decode(self, input_t: Tensor) -> Tensor:
        annos: List[str] = []

        input_chan = input_t.shape[1]
        latent_chan = self.latent_dim[0]
        if input_chan == latent_chan * 2:
            mean = input_t[:, :latent_chan]
            logvar = input_t[:, latent_chan:]
            veo = VarEncoderOutput(mean=mean, logvar=logvar)
            input_t = veo.sample()

        return self.decoder_fn(input_t)[0]
    