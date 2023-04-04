import datetime
import random
import sys
import math
from typing import Tuple, List, Union, Callable

from torch import Tensor

sys.path.append("..")
import noisegen
from experiment import Experiment
import dn_util
from models.mtypes import VarEncoderOutput
from loggers import image_progress

"""
always  noised_input    (noise + src)
always  truth_src       (truth)
 maybe  input - output  (noise + src) - (predicted noise)
 maybe  truth_noise     (just noise)
always  output          (either predicted noise or denoised src)
"""
class DenoiseProgress(image_progress.ImageProgressGenerator):
    truth_is_noise: bool
    noise_sched: noisegen.NoiseSchedule = None
    device: str
    image_size: int
    decoder_fn: Callable[[Tensor], Tensor]
    latent_dim: List[int]

    steps: List[int]

    dataset_idxs: List[int] = None

    # save inputs and noise for each row - across experiments. 
    # these have a batch dimension, are detached, and on the CPU.
    saved_inputs_for_row: List[List[Tensor]] = None
    saved_noise_for_row: List[Tensor] = None

    gen_steps: List[int] = None

    def __init__(self, *,
                 truth_is_noise: bool, 
                 noise_schedule: noisegen.NoiseSchedule,
                 decoder_fn: Callable[[Tensor], Tensor],
                 latent_dim: List[int],
                 gen_steps: List[int] = None,
                 device: str):
        self.truth_is_noise = truth_is_noise
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
        truth_noise, truth_src, noised_input, timestep = self._get_inputs(row)

        res: List[Tuple[Tensor, str]] = list()
        res.append(self.decode(truth_src))
        res.append(self.decode(noised_input))

        # add timestep to noise annotation
        noise_t = self.decode(truth_noise)
        noise_anno = f"time {timestep[0]:.2f}"
        res.append((noise_t, noise_anno))

        return res

    def get_exp_num_cols(self) -> int:
        return len(self.get_exp_col_labels())
        
    def get_exp_col_labels(self) -> List[str]:
        if self.truth_is_noise:
            res = ["output (predicted noise)", "output (in-out)"]
        else:
            res = ["output (denoised)"]

        if self.gen_steps:
            for steps in self.gen_steps:
                res.append(f"noise @{steps}")
        
        return res
    
    def get_exp_images(self, exp: Experiment, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        truth_noise, _truth_src, noised_input, timestep = self._get_inputs(row)

        out = exp.net(noised_input, timestep)
        tloss = exp.last_train_loss
        # vloss = exp.last_val_loss

        out_t = self.decode(out)
        tloss_str = f"tloss {tloss:.5f}"
        res = [(out_t, f"{tloss_str}\npredicted noise")]

        if self.truth_is_noise:
            truth_t = self.decode(noised_input - out)
            res.append((truth_t, f"{tloss_str}\ndenoised"))

        if self.gen_steps:
            for i, steps in enumerate(self.gen_steps):
                out = self.noise_sched.gen(net=exp.net, 
                                           inputs=truth_noise, steps=steps)
                out_t = self.decode(out)
                res.append((out_t, f"noise @{steps}"))
        return res

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataset = exp.train_dataloader.dataset
        first_input = self.dataset[0][0]

        if self.saved_inputs_for_row is None:
            self.saved_inputs_for_row = [list() for _ in range(nrows)]
        
        if self.saved_noise_for_row is None and self.gen_steps:
            latent_dim = [1, *first_input.shape]
            self.saved_noise_for_row = list()
            for _ in range(nrows):
                noise, _amount = self.noise_sched.noise(size=latent_dim)
                self.saved_noise_for_row.append(noise)
    
        # pick the same sample indexes for each experiment.
        if self.dataset_idxs is None:
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            self.dataset_idxs = all_idxs[:nrows]

        self.image_size = first_input.shape[-1]
        # self.steps = [2, 5, 10, 20, 50]

    """
    Returns (truth_noise, truth_src, noised_input, timestep).

    Returned WITH batch dimension.
    Returned on device if device is set.

    Sizes:
    - (1, nchan, size, size)   input, truth_noise, truth_src
    - (1,)                     timestep
    """
    def _get_inputs(self, row: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # return memoized input for consistency across experiments.
        if not len(self.saved_inputs_for_row[row]):
            ds_idx = self.dataset_idxs[row]

            # take one from batch.
            noised_input, timestep, twotruth = self.dataset[ds_idx]
            truth_noise, truth_src = twotruth
            
            # memoize the inputs for this row so we can use the same ones across
            # experiments.
            memo = [t.unsqueeze(0)
                    for t in [truth_noise, truth_src, noised_input, timestep]]
            self.saved_inputs_for_row[row] = memo
        
        return [t.to(self.device) for t in self.saved_inputs_for_row[row]]

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
    