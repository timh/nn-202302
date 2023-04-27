import random
from typing import Tuple, List, Union

from torch import Tensor

from nnexp.experiment import Experiment
from nnexp.loggers import image_progress
from . import dn_util
from .models import vae, flat2conv

class Flat2ConvProgress(image_progress.ImageProgressGenerator):
    vae_net: vae.VarEncDec
    latent_dim: List[int]
    image_size: int
    device: str

    dataset_idxs: List[int] = None

    def __init__(self, *, vae_net: vae.VarEncDec, device: str):
        self.vae_net = vae_net
        self.latent_dim = vae_net.latent_dim
        self.image_size = vae_net.image_size
        self.device = device

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [dn_util.exp_descr(exp, include_loss=False) for exp in exps]
    
    def get_fixed_labels(self) -> List[str]:
        return ["original"]

    def get_fixed_images(self, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        ds_idx = self.dataset_idxs[row]
        _inputs, truth = self.dataset[ds_idx]
        return [self.decode(truth)]

    def get_exp_num_cols(self) -> int:
        return len(self.get_exp_col_labels())
        
    def get_exp_col_labels(self) -> List[str]:
        return ["output"]
    
    def get_exp_images(self, exp: Experiment, row: int, train_loss_epoch: float) -> List[Union[Tuple[Tensor, str], Tensor]]:
        ds_idx = self.dataset_idxs[row]
        inputs, truth = self.dataset[ds_idx]
        
        flat_model: flat2conv.EmbedToLatent = exp.net
        inputs = inputs.unsqueeze(0).to(self.device)
        out = flat_model.forward(inputs)[0]

        out_t = self.decode(out)
        loss_str = f"loss {train_loss_epoch:.5f}\ntloss {exp.last_train_loss:.5f}"

        return [(out_t, loss_str)]

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataset = exp.train_dataloader.dataset
    
        # pick the same sample indexes for each experiment.
        if self.dataset_idxs is None:
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            self.dataset_idxs = all_idxs[:nrows]

    def decode(self, input_t: Tensor) -> Tensor:
        input_t = input_t.unsqueeze(0).to(self.device)
        return self.vae_net.decode(input_t)[0]
    