import random
from typing import Tuple, List, Union

import torch
from torch import Tensor, FloatTensor
import torch.nn.functional as F

from nnexp.experiment import Experiment
from nnexp.loggers import image_progress
from . import dn_util
from .models import upscale

class UpscaleProgress(image_progress.ImageProgressGenerator):
    image_size: int
    device: str

    inputs: List[Tensor] = None
    originals: List[Tensor] = None

    def __init__(self, device: str):
        self.device = device

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        return [dn_util.exp_descr(exp, include_loss=False) for exp in exps]
    
    def get_fixed_labels(self) -> List[str]:
        return ["original", "input"]

    def get_fixed_images(self, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        input = self.inputs[row]
        orig = self.originals[row]
        return [orig, input]

    def get_exp_num_cols(self) -> int:
        return len(self.get_exp_col_labels())
        
    def get_exp_col_labels(self) -> List[str]:
        # return ["output", "orig - output", "input - output"]
        return ["output"]
    
    def get_exp_images(self, exp: Experiment, row: int, train_loss_epoch: float) -> List[Union[Tuple[Tensor, str], Tensor]]:
        input = self.inputs[row]
        orig = self.originals[row]
        
        upscale_model: upscale.UpscaleModel = exp.net
        out_t = upscale_model.forward(input.unsqueeze(0).to(self.device))[0]
        out_t = out_t.cpu()

        input = F.interpolate(input.unsqueeze(0), size=orig.shape[1:])[0]

        # def sub_norm(t0: FloatTensor, t1: FloatTensor) -> Tuple[Tensor, str]:
        #     res = t0 - t1
        #     sub_str = f"mean {res.mean():.3f}, std {res.std():.3f}"

        #     min_, max_ = torch.min(res), torch.max(res)
        #     diff = max_ - min_
        #     return res / diff - min_, sub_str

        # orig_minus_out, orig_str = sub_norm(orig, out_t)
        # input_minus_out, input_str = sub_norm(input, out_t)

        loss_str = f"loss {train_loss_epoch:.5f}\ntloss {exp.last_train_loss:.5f}, vloss {exp.last_val_loss:.5f}"
        return [
            (out_t, loss_str),
            # (orig_minus_out, orig_str),
            # (input_minus_out, input_str)
        ]

    def on_exp_start(self, exp: Experiment, nrows: int):
        self.dataloader = exp.train_dataloader
    
        # pick the same sample indexes for each experiment.
        if self.inputs is None:
            self.inputs = list()
            self.originals = list()

            dl_it = iter(self.dataloader)

            while len(self.inputs) < nrows:
                inputs, original = next(dl_it)
                self.inputs.extend(inputs)
                self.originals.extend(original)
            
            self.inputs = self.inputs[:nrows]
            self.originals = self.originals[:nrows]
            

    