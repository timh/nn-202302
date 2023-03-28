from typing import List, Union

import torch
from torch import Tensor

# TODO: make this output work through trainer
class VarEncoderOutput:
    mean: Tensor     # result from mean(out)
    logvar: Tensor   # result from logvar(out)

    def __init__(self, mean: Tensor, logvar: Tensor = None, std: Tensor = None):
        if logvar is None and std is None:
            raise ValueError("either logvar or std must be set")
        
        if logvar is None:
            logvar = torch.log(std) * 2.0
        
        self.mean = mean
        self.logvar = logvar
    
    def copy(self, mean: Tensor = None, logvar: Tensor = None, std: Tensor = None):
        if mean is None:
            mean = self.mean

        if std is not None:
            logvar = torch.log(std) * 2.0
        elif logvar is None:
            logvar = self.logvar
        
        return VarEncoderOutput(mean=mean, logvar=logvar)

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)

    @property    
    def kl_loss(self) -> Tensor:
        return torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mean ** 2 - self.logvar.exp(), dim=1))
    
    def sample(self, std: Tensor = None, mean: Tensor = None, epsilon: Tensor = None,
               device: str = None) -> Tensor:
        mean = mean or self.mean
        std = std or self.std
        epsilon = epsilon or torch.randn_like(self.std)
        res = mean + epsilon * std
        if device is not None:
            res = res.to(device)
        return res
    
    def detach(self) -> 'VarEncoderOutput':
        self.mean = self.mean.detach()
        self.logvar.detach_()
        return self

    def cpu(self) -> 'VarEncoderOutput':
        self.mean = self.mean.cpu()
        self.logvar = self.logvar.cpu()
        return self

    def len(self) -> int:
        return len(self.mean)
    
    """
    turn this (batched) VarEncoderOutput into one that is separated by its contained
    images.
    """
    def to_list(self) -> List['VarEncoderOutput']:
        if len(self.mean.shape) != 4:
            raise ValueError(f"can only call on (batch, chan, height, width): {self.mean.shape=}")

        res: List[VarEncoderOutput] = list()
        for mean, logvar in zip(self.mean, self.logvar):
            res.append(VarEncoderOutput(mean=mean, logvar=logvar))
        return res
    
    def cat_mean_logvar(self) -> Tensor:
        if len(self.mean.shape) == 3:
            dim = 0
        elif len(self.mean.shape) == 4:
            dim = 1
        return torch.cat([self.mean, self.logvar], dim=dim)
    
    @staticmethod
    def from_cat(mean_logvar: Tensor) -> 'VarEncoderOutput':
        if len(mean_logvar.shape) == 4:
            mean_len = mean_logvar.shape[1] // 2
            mean = mean_logvar[:, :mean_len]
            logvar = mean_logvar[:, mean_len:]
        elif len(mean_logvar.shape) == 3:
            mean_len = mean_logvar.shape[0] // 2
            mean = mean_logvar[:mean_len]
            logvar = mean_logvar[mean_len:]
        return VarEncoderOutput(mean=mean, logvar=logvar)
    
    def __mul__(self, other: Union[Tensor, 'VarEncoderOutput']) -> 'VarEncoderOutput':
        if isinstance(other, VarEncoderOutput):
            other_mean = other.mean
            other_logvar = other.logvar
        else:
            other_mean = other
            other_logvar = other
        
        return VarEncoderOutput(mean=self.mean * other_mean, logvar=self.logvar * other_logvar)

    def __add__(self, other: Tensor) -> 'VarEncoderOutput':
        if isinstance(other, VarEncoderOutput):
            other_mean = other.mean
            other_logvar = other.logvar
        else:
            other_mean = other
            other_logvar = other
        
        return VarEncoderOutput(mean=self.mean + other_mean, logvar=self.logvar + other_logvar)


