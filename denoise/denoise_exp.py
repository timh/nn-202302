# %%
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import sys
import importlib
import datetime
import torch

sys.path.append("..")
import experiment
from experiment import Experiment
from model import ConvEncDec

importlib.reload(experiment)

#CONV_ENCDEC_FIELDS = "image_size emblen nlinear hidlen do_layernorm do_batchnorm use_bias descs nchannels".split(" ")

@dataclass(kw_only=True)
class DNExperiment(Experiment):
    image_size: int = 0
    nchannels: int = 0
    nlinear: int = 0
    emblen: int = 0
    hidlen: int = 0
    do_layernorm: bool = False
    do_batchnorm: bool = False
    use_bias: bool = False
    conv_descs: str = ""

    def metadata_dict(self) -> Dict[str, any]:
        res = super().metadata_dict()
        # superclass handles this completely, including the net
        return res
    
    def load_model_dict(self, model_dict: Dict[str, any]) -> 'DNExperiment':
        # load base experiment fields, and populate this object with 
        # the values in net_args.
        super().load_model_dict(model_dict, ['net'])
        if 'net' in model_dict:
            pass
        if 'sched' in model_dict:
            pass
        if 'optim' in model_dict:
            pass
        return self

def load_model_checkpoint(path: Path) -> DNExperiment:
    with open(path, "rb") as file:
        model_dict = torch.load(file)

    required_fields = 'net net_class'.split(" ")
    if not all(field in model_dict for field in required_fields):
        raise ValueError(f"missing keys in checkpoint")
    
    net_class = model_dict['net']
    #if net_class == 
