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
    do_variational: bool = False
    do_layernorm: bool = False
    do_batchnorm: bool = False
    use_bias: bool = False
    conv_descs: str = ""
    truth_is_noise: bool = True
