# %%
from dataclasses import dataclass
from typing import Dict
import sys
import importlib
import datetime
import torch

sys.path.append("..")
import experiment
from experiment import Experiment
from model import ConvEncDec
import denoise_logger

importlib.reload(experiment)
importlib.reload(denoise_logger)

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
    
    def load_state_dict(self, state_dict: Dict[str, any]) -> 'DNExperiment':
        # load base experiment fields, and populate this object with 
        # the values in net_args.
        super().load_state_dict(state_dict, ['net'])
        if 'net' in state_dict:
            pass
        if 'sched' in state_dict:
            pass
        if 'optim' in state_dict:
            pass
        return self

if __name__ == "__main__":
    all_checkpoints = denoise_logger.find_all_checkpoints()
    all_exps = [cp[1] for cp in all_checkpoints]
    exp = all_exps[0]

    print(exp)
    state_dict = exp.metadata_dict()

    load_exp = experiment.load_from_dict(state_dict)
    print(load_exp)
    
    noise_exp = DNExperiment()
    noise_exp.load_state_dict(state_dict)
    print(noise_exp)

    noise_exp_full = DNExperiment()
    with open(all_checkpoints[0][0], "rb") as file:
        torch_state_dict = torch.load(file)
        print(torch_state_dict.get('train_loss_hist', None))
        # noise_exp_full.load_state_dict(torch_state_dict)
        # print(noise_exp_full)

