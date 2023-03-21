# %%
import sys
from typing import Dict, List, Union, Type, Tuple, Callable
from pathlib import Path

import torch
from torch import Tensor
import torchvision
from torchvision import transforms

import model_sd
import model_new
import model_denoise
sys.path.append("..")
import conv_types
import model_util
import train_util
from experiment import Experiment

def get_model_type(model_dict: Dict[str, any]) -> \
        Union[Type[model_new.VarEncDec], Type[model_sd.Model], Type[model_denoise.DenoiseModel]]:
    
    net_class = model_dict['net_class']
    if net_class == 'Model' or 'num_res_blocks' in model_dict:
        return model_sd.Model

    if net_class == 'VarEncDec' or 'cfg' in model_dict['net']:
        return model_new.VarEncDec
    
    if net_class == 'DenoiseModel':
        return model_denoise.DenoiseModel
    
    model_dict_keys = "\n  ".join(sorted(list(model_dict.keys())))
    raise ValueError(f"can't figure out model type for {net_class=}:\n{model_dict_keys=}")

def load_model(model_dict: Dict[str, any]) -> \
        Union[model_new.VarEncDec, model_sd.Model, model_denoise.DenoiseModel]:
    fix_fields = lambda sd: {k.replace("_orig_mod.", ""): sd[k] for k in sd.keys()}
    model_dict = fix_fields(model_dict)
    model_type = get_model_type(model_dict)

    net_dict = fix_fields(model_dict['net'])
    if model_type in [model_new.VarEncDec, model_denoise.DenoiseModel]:
        cfg_ctor_args = {field: net_dict.pop(field) 
                         for field in conv_types.ConvConfig._metadata_fields}
        cfg_ctor_args['layers'] = conv_types.parse_layers(cfg_ctor_args.pop('layers_str'))
        conv_cfg = conv_types.ConvConfig(**cfg_ctor_args)

    if model_type == model_new.VarEncDec:
        ctor_args = {k: net_dict.get(k) for k in model_new.VarEncDec._model_fields}
        ctor_args['cfg'] = conv_cfg

        net = model_new.VarEncDec(**ctor_args)
        net.load_model_dict(net_dict, True)

    elif model_type == model_denoise.DenoiseModel:
        ctor_args = {k: net_dict.get(k) for k in model_denoise.DenoiseModel._model_fields}
        ctor_args['cfg'] = conv_cfg

        net = model_denoise.DenoiseModel(**ctor_args)
        net.load_model_dict(net_dict, True)
    else:
        raise NotImplementedError(f"not implemented for {model_type=}")
    
    return net

# def get_image_dataloaders(*, 
#                         use_timestep = False,
#                         amount_fn: Callable[[Tuple], Tensor] = None,
#                         noise_fn: Callable[[Tuple], Tensor] = None,
#                         image_size: int = 128,
#                         image_dir: str,
#                         batch_size: int,
#                         limit_dataset = None,
#                         train_split = 0.9,
#                         shuffle = True):
#     import noised_data
#     from torch.utils import data

#     if amount_fn is None:
#         amount_fn = noised_data.gen_amount_range(0.0, 1.0)
#     if noise_fn is None:
#         noise_fn = noised_data.gen_noise_rand

#     dataset = noised_data.load_dataset(image_dirname=image_dir, image_size=image_size,
#                                        use_timestep=use_timestep,
#                                        noise_fn=noise_fn, amount_fn=amount_fn)

#     train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, 
#                                                         train_all_data=True, val_all_data=True)

#     return train_dl, val_dl

