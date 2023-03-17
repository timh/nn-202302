# %%
import sys
from typing import Dict, List, Union, Type, Tuple, Callable
from pathlib import Path

import torch
from torch import Tensor
import torchvision
from torchvision import transforms

import model
import model_sd
import model_new
sys.path.append("..")
import conv_types
import model_util
import train_util
from experiment import Experiment

def get_model_type(model_dict: Dict[str, any]) -> \
        Union[Type[model.ConvEncDec], Type[model_sd.Model]]:
    
    net_class = model_dict['net_class']
    if net_class == 'Model' or 'num_res_blocks' in model_dict:
        return model_sd.Model

    if net_class == 'VarEncDec' or 'conv_cfg' in model_dict['net']:
        return model_new.VarEncDec
    
    if net_class == 'ConvEncDec' or 'nlinear' in model_dict['net']:
        return model.ConvEncDec
    
    model_dict_keys = "\n  ".join(sorted(list(model_dict.keys())))
    raise ValueError(f"can't figure out model type for {net_class=}:\n{model_dict_keys=}")

def load_model(model_dict: Dict[str, any]) -> \
        Union[model.ConvEncDec, model_sd.Model]:
    fix_fields = lambda sd: {k.replace("_orig_mod.", ""): sd[k] for k in sd.keys()}
    model_dict = fix_fields(model_dict)
    model_type = get_model_type(model_dict)

    if model_type == model.ConvEncDec:
        net_dict = fix_fields(model_dict['net'])
        ctor_args = {k: net_dict.get(k) for k in model.ConvEncDec._model_fields}
        net = model.ConvEncDec(**ctor_args)
        net.load_model_dict(net_dict, True)
    elif model_type == model_new.VarEncDec:
        net_dict = fix_fields(model_dict['net'])

        ctor_args = {k: net_dict.get(k) for k in model_new.VarEncDec._model_fields}

        cfg_ctor_args = {k: net_dict.pop(k) for k in conv_types.ConvConfig._metadata_fields}
        cfg_ctor_args['layers'] = conv_types.parse_layers(cfg_ctor_args.pop('layers_str'))
        ctor_args['cfg'] = conv_types.ConvConfig(**cfg_ctor_args)

        net = model_new.VarEncDec(**ctor_args)
        net.load_model_dict(net_dict, True)
    else:
        raise NotImplementedError(f"not implemented for {model_type=}")
    
    return net

if __name__ == "__main__":
    import model_util, base_model, experiment, model
    import importlib
    for m in [experiment, model_util, base_model, model]:
        importlib.reload(m)

    descstr = "k3-s1-mp2-c64,mp2-c16,mp2-c4"
    descs = model.gen_descs(descstr)
    emblen = 32
    nlinear = 0
    hidlen = 0
    size = 128

    net = model.ConvEncDec(image_size=size, emblen=emblen, nlinear=nlinear, hidlen=hidlen, descs=descs)

    print("net.metadata_dict:")
    metadata_dict = net.metadata_dict()
    model_util.print_dict(metadata_dict, 0)
    print()

    print("exp.metadata_dict:")
    exp = Experiment(label="foo", net=net)
    exp_meta = exp.metadata_dict()
    model_util.print_dict(exp_meta, 0)
    print()

    newexp = Experiment().load_model_dict(exp_meta)
    newexp_meta = newexp.metadata_dict()
    all_keys = set(list(exp_meta.keys()) + list(newexp_meta.keys()))
    for key in all_keys:
        expval = exp_meta.get(key, None)
        newexpval = newexp_meta.get(key, None)
        if expval != newexpval:
            raise Exception(f"{key=}: {expval=} {newexpval=}")

    print("exp.model_dict:")
    exp_state = exp.model_dict()
    model_util.print_dict(exp_state, 0)
    print()

    newnet = load_model(exp_state)

    if repr(net) != repr(newnet):
        raise Exception("foo")

def get_dataloaders(*, 
                    disable_noise: bool = False, 
                    use_timestep = False,
                    amount_fn: Callable[[Tuple], Tensor] = None,
                    noise_fn: Callable[[Tuple], Tensor] = None,
                    image_size: int = 128,
                    image_dir: str,
                    batch_size: int,
                    limit_dataset = None):
    import noised_data
    from torch.utils import data

    if amount_fn is None:
        amount_fn = noised_data.gen_amount_range(0.0, 1.0)
    if noise_fn is None:
        noise_fn = noised_data.gen_noise_rand

    if disable_noise:
        dataset = torchvision.datasets.ImageFolder(
            root=image_dir,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
        ]))
        dataset = noised_data.PlainDataset(dataset)
        if limit_dataset is not None:
            dataset = data.Subset(dataset, range(0, limit_dataset))

        train_split = int(len(dataset) * 0.9)
        train_data = data.Subset(dataset, range(0, train_split))
        val_data = data.Subset(dataset, range(train_split, len(dataset)))
        train_dl = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    else:
        dataset = noised_data.load_dataset(image_dirname=image_dir, image_size=image_size,
                                           use_timestep=use_timestep,
                                           noise_fn=noise_fn, amount_fn=amount_fn)

        train_dl, val_dl = noised_data.create_dataloaders(dataset, batch_size=batch_size, 
                                                          train_all_data=True, val_all_data=True)

    return train_dl, val_dl
