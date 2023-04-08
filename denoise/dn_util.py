import sys
from typing import Dict, List, Union, Type
from pathlib import Path
import functools, operator

import torch

sys.path.append("..")
import conv_types
from experiment import Experiment
from models import vae, sd, denoise, unet, ae_simple, linear
import image_util

def get_model_type(model_dict: Dict[str, any]) -> \
        Union[Type[vae.VarEncDec], Type[sd.Model], Type[denoise.DenoiseModel], Type[unet.Unet]]:
    
    net_class = model_dict['net_class']
    if net_class == 'Model' or 'num_res_blocks' in model_dict:
        return sd.Model

    if net_class == 'VarEncDec' or 'encoder_kernel_size' in model_dict['net']:
        return vae.VarEncDec
    
    if net_class == 'DenoiseModel':
        return denoise.DenoiseModel
    
    if net_class == 'Unet':
        return unet.Unet

    if net_class == 'Autoencoder':
        return ae_simple.Autoencoder
    if net_class == 'AEDenoise':
        return ae_simple.AEDenoise
    
    if net_class == 'DenoiseLinear':
        return linear.DenoiseLinear
    
    model_dict_keys = "\n  ".join(sorted(list(model_dict.keys())))
    print("  " + model_dict_keys, file=sys.stderr)
    raise ValueError(f"can't figure out model type for {net_class=}")

def load_model(model_dict: Union[Dict[str, any], Path]) -> \
        Union[vae.VarEncDec, sd.Model, denoise.DenoiseModel]:
    fix_fields = lambda sd: {k.replace("_orig_mod.", ""): sd[k] for k in sd.keys()}

    if isinstance(model_dict, Path) or isinstance(model_dict, str):
        model_dict = torch.load(model_dict)
    model_dict = fix_fields(model_dict)
    model_type = get_model_type(model_dict)

    net_dict = fix_fields(model_dict['net'])
    net_dict.pop('class', None)

    if model_type in [vae.VarEncDec, denoise.DenoiseModel, ae_simple.Autoencoder, ae_simple.AEDenoise]:
        cfg_ctor_args = {field: net_dict.pop(field) 
                         for field in conv_types.ConvConfig._metadata_fields}
        if model_type == vae.VarEncDec:
            in_chan = net_dict['nchannels']
            in_size = net_dict['image_size']
        else:
            in_chan = net_dict['in_chan']
            in_size = net_dict['in_size']
        
        print(f"model_type {model_type}: {in_chan=} {in_size=}")

        cfg_ctor_args['layers'] = conv_types.parse_layers(in_chan=in_chan, in_size=in_size, layers_str=cfg_ctor_args.pop('layers_str'))
        cfg_ctor_args['in_size'] = in_size
        cfg_ctor_args['in_chan'] = in_chan
        conv_cfg = conv_types.ConvConfig(**cfg_ctor_args)

        ctor_args = {k: net_dict.get(k) for k in model_type._model_fields}
        ctor_args['cfg'] = conv_cfg

        net = model_type(**ctor_args)
        net.load_model_dict(net_dict, True)

    elif model_type in [unet.Unet, linear.DenoiseLinear]:
        ctor_args = {k: net_dict.get(k) for k in model_type._model_fields}
        net = model_type(**ctor_args)
        net.load_model_dict(net_dict, True)

    else:
        raise NotImplementedError(f"not implemented for {model_type=}")
    
    return net

def exp_image_size(exp: Experiment):
    if exp.net_class == 'Unet':
        return exp.image_size
    return exp.net_image_size

def exp_descr(exp: Experiment, 
              include_label = True,
              include_loss = True) -> List[Union[str, List[str]]]:
    descr: List[str] = list()
    descr.append(f"code {exp.shortcode},")
    descr.append(f"nepochs {exp.nepochs},")
    descr.append(f"rel {exp.saved_at_relative()},")
    descr.append(f"loss_type {exp.loss_type},")

    if exp.net_class == 'Unet':
        descr.append(f"dim {exp.net_dim},")
        descr.append("dim_mults " + "-".join(map(str, exp.net_dim_mults)) + ",")
        descr.append(f"rnblks {exp.net_resnet_block_groups},")
        descr.append(f"selfcond {exp.net_self_condition}")
    elif exp.net_class == 'AEDenoise':
        descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))
    elif exp.net_class == 'DenoiseLinear':
        descr.append(f"nlayers {exp.net_nlayers}")
        descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))
    elif exp.net_class == 'VarEncDec':
        layers_list = exp.net_layers_str.split("-")
        layers_list[:-1] = [s + "-" for s in layers_list[:-1]]
        descr.append(layers_list)
        descr.append(f"klw {exp.kld_weight:.1E},")
        descr.append(f"bltrue {exp.last_bl_true_loss:.3f},")

        lat_dim_flat = functools.reduce(operator.mul, exp.net_latent_dim, 1)
        image_dim_flat = functools.reduce(operator.mul, [3, exp.net_image_size, exp.net_image_size], 1)
        ratio = f"{lat_dim_flat/image_dim_flat:.3f}"
        descr.append(f"ratio {ratio}")

    if include_loss:
        descr[-1] += ","
        descr.append(f"tloss {exp.last_train_loss:.3f}")
    
    return descr