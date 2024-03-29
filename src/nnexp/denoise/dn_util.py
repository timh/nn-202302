import sys
from typing import Dict, List, Union, Type
from pathlib import Path
import functools, operator

import torch

from nnexp.images import conv_types
from nnexp.experiment import Experiment
from .models import vae, sd, denoise, denoise_new, unet, flat2conv, upscale

ModelType = Union[vae.VarEncDec, flat2conv.EmbedToLatent, upscale.UpscaleModel,
                  denoise.DenoiseModel, denoise_new.DenoiseModelNew, unet.Unet, 
                  sd.Model]
DNModelType = Union[denoise.DenoiseModel, denoise_new.DenoiseModelNew, unet.Unet]

def get_model_type(model_dict: Dict[str, any]) -> \
        Union[Type[vae.VarEncDec], Type[sd.Model], Type[denoise.DenoiseModel], Type[unet.Unet]]:
    
    net_class = model_dict['net_class']
    if net_class == 'Model' or 'num_res_blocks' in model_dict:
        return sd.Model

    if net_class == 'VarEncDec' or 'encoder_kernel_size' in model_dict['net']:
        return vae.VarEncDec
    
    if net_class in ['DenoiseModel', 'DenoiseModel2']:
        return denoise.DenoiseModel
    
    if net_class == 'DenoiseModelNew':
        return denoise_new.DenoiseModelNew
    
    if net_class == 'EmbedToLatent':
        return flat2conv.EmbedToLatent
    
    if net_class == 'UpscaleModel':
        return upscale.UpscaleModel
    
    if net_class == 'Unet':
        return unet.Unet

    model_dict_keys = "\n  ".join(sorted(list(model_dict.keys())))
    print("  " + model_dict_keys, file=sys.stderr)
    raise ValueError(f"can't figure out model type for {net_class=}")

def load_model(model_dict: Union[Dict[str, any], Path]) -> ModelType:
    fix_fields = lambda sd: {k.replace("_orig_mod.", ""): sd[k] for k in sd.keys()}

    if isinstance(model_dict, Path) or isinstance(model_dict, str):
        model_dict = torch.load(model_dict)
    model_dict = fix_fields(model_dict)
    model_type = get_model_type(model_dict)

    net_dict = fix_fields(model_dict['net'])
    net_dict.pop('class', None)

    if model_type in [vae.VarEncDec, denoise.DenoiseModel]:
        cfg_ctor_args = {field: net_dict.pop(field) 
                         for field in conv_types.ConvConfig._metadata_fields}
        if model_type == vae.VarEncDec:
            in_chan = net_dict['nchannels']
            in_size = net_dict['image_size']
        else:
            in_chan = net_dict['in_chan']
            in_size = net_dict['in_size']
        
        cfg_ctor_args['layers'] = conv_types.parse_layers(in_chan=in_chan, in_size=in_size, layers_str=cfg_ctor_args.pop('layers_str'))
        cfg_ctor_args['in_size'] = in_size
        cfg_ctor_args['in_chan'] = in_chan
        conv_cfg = conv_types.ConvConfig(**cfg_ctor_args)

        ctor_args = {k: net_dict.get(k) 
                     for k in model_type._model_fields 
                     if k in net_dict}
        ctor_args['cfg'] = conv_cfg

        net = model_type(**ctor_args)
        net.load_model_dict(net_dict, True)

    elif model_type in [unet.Unet, denoise_new.DenoiseModelNew, flat2conv.EmbedToLatent,
                        upscale.UpscaleModel]:
        ctor_args = {k: net_dict.get(k) for k in model_type._model_fields}
        net = model_type(**ctor_args)
        net.load_model_dict(net_dict, True)

    else:
        raise NotImplementedError(f"not implemented for {model_type=}")
    
    return net

def exp_image_size(exp: Experiment):
    if getattr(exp, 'is_denoiser', None):
        return exp.image_size
    return exp.net_image_size

def exp_descr(exp: Experiment, 
              include_label = True,
              include_loss = True) -> List[Union[str, List[str]]]:
    descr: List[str] = list()
    descr.append(f"code {exp.shortcode}")
    descr.append(f"nepochs {exp.nepochs}")
    descr.append(f"rel {exp.saved_at_relative()}")
    descr.append(f"loss_type {exp.loss_type}")

    if exp.net_class == 'Unet':
        descr.append(f"dim {exp.net_dim}")
        descr.append("dim_mults " + "-".join(map(str, exp.net_dim_mults)))
        descr.append(f"rnblks {exp.net_resnet_block_groups}")
        descr.append(f"selfcond {exp.net_self_condition}")

    elif exp.net_class == 'DenoiseModel':
        descr.append("in_dim " + "-".join(map(str, exp.net_in_dim)))
        descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))
        if exp.net_do_residual:
            descr.append("residual")
        if exp.net_clip_scale_default != 1.0:
            descr.append(f"clip_scale_{exp.net_clip_scale_default:.1f}")

        layers_list = exp.net_layers_str.split("-")
        layers_list[:-1] = [s + "-" for s in layers_list[:-1]]
        descr.append(layers_list)

    elif exp.net_class == 'DenoiseModelNew':
        descr.append("in_dim " + "-".join(map(str, exp.net_in_dim)))
        descr.append("latent_dim " + "-".join(map(str, exp.net_latent_dim)))
        descr.append("channels " + "-".join(map(str, exp.net_channels)))
        descr.append(f"num/stride1 {exp.net_nstride1}")
        if hasattr(exp, 'net_sa_nheads'):
            descr.append(f"sa_nheads {exp.net_sa_nheads}")
        if getattr(exp, 'net_sa_pos'):
            descr.append(f"sa_pos " + "-".join(exp.net_sa_pos))
        if hasattr(exp, 'net_ca_nheads'):
            descr.append(f"ca_nheads {exp.net_ca_nheads}")
        if exp.net_ca_pos:
            descr.append(f"ca_pos " + "-".join(exp.net_ca_pos))
        if exp.net_ca_pos_conv:
            descr.append(f"ca_pos_conv " + "-".join(exp.net_ca_pos_conv))
        if exp.net_ca_pos_lin:
            descr.append(f"ca_pos_lin " + "-".join(exp.net_ca_pos_lin))
        if exp.net_time_pos:
            descr.append(f"time " + "-".join(exp.net_time_pos))
        if exp.net_clip_scale_default != 1.0:
            descr.append(f"clip_scale_{exp.net_clip_scale_default:.1f}")
    
    elif exp.net_class == 'EmbedToLatent':
        descr.append(f"in_len {exp.net_in_len}")
        descr.append("first_dim " + "-".join(map(str, exp.net_first_dim)))
        descr.append("chan " + "-".join(map(str, exp.net_channels)))
        descr.append(f"num/stride1 {exp.net_nstride1}")
        descr.append(f"nlinear {exp.net_nstride1}")
        descr.append(f"hidlen {exp.net_hidlen}")
        if hasattr(exp, 'net_sa_nheads'):
            descr.append(f"sa_nheads {exp.net_sa_nheads}")
        if hasattr(exp, 'net_sa_pos'):
            descr.append(f"sa_pos " + "-".join(exp.net_sa_pos))
        
    elif exp.net_class == 'UpscaleModel':
        descr.append(f"down {exp.net_down_str}")
        descr.append(f"up {exp.net_up_str}")
        if exp.net_do_residual:
            descr.append("do_residual")
        if exp.net_dropout:
            descr.append(f"dropout {exp.net_dropout:.2f}")
        
    elif exp.net_class == 'VarEncDec':
        layers_list = exp.net_layers_str.split("-")
        layers_list[:-1] = [s + "-" for s in layers_list[:-1]]
        descr.append(layers_list)
        descr.append(f"klw {exp.kld_weight:.1E}")

        lat_dim_flat = functools.reduce(operator.mul, exp.net_latent_dim, 1)
        image_dim_flat = functools.reduce(operator.mul, [3, exp.net_image_size, exp.net_image_size], 1)
        ratio = f"{lat_dim_flat/image_dim_flat:.3f}"
        descr.append(f"ratio {ratio}")

    if include_loss:
        descr.append(f"tloss {exp.last_train_loss:.3f}")
        descr.append(f"vloss {exp.last_val_loss:.3f}")

    for i in range(len(descr[:-1])):
        if isinstance(descr[i], str):
            descr[i] = descr[i] + ","

    return descr