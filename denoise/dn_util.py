# %%
import sys
from typing import Dict, List, Union, Type
from pathlib import Path

import model
import model_sd
sys.path.append("..")
import model_util
import train_util
from experiment import Experiment

def get_model_type(state_dict: Dict[str, any]) -> \
        Union[Type[model.ConvEncDec], Type[model_sd.Model]]:
    
    net_class = state_dict['net_class']
    if net_class == 'Model' or 'num_res_blocks' in state_dict:
        return model_sd.Model
    
    if net_class == 'ConvEncDec' or 'nlinear' in state_dict:
        nl = state_dict.get('nlinear', None)
        return model.ConvEncDec
    
    state_dict_keys = "\n  ".join(sorted(list(state_dict.keys())))
    raise ValueError(f"can't figure out model type for {net_class=}:\n{state_dict_keys=}")

def load_model(state_dict: Dict[str, any]) -> \
        Union[model.ConvEncDec, model_sd.Model]:
    fix_fields = lambda sd: {k.replace("_orig_mod.", ""): sd[k] for k in sd.keys()}
    state_dict = fix_fields(state_dict)
    model_type = get_model_type(state_dict)

    if model_type == model.ConvEncDec:
        #ctor_args = {k: state_dict.pop(k) for k in model.ENCDEC_FIELDS if k in state_dict}
        net_dict = fix_fields(state_dict['net'])
        ctor_args = {k: net_dict.get(k) for k in model.ConvEncDec._statedict_fields}
        net = model.ConvEncDec(**ctor_args)
        net.load_state_dict(net_dict, True)
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

    newexp = Experiment().load_state_dict(exp_meta)
    newexp_meta = newexp.metadata_dict()
    all_keys = set(list(exp_meta.keys()) + list(newexp_meta.keys()))
    for key in all_keys:
        expval = exp_meta.get(key, None)
        newexpval = newexp_meta.get(key, None)
        if expval != newexpval:
            raise Exception(f"{key=}: {expval=} {newexpval=}")

    print("exp.state_dict:")
    exp_state = exp.state_dict()
    model_util.print_dict(exp_state, 0)
    print()

    newnet = load_model(exp_state)

    if repr(net) != repr(newnet):
        raise Exception("foo")



