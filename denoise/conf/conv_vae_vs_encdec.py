import sys
import argparse
from typing import List, Dict, Callable
from torch import Tensor

sys.path.append("..")
sys.path.append("../..")
import model
import model_vanvae
import train_util
from experiment import Experiment

# these are assumed to be defined when this config is eval'ed.
cfg: argparse.Namespace
device: str
exps: List[Experiment]

loss_type = "l1"
loss_type_kld = f"{loss_type}+kld"

startlr = 1e-3
endlr = 1e-4
sched_type = "nanogpt"
optim_type = "adamw"
image_size = cfg.image_size

kld_weight    = 2.5e-4
vanilla_hdims = [32, 64, 128, 256, 512]
vanilla_ldim  = 128
# maxpool_descr = "k3-s1-mp2-c32-c64-c128-c256-c512"
conv_descr = "k3-s2-c32,c64,c128,c256,c512"

def encdec_net(kwargs: Dict[str, any]):
    def fn(exp: Experiment):
        return model.ConvEncDec(**kwargs)
    return fn

def vanilla_net(kwargs: Dict[str, any]):
    def fn(exp: Experiment):
        return model_vanvae.VanillaVAE(**kwargs)
    return fn

def make_exp(kwargs: Dict[str, any], base_label: str, lazy_net_fn) -> Experiment:
    label_parts: List[str] = list()

    for field, val in kwargs.items():
        if isinstance(val, list):
            val = "_".join(map(str, val))
        elif field == 'descs':
            continue
        label_parts.append(f"{field}_{val}")


    label = ",".join(label_parts)
    if 'descs' in kwargs:
        descs = kwargs.pop('descs')
        label = f"{descs},{label}"
        kwargs['descs'] = model.gen_descs(descs)
    label = f"{base_label}_{label}"

    kwargs.pop('loss_type')

    exp = Experiment(label=label, 
                     loss_type=loss_type, startlr=startlr, endlr=endlr,
                     sched_type=sched_type, optim_type=optim_type,
                     lazy_net_fn=lazy_net_fn(kwargs))
    for field, val in kwargs.items():
        setattr(exp, field, val)
    return exp

# ConvEncDec, stride=2, no linear.
base_encdec = dict(image_size=image_size, nchannels=3,
                   emblen=0, nlinear=0, hidlen=0, 
                   loss_type=loss_type_kld)
base_loss_fn = train_util.get_loss_fn(loss_type)

# encdec_conv_args = base_encdec.copy()
# encdec_conv_args['descs'] = conv_descr
# encdec_conv_exp = make_exp(encdec_conv_args, "encdec-conv", encdec_net)
# encdec_conv_exp.loss_fn = train_util.get_loss_fn(loss_type)

# ConvEncDec, maxpool intead of stride=2, no linear.
# encdec_maxpool_args = base_encdec.copy()
# encdec_maxpool_args['descs'] = maxpool_descr
# encdec_maxpool_exp = make_exp(encdec_maxpool_args, "encdec-maxpool", encdec_net)
# encdec_maxpool_exp.loss_fn = train_util.get_loss_fn(loss_type)

# ConvEncDec, maxpool intead of stride=2, no linear.
# encdec_maxpool_lin_args = base_encdec.copy()
# encdec_maxpool_lin_args['descs'] = maxpool_descr
# encdec_maxpool_lin_args.extend(dict(emblen=128))
# encdec_maxpool_lin_args['descs'] = model.gen_descs(maxpool_descr)
# encdec_maxpool_lin_exp = make_exp(encdec_maxpool_lin_args, "enddec-maxpool-lin", encdec_net)
# encdec_maxpool_lin_exp.loss_fn = train_util.get_loss_fn(loss_type)

encdec_vae_args = base_encdec.copy()
encdec_vae_args['descs'] = conv_descr
encdec_vae_args['do_variational'] = True
encdec_vae_exp = make_exp(encdec_vae_args, "encdec-conv", encdec_net)
# encdec_vae_exp.loss_fn = train_util.get_loss_fn(loss_type)
encdec_vae_exp.loss_fn = model.get_kl_loss_fn(encdec_vae_exp, kld_weight, base_loss_fn)

vanilla_args = dict(image_size=cfg.image_size, in_channels=3, 
                    latent_dim=vanilla_ldim, hidden_dims=vanilla_hdims,
                    loss_type=loss_type_kld)
vanilla_exp = make_exp(vanilla_args, "vanilla", vanilla_net)
vanilla_exp.loss_fn = model_vanvae.get_loss_fn(vanilla_exp, kld_weight, loss_type)

exps = [
    vanilla_exp,
    encdec_vae_exp
]

print(f"{len(exps)=}")
# import random
# random.shuffle(exps)
