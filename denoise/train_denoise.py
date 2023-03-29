# %%
import sys
import argparse
import datetime
from typing import List, Literal, Tuple, Dict, Callable
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
import train_util
from experiment import Experiment
import noised_data
import conv_types
import dn_util
import cmdline_image
import checkpoint_util
import re

import loggers.image_progress as img_prog
import denoise_progress as dn_prog
import loggers.chain as chain_logger

import dataloader
from models import denoise, vae, unet
import noisegen

class Config(cmdline_image.ImageTrainerConfig):
    truth_is_noise: bool
    attribute_matches: List[str]
    pattern: re.Pattern
    enc_batch_size: int
    gen_steps: List[int]

    noise_fn_str: str
    noise_steps: int
    noise_beta_type: str
    noise_schedule: noisegen.NoiseSchedule = None

    checkpoints: List[Tuple[Path, Experiment]]

    def __init__(self):
        super().__init__("denoise")
        self.add_argument("--truth", choices=["noise", "src"], default="noise")
        self.add_argument("--noise_fn", dest='noise_fn_str', default='normal', choices=['rand', 'normal'])
        self.add_argument("--noise_steps", type=int, default=300)
        self.add_argument("--noise_beta_type", type=str, default='cosine')
        self.add_argument("--gen_steps", type=int, nargs='+', default=None)
        self.add_argument("-B", "--enc_batch_size", type=int, default=4)
        # self.add_argument("-p", "--pattern", type=str, default=None)
        # self.add_argument("-a", "--attribute_matchers", type=str, nargs='+', default=[])

    def parse_args(self) -> 'Config':
        super().parse_args()
        # if self.pattern:
        #     self.pattern = re.compile(self.pattern)

        # self.checkpoints = \
        #     checkpoint_util.find_checkpoints(attr_matchers=self.attribute_matchers,
        #                                      only_paths=self.pattern)
        # self.checkpoints = sorted(self.checkpoints,
        #                           key=lambda tup: tup[1].lastepoch_train_loss)
        
        self.truth_is_noise = (self.truth == "noise")

        self.noise_schedule = \
            noisegen.make_noise_schedule(type=self.noise_beta_type,
                                         timesteps=self.noise_steps,
                                         noise_type=self.noise_fn_str)
        return self
    
    def get_dataloaders(self, vae_net: vae.VarEncDec, vae_net_path: Path) -> Tuple[DataLoader, DataLoader]:
        src_train_ds, src_val_ds = super().get_datasets()

        eds_item_type: dataloader.EDSItemType = 'sample'

        dl_args = dict(vae_net=vae_net, vae_net_path=vae_net_path,
                       batch_size=self.batch_size, enc_batch_size=self.enc_batch_size,
                       noise_schedule=self.noise_schedule,
                       eds_item_type=eds_item_type, 
                       shuffle=True, device=self.device)
        train_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_train_ds, **dl_args)
        val_dl = dataloader.NoisedEncoderDataLoader(base_dataset=src_val_ds, **dl_args)
        return train_dl, val_dl
    
    def get_loggers(self, 
                    vae_net: vae.VarEncDec,
                    exps: List[Experiment]) -> chain_logger.ChainLogger:
        logger = super().get_loggers()
        # return logger
        dn_gen = dn_prog.DenoiseProgress(truth_is_noise=self.truth_is_noise,
                                         noise_schedule=self.noise_schedule,
                                         device=self.device,
                                         gen_steps=self.gen_steps,
                                         decoder_fn=vae_net.decode,
                                         latent_dim=vae_net.latent_dim)
        img_logger = \
            img_prog.ImageProgressLogger(dirname=self.log_dirname,
                                         progress_every_nepochs=self.progress_every_nepochs,
                                         generator=dn_gen,
                                         image_size=self.image_size,
                                         exps=exps)
        logger.loggers.append(img_logger)
        return logger


def parse_args() -> Config:
    cfg = Config()
    cfg.parse_args()
    return cfg

def build_experiments(cfg: Config, exps: List[Experiment],
                      train_dl: DataLoader, val_dl: DataLoader) -> List[Experiment]:
    # NOTE: need to update do local processing BEFORE calling super().build_experiments,
    # because it looks for resume candidates and uses image_dir as part of that..
    return cfg.build_experiments(exps, train_dl, val_dl, resume_ignore_fields={'net_vae_path'})

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = parse_args()

    # grab the first vae model that we can find..
    checkpoints = checkpoint_util.find_checkpoints(attr_matchers=[
        # "loss_type ~ edge",
        # "net_class = VarEncDec",
        # "net_do_residual != True",
        # f"net_image_size = {cfg.image_size}",
        "shortcode = swssvb"
    ])
    checkpoints = sorted(checkpoints, key=lambda tup: tup[1].last_train_loss)
    vae_path, vae_exp = checkpoints[0]
    # vae_path, vae_exp = cfg.checkpoints[0]
    with open(vae_path, "rb") as file:
        model_dict = torch.load(file)

    vae_net = dn_util.load_model(model_dict=model_dict).to(cfg.device)
    vae_net.requires_grad_(False)
    vae_net.eval()
    vae_exp.net = vae_net

    print(f"""{vae_path}:
  last_train_loss: {vae_exp.last_train_loss:.3f}
    last_val_loss: {vae_exp.last_val_loss:.3f}
          nepochs: {vae_exp.nepochs}
         saved_at: {vae_exp.saved_at}
         relative: {vae_exp.saved_at_relative()}
        shortcode: {vae_exp.shortcode}
          nparams: {vae_exp.nparams() / 1e6:.3f}M""")


    # set up noising dataloaders that use vae_net as the decoder.
    train_dl, val_dl = cfg.get_dataloaders(vae_net=vae_net, vae_net_path=vae_path)

    exps: List[Experiment] = list()
    # load config file
    # with open(cfg.config_file, "r") as cfile:
    #     print(f"reading {cfg.config_file}")
    #     exec(cfile.read())

    def lazy_net_vae(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(exp: Experiment) -> nn.Module:
            return vae.VAEDenoise(**kwargs)
        return fn
        
    def lazy_net_denoise(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(exp: Experiment) -> nn.Module:
            return denoise.DenoiseModel(**kwargs)
        return fn
        
    def lazy_net_unet(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
        def fn(exp: Experiment) -> nn.Module:
            return unet.Unet(**kwargs)
        return fn
        
    # TODO hacked up experiment
    # for net_type in ['denoise', 'vae']:
    # for net_type in ['denoise']:
    net_type = 'unet'
    for self_condition in [False, True]:
    # for self_condition in [True]:
        for dim_mults in [
                [1, 2, 4, 8],
                [1, 2, 4],
                [1, 2, 4, 8, 16]
            ]:
            # for resnet_block_groups in [1, 2, 4]:
            for resnet_block_groups in [1, 4]:
                exp = Experiment()
                latent_dim = vae_net.latent_dim.copy()
                lat_chan, lat_size, _ = latent_dim

                label_parts = [
                    f"type_{net_type}",
                    "img_latdim_" + "_".join(map(str, latent_dim)),
                    f"noisefn_{cfg.noise_fn_str}",
                ]

                exp.lazy_dataloaders_fn = lambda exp: train_dl, val_dl
                exp.startlr = cfg.startlr or 1e-4
                exp.endlr = cfg.endlr or 1e-5
                exp.sched_type = "nanogpt"
                exp.loss_type = "l2"
                # exp.loss_type = "l1_smooth"
                # exp.loss_type = "l1"
                exp.is_denoiser = True
                exp.noise_steps = cfg.noise_steps
                exp.noise_beta_type = cfg.noise_beta_type
                exp.truth_is_noise = cfg.truth_is_noise
                exp.net_vae_path = str(vae_path)
                label_parts.append(f"noise_{cfg.noise_beta_type}_{cfg.noise_steps}")

                backing_loss = train_util.get_loss_fn(exp.loss_type, device=cfg.device)
                if net_type in ['denoise', 'vae']:
                    label_parts.insert(1, f"denoise-{layer_str}")
                    conv_cfg = conv_types.make_config(layer_str, final_nl_type='relu')
                    dn_chan = conv_cfg.get_channels_down(lat_chan)[-1]
                    dn_size = conv_cfg.get_sizes_down_actual(lat_size)[-1]
                    dn_dim = [dn_chan, dn_size, dn_size]
                    label_parts.append("dn_latdim_" + "_".join(map(str, dn_dim)),)

                    if net_type == 'vae':
                        backing_loss = vae.get_kld_loss_fn(exp=exp, kld_weight=2e-5,
                                                        backing_loss_fn=backing_loss,
                                                        dirname=cfg.log_dirname)
                        exp.loss_type += "+kl"
                        in_chan, in_size, _ = latent_dim
                        args = dict(in_size=in_size, in_chan=in_chan, 
                                    encoder_kernel_size=3, cfg=conv_cfg)
                        exp.lazy_net_fn = lazy_net_vae(args)
                    else:
                        args = dict(in_latent_dim=latent_dim, cfg=conv_cfg)
                        exp.lazy_net_fn = lazy_net_denoise(args)
                else:
                    # unet.Unet(init_dim=dim, out_dim=, dim_mults=[1,2,4,8],self_condition=True, resnet_block_groups=4)
                    args = dict(dim=lat_size, 
                                dim_mults=dim_mults, 
                                self_condition=self_condition, 
                                resnet_block_groups=resnet_block_groups, 
                                channels=lat_chan)
                    exp.lazy_net_fn = lazy_net_unet(args)

                    label_parts.append(f"dim_mults_{'_'.join(map(str, dim_mults))}")
                    label_parts.append(f"selfcond_{self_condition}")
                    label_parts.append(f"resblk_{resnet_block_groups}")

                exp.loss_fn = \
                    noised_data.twotruth_loss_fn(backing_loss_fn=backing_loss,
                                                truth_is_noise=cfg.truth_is_noise, 
                                                device=cfg.device)

                exp.label = ",".join(label_parts)
                exps.append(exp)


    exps = build_experiments(cfg, exps, train_dl=train_dl, val_dl=val_dl)

    logger = cfg.get_loggers(vae_net, exps)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=10)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
