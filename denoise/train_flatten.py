import sys
from typing import List, Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.append("..")
import trainer
from experiment import Experiment
import checkpoint_util
import dn_util
import train_util
import conv_types
import dataloader
from models import conv2flat, vae

from cmdline_image import ImageTrainerConfig

class Config(ImageTrainerConfig):
    vae_shortcode: str
    vae_net: vae.VarEncDec
    vae_path: Path
    vae_exp: Experiment

    loss_type: str
    nonlinearity: str

    nlayers: int
    conv_layer_str: int

    latent_dim: List[int]

    def __init__(self):
        super().__init__("flatten")
        self.add_argument("-vsc", dest='vae_shortcode', required=True)
        self.add_argument("--loss_type", type=str, required=True)
        self.add_argument("--nonlinearity", "--nl", type=str, choices=["relu"], default="relu")
        self.add_argument("--nlayers", type=int, default=None)
        self.add_argument("--conv_layer_str", type=str, default=None)

    def parse_args(self) -> 'Config':
        res = super().parse_args()
        self._load_vae()

        self.image_size = self.vae_net.image_size

        dataset = self.get_dataset()
        enc_ds = \
            dataloader.EncoderDataset(dataset=dataset,
                                      vae_net=self.vae_net, vae_net_path=self.vae_path,
                                      image_dir=self.image_dir, clip_model_name="RN50",
                                      device=self.device, batch_size=self.batch_size)
        train_ds, val_ds = dataloader.split_dataset(enc_ds)

        self.train_dl = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=True)

        self.clip_emblen = enc_ds._clip_cache.get_clip_emblen()

        return res

    def _load_vae(self):
        vae_exp = checkpoint_util.find_experiment(self.vae_shortcode)
        if vae_exp is None or vae_exp.net_class != 'VarEncDec':
            raise Exception(f"whoops, can't find VAE with shortcode {self.vae_shortcode}")

        vae_path = vae_exp.get_run().checkpoint_path
        vae_net = dn_util.load_model(vae_path).to(cfg.device)
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
    
        self.vae_net = vae_net
        self.vae_path = vae_path
        self.vae_exp = vae_exp
        self.latent_dim = vae_net.latent_dim

    def get_loss_fn(self, exp: Experiment) -> Callable[[Tensor, List[Tensor]], Tensor]:
        backing_loss_fn = train_util.get_loss_fn(exp.loss_type, device=self.device)
        def fn(out_embeds: Tensor, truth: List[Tensor]) -> Tensor:
            _truth_orig, truth_clip_embed = truth
            clip_loss = backing_loss_fn(truth_clip_embed, out_embeds)
            return clip_loss
        return fn


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = Config()
    cfg.parse_args()

    in_dim = cfg.latent_dim
    in_dim_str = "_".join(map(str, in_dim))
    out_len = cfg.clip_emblen

    # these params are often set by configs, but can be overridden here.
    exp = Experiment()
    exp.loss_type = cfg.loss_type
    exp.vae_shortcode = cfg.vae_shortcode
    exp.in_dim = cfg.latent_dim
    exp.out_len = cfg.clip_emblen
    exp.image_dir = cfg.image_dir
    exp.image_size = cfg.image_size
    exp.train_dataloader = cfg.train_dl
    exp.val_dataloader = cfg.val_dl
    exp.loss_fn = cfg.get_loss_fn(exp)

    label_parts = [
        f"vae_{cfg.vae_shortcode}",
        f"in_dim_{in_dim_str}",
        f"out_len_{out_len}",
        f"nl_{cfg.nonlinearity}",
        f"loss_{cfg.loss_type}"
    ]
    if cfg.conv_layer_str:
        conv_cfg = conv_types.make_config(in_chan=in_dim[0], in_size=in_dim[1], layers_str=cfg.conv_layer_str)
        exp.lazy_net_fn = lambda _exp: conv2flat.FlattenConv(in_dim=in_dim, out_len=out_len, cfg=conv_cfg)
        label_parts.append(cfg.conv_layer_str)
    elif cfg.nlayers:
        exp.lazy_net_fn = lambda _exp: conv2flat.FlattenLinear(in_dim=in_dim, out_len=out_len, nlayers=cfg.nlayers, nonlinearity=cfg.nonlinearity)
        label_parts.append(f"nlayers_{cfg.nlayers}")
    else:
        print(f" in_dim: {in_dim_str}")
        print(f"out_len: {out_len}")
        cfg.error("need --conv_layer_str or --nlayers")

    exp.label = ",".join(label_parts)

    exps = cfg.build_experiments([exp], train_dl=cfg.train_dl, val_dl=cfg.val_dl)

    # build loggers
    logger = cfg.get_loggers()

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), 
                        logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
