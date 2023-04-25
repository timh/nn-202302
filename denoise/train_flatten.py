import sys
from typing import List, Dict, Tuple, Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
import trainer
from experiment import Experiment
import checkpoint_util
import dn_util
import train_util
import dataloader
from models import vae, flat2conv

from cmdline_image import ImageTrainerConfig
from loggers.image_progress import ImageProgressLogger
import flat2conv_progress

class EmbedLatentDataset(dataloader.DSBase):
    def __init__(self, base_dataset: Dataset):
        super().__init__(base_dataset)
    
    def _ds_getitem(self, idx: int) -> Tuple[Tensor, Tensor]:
        inputs, _truth = self.base_dataset[idx]
        sample, clip_embed = inputs
        return (clip_embed, sample)

class Config(ImageTrainerConfig):
    vae_shortcode: str
    vae_net: vae.VarEncDec
    vae_path: Path
    vae_exp: Experiment

    loss_type: str
    nonlinearity: str

    clip_model_name: str
    latent_dim: List[int]

    def __init__(self):
        super().__init__("flatten")
        self.add_argument("-vae", dest='vae_shortcode', required=True)
        self.add_argument("--loss_type", type=str, required=True)
        self.add_argument("--nonlinearity", "--nl", type=str, choices=["relu", "silu"], default="relu")
        self.add_argument("--clip_model_name", default="RN50")

    def parse_args(self) -> 'Config':
        res = super().parse_args()
        self._load_vae()

        self.image_size = self.vae_net.image_size

        dataset = self.get_dataset()
        enc_ds = \
            dataloader.EncoderDataset(dataset=dataset,
                                      vae_net=self.vae_net, vae_net_path=self.vae_path,
                                      image_dir=self.image_dir, clip_model_name=self.clip_model_name,
                                      device=self.device, batch_size=self.batch_size)
        embed_ds = EmbedLatentDataset(enc_ds)
        train_ds, val_ds = dataloader.split_dataset(embed_ds)

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

        print(f"using VAE {self.vae_shortcode}")
    
        self.vae_net = vae_net
        self.vae_path = vae_path
        self.vae_exp = vae_exp
        self.latent_dim = vae_net.latent_dim

    def get_loss_fn(self, exp: Experiment) -> Callable[[Tensor, List[Tensor]], Tensor]:
        backing_loss_fn = train_util.get_loss_fn(exp.loss_type, device=self.device)
        def fn(out_embeds: Tensor, truth: List[Tensor]) -> Tensor:
            truth_clip_embed = truth[0]
            clip_loss = backing_loss_fn(truth_clip_embed, out_embeds)
            return clip_loss
        return fn


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = Config()
    cfg.parse_args()

    in_len = cfg.clip_emblen
    out_dim = cfg.latent_dim
    out_dim_str = "_".join(map(str, out_dim))
    
    # 1024 -> [8, 64, 64]
    # 1024 -> [4, 16, 16] -> [8, 64, 64]
    configs = [
        [ [4, 16, 16],    # first_dim
          [32, 8],        # channels
          2,              # nstride1
          2,              # nlinear
          1024,           # hidlen
          4,              # sa_nheads
          'up_first',     # sa_pos
          'silu',         # nonlinearity_type
        ],
    ]

    exps_in: List[Experiment] = list()
    for first_dim, channels, nstride1, nlinear, hidlen, sa_nheads, sa_pos, nonlinearity_type in configs:
        exp = Experiment()
        exp.loss_type = cfg.loss_type
        exp.vae_shortcode = cfg.vae_shortcode
        exp.image_dir = cfg.image_dir
        exp.image_size = cfg.image_size
        exp.train_dataloader = cfg.train_dl
        exp.val_dataloader = cfg.val_dl
        exp.loss_fn = cfg.get_loss_fn(exp)

        exp.in_len = in_len
        exp.out_dim = out_dim

        channels_str = "_".join(map(str, channels))
        label_parts = [
            f"vae_{cfg.vae_shortcode}",
            f"in_len_{in_len}",
            f"out_dim_{out_dim_str}",
            f"nl_{cfg.nonlinearity}",
            f"loss_{cfg.loss_type}"
        ]

        def net_fn(args: Dict[str, any]):
            def fn(_exp):
                net = flat2conv.EmbedToLatent(**args)
                print(net)
                return net
            return fn

        exp.label = ",".join(label_parts)

        args = dict(in_len=in_len, first_dim=first_dim, out_dim=out_dim,
                    channels=channels, nstride1=nstride1, nlinear=nlinear, hidlen=hidlen,
                    sa_nheads=sa_nheads, sa_pos=sa_pos, nonlinearity_type=nonlinearity_type)
        exp.lazy_net_fn = net_fn(args)
        exps_in.append(exp)

    exps = cfg.build_experiments(config_exps=exps_in, train_dl=cfg.train_dl, val_dl=cfg.val_dl)

    # build loggers
    logger = cfg.get_loggers()
    flat_gen = flat2conv_progress.Flat2ConvProgress(vae_net=cfg.vae_net,
                                                     device=cfg.device)
    img_logger = \
        ImageProgressLogger(basename=cfg.basename,
                            progress_every_nepochs=cfg.progress_every_nepochs,
                            generator=flat_gen,
                            image_size=cfg.image_size,
                            exps=exps)
    logger.loggers.append(img_logger)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), 
                        logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
