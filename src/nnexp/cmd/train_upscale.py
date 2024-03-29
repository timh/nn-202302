from typing import List, Dict, Tuple, Iterator
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
import tqdm

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nnexp.images import image_util
from nnexp.utils.cmdline_image import ImageTrainerConfig
from nnexp.training import trainer, train_util
from nnexp.experiment import Experiment
from nnexp.denoise import dn_util, dataloader
from nnexp.denoise.models import upscale
from nnexp.denoise import upscale_progress
from nnexp.loggers.image_progress import ImageProgressLogger

class CacheDataset:
    orig: List[Tensor]

    def __init__(self, image_dir: str, image_size: int):
        path = Path(image_dir, f"cache-{image_size}x.pt")
        if path.exists():
            self.orig = torch.load(path)
            return

        print(f"generating {path} for size {image_size}")
        self.orig = list()
        dataset = image_util.get_dataset(image_size=image_size, image_dir=image_dir)
        for image, _truth in tqdm.tqdm(dataset, total=len(dataset)):
            self.orig.append(image)

        print(f".. wrote {path}")
        torch.save(self.orig, path)
    
    def __len__(self) -> int:
        return len(self.orig)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.orig[idx], self.orig[idx]

class CombinedDataset:
    ds1: Dataset
    ds2: Dataset

    def __init__(self, ds1: Dataset, ds2: Dataset):
        self.ds1 = ds1
        self.ds2 = ds2
    
    def __len__(self) -> int:
        return len(self.ds1)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.ds1[idx][0], self.ds2[idx][0]

class Config(ImageTrainerConfig):
    loss_type: str
    progress_image_size: int

    def __init__(self):
        super().__init__("upscale")
        self.add_argument("--loss_type", type=str, required=True)
        self.add_argument("--progress_image_size", type=int, default=None)

    def parse_args(self) -> 'Config':
        res = super().parse_args()

        orig_ds = CacheDataset(self.image_dir, self.image_size)
        downsize_ds = CacheDataset(self.image_dir, self.image_size // 2)
        combined_ds = CombinedDataset(downsize_ds, orig_ds)

        train_ds, val_ds = dataloader.split_dataset(combined_ds)

        self.train_dl = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_dl = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

        return res

    def get_loss_fn(self, exp: Experiment):
        return train_util.get_loss_fn(self.loss_type, device=self.device)

    def resume_net_fn(self):
        def fn(exp: Experiment) -> nn.Module:
            run = exp.get_run(loss_type=self.use_best)
            print(f"run {run}")
            return dn_util.load_model(run.checkpoint_path)
        return fn

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = Config()
    cfg.parse_args()

    configs = [
        ("16-32s2-32", "32-32s2-16"),
        ("16-sa4-32s2-32", "32-32s2-sa4-16"),
        ("32-32s2-32", "32-32s2-32"),

        ("32-sa2-64s2-32", "32-64s2-32"),
        ("32-64s2-sa2-32", "32-64s2-32"),
        ("32-64s2-32-sa2", "32-64s2-32"),
        ("32-64s2-32", "sa2-32-64s2-32"),
        ("32-64s2-32", "32-sa2-64s2-32"),
        ("32-64s2-32", "32-64s2-sa2-32"),
        ("32-64s2-32", "32-64s2-32-sa2"),

        ("32-sa4-64s2-32", "32-64s2-32"),
        ("32-64s2-sa4-32", "32-64s2-32"),
        ("32-64s2-32-sa4", "32-64s2-32"),
        ("32-64s2-32", "sa4-32-64s2-32"),
        ("32-64s2-32", "32-sa4-64s2-32"),
        ("32-64s2-32", "32-64s2-sa4-32"),
        ("32-64s2-32", "32-64s2-32-sa4"),

        ("64-sa4-128s2-64", "64-128s2-64"),
        ("64-128s2-sa4-64", "64-128s2-64"),
        ("64-128s2-64-sa4", "64-128s2-64"),
        ("64-128s2-64", "sa4-64-128s2-64"),
        ("64-128s2-64", "64-sa4-128s2-64"),
        ("64-128s2-64", "64-128s2-sa4-64"),
        ("64-128s2-64", "64-128s2-64-sa4"),
    ]

    exps_in: List[Experiment] = list()
    for do_residual, dropout in [(True, 0.1), (True, 0.0)]:
        for down_str, up_str in configs:
            exp = Experiment()
            exp.loss_type = cfg.loss_type
            exp.image_dir = cfg.image_dir
            exp.image_size_in = cfg.image_size // 2
            exp.image_size_out = cfg.image_size
            exp.train_dataloader = cfg.train_dl
            exp.val_dataloader = cfg.val_dl
            exp.loss_fn = cfg.get_loss_fn(exp)

            label_parts = [
                f"loss-{cfg.loss_type}",
                f"down-{down_str}",
                f"up-{up_str}",
            ]
            if do_residual:
                label_parts.append("residual")
            if dropout:
                label_parts.append(f"dropout-{dropout:.2f}")

            def net_fn(args: Dict[str, any]):
                def fn(_exp):
                    net = upscale.UpscaleModel(**args)
                    # print(net)
                    return net
                return fn

            exp.label = ",".join(label_parts)

            args = dict(down_str=down_str, up_str=up_str, do_residual=do_residual, dropout=dropout)
            exp.lazy_net_fn = net_fn(args)
            exps_in.append(exp)
    
    if cfg.resume_shortcodes:
        exps_in = None
    exps = cfg.build_experiments(config_exps=exps_in, train_dl=cfg.train_dl, val_dl=cfg.val_dl, 
                                 loss_fn=cfg.get_loss_fn, resume_net_fn=cfg.resume_net_fn())
    exps = exps[:10]

    # build loggers
    logger = cfg.get_loggers()
    flat_gen = upscale_progress.UpscaleProgress(device=cfg.device)
    img_logger = \
        ImageProgressLogger(basename=cfg.basename,
                            progress_every_nepochs=cfg.progress_every_nepochs,
                            generator=flat_gen,
                            image_size=cfg.progress_image_size or cfg.image_size,
                            exps=exps)
    logger.loggers.append(img_logger)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), 
                        logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
