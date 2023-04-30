from typing import List, Dict, Tuple, Iterator
from PIL import Image
from dataclasses import dataclass

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

@dataclass
class _DLIter:
    dl_iter: Iterator[DataLoader]

    def __next__(self):
        orig, _ = next(self.dl_iter)
        _batch, _chan, height, width = orig.shape
        downsized = F.interpolate(orig, size=(height // 2, width // 2))
        return downsized, orig

class DownsampleDataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
    
    def __iter__(self):
        return _DLIter(dl_iter=iter(self.dataloader))

    def __len__(self):
        return len(self.dataloader)

class Config(ImageTrainerConfig):
    loss_type: str
    progress_image_size: int

    def __init__(self):
        super().__init__("upscale")
        self.add_argument("--loss_type", type=str, required=True)
        self.add_argument("--progress_image_size", type=int, default=None)

    def parse_args(self) -> 'Config':
        res = super().parse_args()

        base_dataset = self.get_dataset()
        train_ds, val_ds = dataloader.split_dataset(base_dataset)

        train_dl = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.train_dl = DownsampleDataLoader(train_dl)
        self.val_dl = DownsampleDataLoader(val_dl)

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
        ("4-4s2", "4s2-4"),
        ("4-4-4s2", "4s2-4-4"),
        ("4-4-4-4s2", "4s2-4-4-4"),
        ("4-sa2-4-4-4s2", "4s2-4-4-sa2-4"),
        # ("8-8s2", "8s2-8-sa4"),
        # ("8-8s2", "8s2-sa4-8"),
        # ("8-8s2", "sa4-8s2-8"),
        # ("8-8s2-sa4", "8s2-8"),
        # ("8-sa4-8s2", "8s2-8"),
        # ("8-sa4-8s2", "8s2-sa4-8"),
        # ("16-sa4-16s2", "16s2-sa4-16"),
        # ("32-sa4-32s2", "32s2-sa4-32"),
        # ("8-8s2-8s2", "8s2-8s2-8", 0),
        # ("8-16s2-32s2", "32s2-16s2-8", 0),
        # ([ 4,  8],       [3, 3], 0),
        # ([ 8,  8],       [3, 3], 0),
        # ([ 4,  8],       [3, 5], 0),
        # ([ 8,  8],       [3, 5], 0),
        # ([ 4,  8],       [5, 5], 0),
        # ([ 8,  8],       [5, 5], 0),
        # ([16, 16],       [3, 3], 0),
        # ([16, 16],       [3, 5], 0),
        # ([16] * 4,       [3, 3], 0),
        # ([16] * 8,       [3, 3], 0),
        # ([16] * 4,       [3, 5], 0),
        # ([16] * 8, [3, 3, 5, 5], 0),
    ]

    exps_in: List[Experiment] = list()
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

        def net_fn(args: Dict[str, any]):
            def fn(_exp):
                net = upscale.UpscaleModel(**args)
                # print(net)
                return net
            return fn

        exp.label = ",".join(label_parts)

        args = dict(down_str=down_str, up_str=up_str)
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
