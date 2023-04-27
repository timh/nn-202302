# %%
import sys
from typing import List

import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from nnexp.training import trainer
from nnexp.experiment import Experiment
from nnexp.denoise import ae_progress, dataloader
from nnexp.images import image_util

from nnexp.denoise.cmdline_image import ImageTrainerConfig
from nnexp.loggers import image_progress as im_prog

# python train_ae.py -b 8 --amp -n 500 --startlr 2e-3 --endlr 2e-4 
#   --resume -c conf/ae_vae.py -I 256 -d images.alex-1024
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cfg = ImageTrainerConfig("ae")
    cfg.parse_args()

    # eval the config file. 
    exps: List[Experiment] = list()
    with open(cfg.config_file, "r") as cfile:
        print(f"reading {cfg.config_file}")
        exec(cfile.read())
    
    # these params are often set by configs, but can be overridden here.
    dataset = image_util.get_dataset(image_size=cfg.image_size, image_dir=cfg.image_dir)
    train_ds, val_ds = dataloader.split_dataset(dataset)
    train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=cfg.batch_size)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, batch_size=cfg.batch_size)

    for exp in exps:
        exp.image_dir = cfg.image_dir

    cfg.build_experiments(config_exps=exps, train_dl=train_dl, val_dl=val_dl)

    # build loggers
    logger = cfg.get_loggers()
    ae_gen = ae_progress.AutoencoderProgress(device=cfg.device)
    img_logger = im_prog.ImageProgressLogger(basename=cfg.basename,
                                             progress_every_nepochs=cfg.progress_every_nepochs,
                                             generator=ae_gen,
                                             image_size=cfg.image_size,
                                             exps=exps)

    logger.loggers.append(img_logger)

    # train.
    t = trainer.Trainer(experiments=exps, nexperiments=len(exps), logger=logger, 
                        update_frequency=30, val_limit_frequency=0)
    t.train(device=cfg.device, use_amp=cfg.use_amp)
