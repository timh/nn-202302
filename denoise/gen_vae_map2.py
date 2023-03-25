import sys
from typing import List, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
from pathlib import Path
import random
import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append("..")
import experiment
from experiment import Experiment
import image_latents
import image_util
import cmdline
import model_new
import dn_util
import image_latents
import diffusers.models.autoencoder_kl as aekl

class Config(cmdline.QueryConfig):
    # chart_size: int
    num_images: int
    image_dir: str
    tile_size: int

    load_sd_vae: bool

    def __init__(self):
        super().__init__()
        # self.add_argument("-c", "--chart_size", type=int, default=1024)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-n", "--num_images", type=int, default=128)
        self.add_argument("-i", "--tile_size", type=int, default=512)
        self.add_argument("--load_sd_vae", default=False, action='store_true')

    def list_checkpoints(self) -> List[Tuple[Path, Experiment]]:
        res = super().list_checkpoints()
        if self.load_sd_vae:
            name = "AutoEncoderKL"
            exp = Experiment(label=name)
            exp.net_class_name = name
            path = "/home/tim/Downloads/vae-ft-mse-840000-ema-pruned.ckpt"
            exp.net_image_size = 512
            exp.lastepoch_train_loss = 0.0
            exp.lastepoch_val_loss = 0.0
            exp.net_latent_dim = [4, 64, 64]
            res.append((path, exp))
        return res

def get_exp_descrs(exps: List[Experiment]) -> List[List[str]]:
    res: List[List[str]] = list()
    last_val: Dict[str, any] = dict()
    for exp in exps:
        exp_list: List[str] = list()
        exp_list.append(exp.saved_at_relative() + " ago")

        for field in 'nepochs loss_type net_latent_dim net_layers_str lastepoch_val_loss lastepoch_train_loss lastepoch_kl_loss'.split():
            if not hasattr(exp, field):
                last_val.pop(field, None)
                continue
            val = getattr(exp, field)
            if last_val.get(field, None) == val and field != 'net_latent_dim':
                continue
            last_val[field] = val
            if isinstance(val, float):
                val = format(val, ".5f")
            if field == 'lastepoch_val_loss':
                field = "vloss"
            elif field == 'lastepoch_train_loss':
                field = "tloss"
            elif field == 'lastepoch_kl_loss':
                field = "kl_loss"
            elif field == 'net_layers_str':
                layer_parts = val.split("-")
                for i in range(len(layer_parts) - 1):
                    layer_parts[i] += "-"
                layer_parts[0] = f"layers {layer_parts[0]}"
                exp_list.append(layer_parts)
                continue

            exp_list.append(f"{field} {val}")

        for i in range(len(exp_list) - 1):
            exp_list[i] += ","

        res.append(exp_list)
    return res

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    checkpoints = cfg.list_checkpoints()
    if cfg.sort_key == 'time':
        # if sorted by time, process the most recent (highest) first
        checkpoints = list(reversed(checkpoints))
    exps = [exp for _path, exp in checkpoints]

    font_size = 12
    font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    exp_descrs, max_descr_height = \
        image_util.fit_strings_multi(get_exp_descrs(exps), max_width=cfg.tile_size, font=font)

    ncols = len(checkpoints)
    nrows = cfg.num_images
    width = (ncols + 1) * cfg.tile_size
    height = max_descr_height + nrows * cfg.tile_size
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.ImageDraw(image)

    def get_pos(*, row: int, col: int) -> Tuple[int, int]:
        return (col * cfg.tile_size, max_descr_height + row * cfg.tile_size)

    datasets: Dict[int, Dataset] = dict()
    ds_idxs: List[int] = None
    for exp_idx, (path, exp) in list(enumerate(checkpoints)):
        print(f"{exp_idx + 1}/{len(checkpoints)}: {path}:")
        with open(path, "rb") as file:
            net: Union[model_new.VarEncDec, aekl.AutoencoderKL]
            if exp.label != "AutoEncoderKL":
                state_dict = torch.load(path)
                net = dn_util.load_model(state_dict)
                net = net.to(cfg.device)
                net.eval()
            else:
                from diffusers import AutoencoderKL
                net = aekl.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
                net = net.to(cfg.device)
                net.eval()
                net.latent_dim = exp.net_latent_dim
                net.image_size = exp.net_image_size

        latent_dim = net.latent_dim
        image_size = net.image_size

        if image_size not in datasets:
            dataloader, _ = image_util.get_dataloaders(image_size=image_size,
                                                       image_dir=cfg.image_dir,
                                                       batch_size=cfg.batch_size,
                                                       shuffle=False,
                                                       train_split=1.0)
            datasets[image_size] = dataloader.dataset

            if ds_idxs is None:
                dataset = dataloader.dataset
                all_idxs = list(range(len(dataset)))
                random.shuffle(all_idxs)
                ds_idxs = all_idxs[:cfg.num_images]

        imlat = image_latents.ImageLatents(net=net, net_path=path,
                                           batch_size=cfg.batch_size, dataloader=dataloader,
                                           device=cfg.device)
        encouts = imlat.encouts_for_idxs()
        print(f"{len(encouts)=}")