import sys
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
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

class Config(cmdline.QueryConfig):
    # chart_size: int
    num_images: int
    image_dir: str
    image_size: int

    def __init__(self):
        super().__init__()
        # self.add_argument("-c", "--chart_size", type=int, default=1024)
        self.add_argument("-N", "--num_images", type=int, default=10)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", type=int, default=None)

def exp_descrs(exps: List[Experiment]) -> List[List[str]]:
    res: List[List[str]] = list()
    for exp in exps:
        exp_list: List[str] = list()
        exp_list.append(exp.saved_at_relative() + " ago")

        for field in 'nepochs loss_type net_layers_str lastepoch_val_loss lastepoch_train_loss lastepoch_kl_loss'.split():
            if not hasattr(exp, field):
                continue
            val = getattr(exp, field)
            if isinstance(val, float):
                val = format(val, ".5f")
            if field == 'lastepoch_val_loss':
                field = "vloss"
            elif field == 'lastepoch_train_loss':
                field = "tloss"
            elif field == 'lastepoch_kl_loss':
                field = "kl_loss"
            # elif field == 'net_layers_str':
            #     layer_parts = val.split(",")
            #     for i in range(len(layer_parts) - 1):
            #         layer_parts[i] += "-"
            #     layer_parts[0] = f"layers {layer_parts[0]}-"
            #     exp_list.append(layer_parts)
            #     continue

            exp_list.append(f"{field} {val}")

        for i in range(len(exp_list) - 1):
            exp_list[i] += ","

        res.append(exp_list)
    return res

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    checkpoints = cfg.list_checkpoints()
    datasets: Dict[int, Dataset] = dict()
    ds_idxs: List[int] = None


    exps = [exp for _path, exp in checkpoints]
    if cfg.image_size is None:
        image_sizes = [exp.net_image_size for exp in exps]
        cfg.image_size = max(image_sizes)
    
    font: ImageFont.ImageFont = ImageFont.truetype(Roboto, 12)

    ncols = len(checkpoints)
    nrows = cfg.num_images
    # exp_descrs, max_label_height = \
    #     image_util.fit_exp_descrs(exps, max_width=cfg.image_size, font=font)
    exp_descrs, max_descr_height = \
        image_util.fit_strings_multi(exp_descrs(exps), max_width=cfg.image_size, font=font)

    width = (ncols + 1) * cfg.image_size
    height = max_descr_height + nrows * cfg.image_size
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.ImageDraw(image)

    def get_pos(*, row: int, col: int) -> Tuple[int, int]:
        return (col * cfg.image_size, max_descr_height + row * cfg.image_size)

    for exp_idx, (path, exp) in tqdm.tqdm(list(enumerate(checkpoints))):
        with open(path, "rb") as file:
            state_dict = torch.load(path)
            net: model_new.VarEncDec = dn_util.load_model(state_dict)
            net = net.to(cfg.device)

        latent_dim = net.latent_dim
        image_size = net.image_size

        if image_size not in datasets:
            dataloader, _ = image_util.get_dataloaders(image_size=image_size,
                                                       image_dir=cfg.image_dir,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       train_split=1.0)
            datasets[image_size] = dataloader.dataset

            if ds_idxs is None:
                dataset = dataloader.dataset
                all_idxs = list(range(len(dataset)))
                random.shuffle(all_idxs)
                ds_idxs = all_idxs[:cfg.num_images]

        dataset = datasets[image_size]
        imglat = image_latents.ImageLatents(net=net, net_path=path, 
                                            batch_size=cfg.batch_size,
                                            dataloader=dataloader, 
                                            device=cfg.device)
        
        ds_images_t = imglat.get_images(ds_idxs)
        latents = imglat.latents_for_idxs(ds_idxs)

        # render the exp descr in the top row
        descr_x, _ = get_pos(col=exp_idx + 1, row=0)
        descr_pos = (descr_x, 0)
        draw.text(xy=descr_pos, text=exp_descrs[exp_idx], font=font, fill='white')

        # render the true images in the first column (NOTE which will be at 
        # whatever image_size the first net is)
        if exp_idx == 0:
            for row, ds_image_t in enumerate(ds_images_t):
                ds_image = image_util.tensor_to_pil(ds_image_t, cfg.image_size)
                ds_pos = get_pos(row=row, col=0)
                image.paste(ds_image, box=ds_pos)

        # normalize the latent values
        for lat in latents:
            # lat_min = torch.min(lat)
            # lat_max = torch.max(lat)
            # lat.add_(lat_min)
            # lat.div_(lat_max - lat_min)
            lat.sigmoid_()

        # put first 3 channels of latents in R, G, B and render them
        lat_image_t = torch.zeros((3, *net.latent_dim[1:]))
        for row, latent in enumerate(latents):
            lat_image_t[:3] = latent[:3]
            lat_image = image_util.tensor_to_pil(lat_image_t, cfg.image_size)
            lat_pos = get_pos(row=row, col=exp_idx + 1)
            image.paste(lat_image, box=lat_pos)

    image.save("vae-map.png")
