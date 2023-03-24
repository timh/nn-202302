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
    do_roundtrip: bool
    exp_width: int

    def __init__(self):
        super().__init__()
        # self.add_argument("-c", "--chart_size", type=int, default=1024)
        self.add_argument("-N", "--num_images", type=int, default=10)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", type=int, default=None)
        self.add_argument("--roundtrip", dest='do_roundtrip', default=False, action='store_true', 
                          help="also render image > encoder > decoder > image")
    
    def parse_args(self) -> 'Config':
        super().parse_args()

def get_exp_descrs(exps: List[Experiment]) -> List[List[str]]:
    res: List[List[str]] = list()
    last_val: Dict[str, any] = dict()
    for exp in exps:
        exp_list: List[str] = list()
        exp_list.append(exp.saved_at_relative() + " ago")

        for field in 'nepochs loss_type net_latent_dim net_layers_str lastepoch_val_loss lastepoch_train_loss lastepoch_kl_loss'.split():
            if not hasattr(exp, field):
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

    datasets: Dict[int, Dataset] = dict()
    ds_idxs: List[int] = None

    exps = [exp for _path, exp in checkpoints]
    if cfg.image_size is None:
        image_sizes = [exp.net_image_size for exp in exps]
        cfg.image_size = max(image_sizes)
    
    cfg.exp_width = cfg.image_size
    
    font_size = 24 if cfg.do_roundtrip else 12
    font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    ncols = len(checkpoints)
    nrows = cfg.num_images
    if cfg.do_roundtrip:
        cfg.exp_width *= 2
        ncols *= 2

    exp_descrs, max_descr_height = \
        image_util.fit_strings_multi(get_exp_descrs(exps), max_width=cfg.exp_width, font=font)

    width = (ncols + 1) * cfg.image_size
    height = max_descr_height + nrows * cfg.image_size
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.ImageDraw(image)

    def get_pos(*, row: int, col: int) -> Tuple[int, int]:
        return (col * cfg.image_size, max_descr_height + row * cfg.image_size)

    for exp_idx, (path, exp) in list(enumerate(checkpoints)):
        print(f"{exp_idx + 1}/{len(checkpoints)}: {path}:")
        with open(path, "rb") as file:
            state_dict = torch.load(path)
            net: model_new.VarEncDec = dn_util.load_model(state_dict)
            net = net.to(cfg.device)
            net.eval()

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
        # latents = imglat.latents_for_images(ds_images_t)
        if cfg.do_roundtrip:
            rt_images_t = imglat.decode(latents)

        # render the exp descr in the top row
        if cfg.do_roundtrip:
            exp_col = exp_idx * 2 + 1
        else:
            exp_col = exp_idx + 1
        descr_x, _ = get_pos(col=exp_col, row=0)
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
        latents = [latent.clone() for latent in latents]
        for latent in latents:
            # lat_min = torch.min(latent)
            # lat_max = torch.max(latent)
            # latent.add_(lat_min)
            # latent.div_(lat_max - lat_min)
            latent.sigmoid_()

        def annotate(pos: Tuple[int, int], text: str):
            image_util.annotate(image=image, draw=draw, font=font,
                                text=text, upper_left=pos,
                                within_size=cfg.image_size,
                                ref="lower_left")
            
        lat_image_t = torch.zeros((3, *net.latent_dim[1:]))
        for row, latent in enumerate(latents):
            if cfg.do_roundtrip:
                rt_col = exp_col
                rt_image = image_util.tensor_to_pil(rt_images_t[row], cfg.image_size)
                rt_pos = get_pos(row=row, col=rt_col)
                image.paste(rt_image, box=rt_pos)
                annotate(rt_pos, "roundtrip")

                lat_col = exp_col + 1
            else:
                lat_col = exp_col

            # put first 3 channels of latents in R, G, B and render them
            lat_image_t[:3] = latent[:3]
            lat_image = image_util.tensor_to_pil(lat_image_t, cfg.image_size)
            lat_pos = get_pos(row=row, col=lat_col)
            image.paste(lat_image, box=lat_pos)
            annotate(lat_pos, "latent")
        
        net.cpu()
        print()

    image.save("vae-map.png")
