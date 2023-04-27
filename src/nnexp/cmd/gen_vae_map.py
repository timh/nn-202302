from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
import random

import torch
from torch.utils.data import Dataset

from nnexp.utils import cmdline
from nnexp.images import image_util
from nnexp.denoise import dn_util, latent_cache
from nnexp.denoise.models import vae

class Config(cmdline.QueryConfig):
    num_images: int
    image_dir: str
    image_size: int
    do_roundtrip: bool
    exp_width: int
    run_nepochs: int

    load_sd_vae: bool

    def __init__(self):
        super().__init__()
        self.add_argument("-N", "--num_images", type=int, default=10)
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", type=int, default=None)
        self.add_argument("--roundtrip", dest='do_roundtrip', default=False, action='store_true', 
                          help="also render image > encoder > decoder > image")
        self.add_argument("--load_sd_vae", default=False, action='store_true')
        self.add_argument("--run_nepochs", type=int, default=None, help="pick the run with this many epochs, instead of the best")

if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    exps = cfg.list_experiments()
    if cfg.sort_key == 'time':
        # if sorted by time, process the most recent (highest) first
        exps = list(reversed(exps))

    datasets: Dict[int, Dataset] = dict()
    ds_idxs: List[int] = None

    if cfg.image_size is None:
        image_sizes = [exp.net_image_size for exp in exps]
        cfg.image_size = max(image_sizes)
    
    cfg.exp_width = cfg.image_size
    
    font_size = 24 if cfg.do_roundtrip else 12
    font: ImageFont.ImageFont = ImageFont.truetype(Roboto, font_size)

    ncols = len(exps)
    nrows = cfg.num_images
    if cfg.do_roundtrip:
        cfg.exp_width *= 2
        ncols *= 2

    exp_descrs = [image_util.exp_descr(exp) for exp in exps]
    exp_descrs, max_descr_height = \
        image_util.fit_strings_multi(exp_descrs, max_width=cfg.exp_width, font=font)

    width = (ncols + 1) * cfg.image_size
    height = max_descr_height + nrows * cfg.image_size
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.ImageDraw(image)

    def get_pos(*, row: int, col: int) -> Tuple[int, int]:
        return (col * cfg.image_size, max_descr_height + row * cfg.image_size)

    for exp_idx, exp in list(enumerate(exps)):
        if cfg.run_nepochs:
            exp_run = exp.run_for_nepochs(cfg.run_nepochs)
            if exp_run.checkpoint_nepochs != cfg.run_nepochs:
                for run in exp.runs:
                    print(f"run.cp_nepochs = {run.checkpoint_nepochs}")
                raise Exception(f"wanted {cfg.run_nepochs=}, got {exp_run.checkpoint_nepochs=}")
        else:
            exp_run = exp.get_run(loss_type='tloss')
        net_path = exp_run.checkpoint_path
        print(f"{exp_idx + 1}/{len(exps)} {exp.shortcode}: {exp_run.checkpoint_nepochs} epochs: {net_path}:")

        state_dict = torch.load(net_path)
        net: vae.VarEncDec = dn_util.load_model(state_dict)
        net = net.to(cfg.device)
        net.eval()

        latent_dim = net.latent_dim
        image_size = net.image_size

        if image_size not in datasets:
            dataset, _ = image_util.get_datasets(image_size=image_size,
                                                 image_dir=cfg.image_dir,
                                                 train_split=1.0)
            datasets[image_size] = dataset

            if ds_idxs is None:
                all_idxs = list(range(len(dataset)))
                random.shuffle(all_idxs)
                ds_idxs = all_idxs[:cfg.num_images]

        dataset = datasets[image_size]
        cache = latent_cache.LatentCache(net=net, net_path=net_path,
                                         batch_size=cfg.batch_size,
                                         dataset=dataset,
                                         device=cfg.device)

        ds_images_t = cache.get_images(ds_idxs)
        latents = cache.samples_for_idxs(ds_idxs)
        # latents = imglat.latents_for_images(ds_images_t)
        if cfg.do_roundtrip:
            rt_images_t = cache.decode(latents)

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
