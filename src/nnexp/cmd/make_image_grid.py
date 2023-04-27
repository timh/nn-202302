# %%
import math
from PIL import Image, ImageDraw, ImageFont
import fonts.ttf

import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nnexp import checkpoint_util
from nnexp.utils import cmdline
from nnexp.experiment import Experiment
from nnexp.images import image_util
from nnexp.denoise import dn_util, latent_cache
from nnexp.denoise.models import vae

class Config(cmdline.BaseConfig):
    image_dir: str
    image_size: int
    shortcode: str
    show_latent: bool

    def __init__(self):
        super().__init__()
        self.add_argument("-d", "--image_dir", default="1star-2008-now-1024px")
        self.add_argument("-i", "--image_size", default=64, type=int)
        self.add_argument("-c", "--shortcode", default=None)
        self.add_argument("--show_latent", default=False, action='store_true')

# checkpoints, num_images, num_frames, frames_per_pair
if __name__ == "__main__":
    cfg = Config()
    cfg.parse_args()

    dl_image_size = cfg.image_size
    image_size = cfg.image_size

    vae_net: vae.VarEncDec = None
    exp: Experiment = None

    if cfg.shortcode:
        exp = checkpoint_util.find_experiment(cfg.shortcode)
        if exp is None:
            raise Exception(f"can't find VarEncDec with shortcode {cfg.shortcode}")

        best_run = exp.get_run(loss_type='vloss')
        cp_path = best_run.checkpoint_path

        vae_net = dn_util.load_model(cp_path)
        vae_net = vae_net.to(cfg.device)
        dl_image_size = vae_net.image_size


    dataset = image_util.get_dataset(image_size=dl_image_size, image_dir=cfg.image_dir)
    if vae_net is not None:
        cache = latent_cache.LatentCache(net=vae_net, net_path=cp_path, batch_size=cfg.batch_size, dataset=dataset, device=cfg.device)
        all_samples = [veo.sample() for veo in cache.encouts_for_idxs()]
        all_samples_batch = torch.stack(tensors=all_samples)
        # all_samples_batch = all_samples_batch.softmax(dim=-1)
        all_samples_min = torch.min(all_samples_batch)
        all_samples_max = torch.max(all_samples_batch)
        # all_samples_batch = (all_samples_batch - all_samples_min) / (all_samples_max - all_samples_min)
        # all_samples_batch = all_samples_batch ** 2
        print(f"after norm, {all_samples_batch.shape=}, {torch.min(all_samples_batch)=}, {torch.max(all_samples_batch)=}")
        all_samples = [sample for sample in all_samples_batch]
        dataset = list(zip(all_samples, all_samples))
        # dataset = dataloader.EncoderDataset(vae_net=vae_net, vae_net_path=cp_path, batch_size=cfg.batch_size, device=cfg.device, base_dataset=dataset)
        # cache = dataset.cache
    img_dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)

    pad_image = 2
    pad_ten = 10
    pad_size = image_size + pad_image

    ncols = int(len(dataset) ** 0.5) // 10 * 10
    ncols_ten = ncols // 10
    nrows = math.ceil(len(dataset) / ncols)
    width = pad_size * ncols + pad_ten * (ncols // 10)
    height = pad_size * nrows + pad_ten * (nrows // 10)
    print(f"{width=} {height=}")

    image = Image.new("RGB", (width, height))

    font_size = max(10, image_size // 10)
    font = ImageFont.truetype(fonts.ttf.Roboto, font_size)
    draw = ImageDraw.ImageDraw(image)

    dl_iter = iter(img_dataloader)
    for batch in tqdm.tqdm(range(len(img_dataloader))):
        inputs, _truth = next(dl_iter)

        if vae_net and cfg.show_latent:
            # show the first 3 channels of the latent representation
            # images = (inputs - all_min) / (all_max - all_min)
            # images = images[:, :3, :, :]
            images = inputs[:, :3, :, :]
        elif vae_net:
            # decode the latent representation then normalize it.
            images = vae_net.decode(inputs.to(cfg.device))
        else:
            # just pass the images through
            images = inputs

        for img_idx, img_t in enumerate(images):
            img_idx = batch * cfg.batch_size + img_idx

            row = img_idx // ncols
            col = img_idx % ncols
            text = str(img_idx)
            imgx = pad_size * col + pad_ten * (col // 10)
            imgy = pad_size * row + pad_ten * (row // 10)

            img = image_util.tensor_to_pil(img_t, image_size)
            image.paste(img, (imgx, imgy))

            image_util.annotate(image=image, draw=draw, font=font, text=text,
                                upper_left=(imgx, imgy), within_size=image_size)

    image_name = "image-grid--" + Path(cfg.image_dir).name + f"x{cfg.image_size}"
    if exp is not None:
        image_name += f"--{exp.shortcode}_{exp.nepochs}"
        if cfg.show_latent:
            image_name += ",show_latent"

    filename = f"{image_name}.png"
    image.save(filename)
    print(f"wrote {filename=}")

