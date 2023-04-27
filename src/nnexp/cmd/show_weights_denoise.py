# %%
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple

import torch
from torch import Tensor
from nnexp.images import image_util
from nnexp.utils import checkpoint_util
from nnexp.denoise import dn_util, noisegen
from nnexp.denoise.models import denoise, vae

def load_nets() -> Tuple[denoise.DenoiseModel, vae.VarEncDec, Path]:
    shortcode = "pbhyur"

    exp = [exp for exp in checkpoint_util.list_experiments() if exp.shortcode == shortcode][0]
    run = exp.get_run()

    dn_net: denoise.DenoiseModel
    dn_net = dn_util.load_model(run.checkpoint_path).to("cuda")
    # print(dn_net)

    vae_net: vae.VarEncDec
    vae_net = dn_util.load_model(Path(exp.vae_path)).to("cuda")

    return dn_net, vae_net, run.checkpoint_path

"""return latent for a loaded image"""
def load_image(vae_net: vae.VarEncDec, vae_path: Path) -> Tensor:
    import latent_cache
    ds, _ = image_util.get_datasets(image_size=vae_net.image_size, 
                                    image_dir="images.alex+1star-1024",
                                    train_split=1.0)
    cache = latent_cache.LatentCache(net=vae_net, net_path=vae_path,
                                     batch_size=4, dataset=ds, device="cuda")

    latent = cache.samples_for_idxs([0])[0].unsqueeze(0).to("cuda")

    return latent

"""return latent for a rendered image"""
@torch.no_grad()
def render_image(vae_net: vae.VarEncDec) -> Tensor:
    from PIL import Image, ImageDraw, ImageFont
    from fonts.ttf import Roboto

    image = Image.new("RGB", (vae_net.image_size, vae_net.image_size))
    draw = ImageDraw.ImageDraw(image)
    font = ImageFont.truetype(Roboto, vae_net.image_size)

    draw.text((0, 0), text="D", fill="white", font=font)

    image_t = image_util.pil_to_tensor(image=image, net_size=vae_net.image_size)
    image_t = image_t.unsqueeze(0).to("cuda").detach()
    latent = vae_net.encode(image_t).detach().cpu()

    return latent


with torch.no_grad():
    dn_net, vae_net, vae_path = load_nets()
    # latent = load_image(vae_net, vae_path)
    latent = render_image(vae_net)

    sched = noisegen.make_noise_schedule('cosine', 300, 'normal')

    noised, noise, amount, _step = sched.add_noise(orig=latent, timestep=100)
    noised = noised.to("cuda")
    amount = amount.to("cuda").unsqueeze(0)

    pred_noise, down_attn_list, up_attn_list = dn_net(noised, time=amount, return_attn=True)

    print(f"{len(down_attn_list)=}")
    print(f"  {len(up_attn_list)=}")

    print(f"{vae_net.latent_dim=}")

    dn_net = dn_net.cpu()
    vae_net = vae_net.cpu()
    dn_net = None
    vae_net = None

    base = 10
    nrows = 3
    ncols = max(3, len(down_attn_list), len(up_attn_list))
    fig = plt.figure(1, figsize=(ncols * base, nrows * base))

    axes_list = fig.subplots(nrows=nrows, ncols=ncols)

    pred_noise_img = pred_noise[0].mean(dim=0).detach().cpu()
    orig_img = latent[0][0].detach().cpu()
    noised_img = noised[0][0].detach().cpu()
    axes_list[0, 0].imshow(orig_img, label="orig")
    axes_list[0, 1].imshow(noised_img, label="noised")
    axes_list[0, 2].imshow(pred_noise_img, label="pred noise")

    def to_image_3(img_t: Tensor, is_weight: bool = False) -> Tensor:
        img_t = img_t.detach().cpu()
        # img_t = torch.softmax(img_t, dim=-1)
        if is_weight:
            print(f"weight: {img_t.shape=}")
            chan, width, height = img_t.shape
            img_t = img_t[:3].permute(1, 2, 0)
            print(f"  {img_t.shape=}")
        else:
            print(f"attn: {img_t.shape=}")
            img_t = img_t.mean(dim=0)
            print(f"  {img_t.shape=}")
        
        amax = torch.amax(img_t)
        amin = torch.amin(img_t)
        img_t = (img_t - amin) / (amax - amin)
        print(f"{img_t.mean()=} {img_t.min()=} {img_t.max()=}")


        return img_t

    # down
    for col, down_attn in enumerate(down_attn_list):
        down_attn = to_image_3(down_attn[0])
        axes_list[1, col].imshow(down_attn)

    # up
    for col, up_attn in enumerate(up_attn_list):
        up_attn = to_image_3(up_attn[0])
        axes_list[2, col].imshow(up_attn)


