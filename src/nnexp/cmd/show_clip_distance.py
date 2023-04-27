# %%
from typing import List, Tuple
import math
import tqdm
import sys
sys.path.append("..")

import clip_cache
import latent_cache
from nnexp.denoise import dn_util
from nnexp.utils import checkpoint_util
from nnexp.images import image_util

import torch
from torch import Tensor

image_dir = "images.2018-2020"
batch_size = 4

vae_shortcode = "ncqvtq"
vae_exp = checkpoint_util.find_experiment(vae_shortcode)
vae_path = vae_exp.get_run().checkpoint_path
vae_net = dn_util.load_model(vae_path).to("cuda")

# conv2flat_shortcode = "abzdjb" # conv
conv2flat_shortcode = "zlpkai" # conv

# conv2flat_shortcode = "saobnk" # linear
conv2flat_exp = checkpoint_util.find_experiment(conv2flat_shortcode)
conv2flat_path = conv2flat_exp.get_run().checkpoint_path
conv2flat_net = dn_util.load_model(conv2flat_path).to("cuda")

dataset512 = image_util.get_dataset(image_size=512, image_dir=image_dir)
lat_c = latent_cache.LatentCache(net=vae_net, net_path=vae_path, batch_size=batch_size, dataset=dataset512, device="cuda")
clip_512 = clip_cache.ClipCache(dataset=dataset512, image_dir=image_dir, batch_size=batch_size, device="cuda")

dataset64 = image_util.get_dataset(image_size=64, image_dir=image_dir)
clip_64 = clip_cache.ClipCache(dataset=dataset64, image_dir=image_dir, batch_size=batch_size, device="cuda")

# images: 512px and 64px
img512_embeds: List[Tensor] = list()
img64_embeds: List[Tensor] = list()
for idx in range(len(dataset512)):
    img512_embeds.append(clip_512[idx].to(dtype=torch.float32))
    img64_embeds.append(clip_64[idx].to(dtype=torch.float32))

print(f"{len(img512_embeds)=}")
nimages = len(dataset512)


# %%

# latents
with torch.no_grad():
    lat_embeds: List[Tensor] = list()
    total = math.ceil(nimages / batch_size)
    for start_idx in tqdm.tqdm(range(0, nimages, batch_size), total=total):
        end_idx = min(nimages, start_idx + batch_size)
        latents = lat_c.samples_for_idxs(list(range(start_idx, end_idx)))
        latent_batch = torch.stack(latents).to("cuda")
        
        # get rid of extra dimensions
        emb_batch = conv2flat_net(latent_batch)
        emb_list = [emb for emb in emb_batch]
        lat_embeds.extend(emb_list)

print(f"{len(lat_embeds)=}")

# %%
def print_stats(label: str, t: Tensor):
    print(f"{label}: mean {t.mean():.5f}, min {t.min():.5f}, max {t.max():.5f}, std {t.std():.5f}")

img512_embeds_t = torch.stack(img512_embeds).cpu()
img64_embeds_t = torch.stack(img64_embeds).cpu()
lat_embeds_t = torch.stack(lat_embeds).cpu()

print_stats("images-512", img512_embeds_t)
print_stats(" images-64", img64_embeds_t)
print_stats("   latents", lat_embeds_t)

def find_n_closest(src_idx: int, embeds: List[Tensor], n: int) -> List[int]:
    distances: List[Tuple[float, int]] = list()

    # idx0
    src_embed = embeds[src_idx]

    for idx, search_embed in enumerate(embeds):
        dist = torch.sqrt(torch.sum((src_embed - search_embed) ** 2))
        distances.append((dist.item(), idx))

    distances = sorted(distances)
    _dists, idxs = zip(*distances)
    return idxs[:n]

nimages = 16
src_idx = torch.randint(0, nimages, size=(1,)).item()
print(f"{src_idx=}")

best_imgs_512 = find_n_closest(src_idx, img512_embeds, n=nimages)
print("best_imgs_512:", ", ".join(map(str, best_imgs_512)))

best_imgs_64 = find_n_closest(src_idx, img64_embeds, n=nimages)
print(" best_imgs_64:", ", ".join(map(str, best_imgs_64)))

best_lats = find_n_closest(src_idx, lat_embeds, n=nimages)
print("best_lats:", ", ".join(map(str, best_lats)))

cell_size = 64

img_grid_512 = image_util.ImageGrid(ncols=4, nrows=4, image_size=cell_size)
for i, img_idx in enumerate(best_imgs_512):
    image_t = lat_c.get_images([img_idx])[0]
    image = image_util.tensor_to_pil(image_t, image_size=cell_size)
    img_grid_512.draw_image(col=i % 4, row=i // 4, image=image)

img_grid_64 = image_util.ImageGrid(ncols=4, nrows=4, image_size=cell_size)
for i, img_idx in enumerate(best_imgs_64):
    image_t = lat_c.get_images([img_idx])[0]
    image = image_util.tensor_to_pil(image_t, image_size=cell_size)
    img_grid_64.draw_image(col=i % 4, row=i // 4, image=image)

lat_grid = image_util.ImageGrid(ncols=4, nrows=4, image_size=cell_size)
for i, img_idx in enumerate(best_lats):
    image_t = lat_c.get_images([img_idx])[0]
    image = image_util.tensor_to_pil(image_t, image_size=cell_size)
    lat_grid.draw_image(col=i % 4, row=i // 4, image=image)


import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(10, 30))

img_axes_512, img_axes_64, lat_axes = fig.subplots(3, 1)
img_axes_512.imshow(img_grid_512._image)
img_axes_64.imshow(img_grid_64._image)
lat_axes.imshow(lat_grid._image)

# %%

# %%
