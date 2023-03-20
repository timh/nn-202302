from typing import List, Tuple, Union, Callable
import heapq
from pathlib import Path
import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append("..")
import model_new
import dn_util

device = "cuda"

class ImageLatents:
    image_dir: Path
    batch_size: int
    device: str

    net: model_new.VarEncDec
    dataloader: DataLoader
    encoder_fn: Callable[[Tensor], Tensor]
    decoder_fn: Callable[[Tensor], Tensor]

    _latents_for_dataset: List[Tensor]

    def __init__(self, net: model_new.VarEncDec, 
                 image_dir: Path, batch_size: int, device = "cpu"):
        train_dl, _ = dn_util.get_dataloaders(disable_noise=True, 
                                              image_size=net.image_size, 
                                              image_dir=image_dir, 
                                              batch_size=batch_size,
                                              train_split=1.0,
                                              shuffle=False)
        self.dataloader = train_dl
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.device = device
        self.net = net
        self.encoder_fn = net.encode
        self.decoder_fn = net.decode
        self._latents_for_dataset = None
    
    def _batch_gen(self, tensors: List[Tensor]) -> List[Tensor]:
        for idx in range(0, len(tensors), self.batch_size):
            sublist = tensors[idx : idx + self.batch_size]
            batch = torch.stack(sublist)
            yield batch.to(self.device)
    
    def _batch_list(self, tensors: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for batch in self._batch_gen(tensors):
            res.extend(batch.tolist())
        return res

    def _load_latents(self):
        self._latents_for_dataset = list()
        dataloader_it = iter(self.dataloader)
        print("generating latents...")
        for _ in tqdm.tqdm(range(len(self.dataloader))):
            image_batch, _truth = next(dataloader_it)
            image_batch = image_batch.to(self.device)
            latents_batch = self.encoder_fn(image_batch).detach()
            for latent in latents_batch:
                self._latents_for_dataset.append(latent)
    
    def get_images(self, img_idxs: List[int]) -> List[Tensor]:
        return [self.dataloader.dataset[idx][0] for idx in img_idxs]

    def latents_for_images(self, image_tensors: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for image_batch in self._batch_gen(image_tensors):
            latents = self.encoder_fn(image_batch)
            res.extend([latent for latent in latents])
        return res
    
    def encode(self, image_tensors: List[Tensor]) -> List[Tensor]:
        return self.latents_for_images(image_tensors)
    
    def decode(self, latents: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for latent_batch in self._batch_gen(latents):
            images = self.decoder_fn(latent_batch)
            res.extend([image for image in images])
        return res
    
    def find_closest_n(self, *, src_idx: int, src_latent: Tensor = None, n: int) -> List[Tuple[int, Tensor]]:
        if self._latents_for_dataset is None:
            self._load_latents()

        if src_latent is None:
            src_latent = self.latents_for_images([src_latent])[0]
        print(f"looking for closest images to {src_idx=}")

        # index, distance, latent
        all_distances: List[Tuple[int, float, Tensor]] = list()
        for latent_idx, latent in enumerate(self._latents_for_dataset):
            if latent_idx == src_idx:
                continue

            distance = (((latent - src_latent) ** 2).sum() ** 0.5).item()
            all_distances.append((latent_idx, distance, latent))

        all_distances = sorted(all_distances, key=lambda tup: tup[1])
        best_distances = all_distances[:n]

        res = [(idx, latent) for idx, _dist, latent in best_distances]
        return res
    
