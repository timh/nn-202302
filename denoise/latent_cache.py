from typing import List, Tuple, Union, Generator
import heapq
from pathlib import Path
import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import diffusers.models.autoencoder_kl as aekl

import sys
sys.path.append("..")
from models.mtypes import VarEncoderOutput
import image_util
from models import vae, denoise

# TODO: while this shouldn't be dependent on Experiment and ExpRun, maybe net_path
# should be?
"""
utility for encoding, decoding from enc/dec models, saving the underlying latents
against a dataset, and sampling results.
"""
ModelType = Union[vae.VarEncDec, denoise.DenoiseModel, aekl.AutoencoderKL]
class LatentCache:
    batch_size: int
    device: str

    net: ModelType
    dataset: Dataset
    image_size: int

    _encouts_for_dataset: List[VarEncoderOutput]

    def __init__(self, *,
                 net: ModelType, net_path: Path = None,
                 batch_size: int,
                 dataset: Dataset, 
                 device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.net = net
        self.net_path = net_path
        self._encouts_for_dataset = None

        image = self.dataset[0][0]
        _chan, self.image_size, _height = image.shape
    
    def _batch_gen(self, tensors: List[Tensor]) -> List[Tensor]:
        for idx in range(0, len(tensors), self.batch_size):
            sublist = tensors[idx : idx + self.batch_size]
            batch = torch.stack(sublist)
            yield batch.detach().to(self.device)
    
    def _batch_list(self, tensors: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for batch in self._batch_gen(tensors):
            res.extend(batch.tolist())
        return res

    def _load_latents(self):
        nimages = len(self.dataset)

        encouts_path: Path = None
        if self.net_path is not None:
            encouts_filename = str(self.net_path).replace(".ckpt", f"-{nimages}n_{self.image_size}x.encout-ckpt")
            encouts_path = Path(encouts_filename)
            if encouts_path.exists():
                # BUG: this doesn't pay attention to the dir the latents came from
                # so if latents came from a different dataloader/image dir that just
                # happens to have the same number of images, the cached latents will
                # be invalid.
                encouts = torch.load(encouts_path)
                
                print(f"  loaded {nimages} latents from {encouts_path}")
                self._encouts_for_dataset = encouts
                return

        print(f"generating {nimages} latents...")
        self._encouts_for_dataset = list()
        for start in tqdm.tqdm(range(0, len(self.dataset), self.batch_size)):
            end = min(start + self.batch_size, len(self.dataset))

            image_list = [self.dataset[idx][0] for idx in range(start, end)]
            enc_outs = self.encouts_for_images(image_list)
            self._encouts_for_dataset.extend(enc_outs)
        
        if encouts_path is not None:
            print(f"saving {nimages} latents to {encouts_path}")
            with open(encouts_path, "wb") as file:
                torch.save(self._encouts_for_dataset, file)
    
    def get_images(self, img_idxs: List[int]) -> List[Tensor]:
        return [self.dataset[idx][0] for idx in img_idxs]

    def samples_for_idxs(self, img_idxs: List[int]) -> List[Tensor]:
        if self._encouts_for_dataset is None:
            self._load_latents()

        return [self._encouts_for_dataset[idx].sample() for idx in img_idxs]

    def encouts_for_idxs(self, img_idxs: List[int] = None) -> List[VarEncoderOutput]:
        if self._encouts_for_dataset is None:
            self._load_latents()

        if img_idxs is None:
            return self._encouts_for_dataset

        return [self._encouts_for_dataset[idx] for idx in img_idxs]
    
    def encouts_for_images(self, image_tensors: List[Tensor]) -> List[VarEncoderOutput]:
        res: List[VarEncoderOutput] = list()

        if isinstance(self.net, aekl.AutoencoderKL):
            for image_batch in self._batch_gen(image_tensors):
                image_batch = image_batch.to(self.device)

                aekl_out: aekl.AutoencoderKLOutput = self.net.encode(image_batch)
                lat_dist = aekl_out.latent_dist

                veo = VarEncoderOutput(mean=lat_dist.mean, logvar=lat_dist.logvar)
                res.append(veo.detach().cpu())

        elif isinstance(self.net, denoise.DenoiseModel):
            for image_batch in self._batch_gen(image_tensors):
                image_batch = image_batch.to(self.device)

                out = self.net.encode(image_batch)
                std = torch.zeros_like(out)
                veo_out = VarEncoderOutput(mean=out, std=std)
                res.append(veo_out.detach().cpu())

        else:
            for image_batch in self._batch_gen(image_tensors):
                image_batch = image_batch.to(self.device)

                veo_out: VarEncoderOutput = \
                    self.net.encode(image_batch, return_veo=True)
                res.append(veo_out.detach().cpu())

        return [one_image_res
                for one_batch in res
                for one_image_res in one_batch.to_list()]

    def samples_for_images(self, image_tensors: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for one_image_res in self.encouts_for_images(image_tensors):
            res.append(one_image_res.sample())
        return res
    
    def encode(self, image_tensors: List[Tensor]) -> List[Tensor]:
        return self.encouts_for_images(image_tensors)
    
    def decode(self, latents: List[Tensor]) -> Generator[Tensor, None, None]:
        res: List[Tensor] = list()
        for latent_batch in self._batch_gen(latents):
            images = self.net.decode(latent_batch)
            if isinstance(images, aekl.DecoderOutput):
                images = images.sample
            images = images.detach().cpu()
            for image in images:
                yield image
    
    def find_closest_n(self, *, 
                       src_idx: int, 
                       src_encout: VarEncoderOutput = None, 
                       n: int) -> List[Tuple[int, VarEncoderOutput]]:
        if self._encouts_for_dataset is None:
            self._load_latents()

        if src_encout is None:
            src_encout = self.encouts_for_images([src_idx])
        # print(f"looking for closest images to {src_idx=}")

        # index, distance, latent
        all_distances: List[Tuple[int, float, Tensor]] = list()
        for encout_idx, encout in enumerate(self._encouts_for_dataset):
            if encout_idx == src_idx:
                continue

            distance = (((encout.mean - src_encout.mean) ** 2).sum() ** 0.5).item()
            all_distances.append((encout_idx, distance, encout))

        all_distances = sorted(all_distances, key=lambda tup: tup[1])
        best_distances = all_distances[:n]

        res = [(idx, encout) for idx, _dist, encout in best_distances]
        return res

