from typing import List, Tuple, Union, Callable
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
import model_new
from model_new import VarEncoderOutput
import image_util

device = "cuda"

ModelType = Union[model_new.VarEncDec, aekl.AutoencoderKL]
class ImageLatents:
    batch_size: int
    device: str

    net: ModelType
    dataloader: DataLoader
    image_size: int

    _encouts_for_dataset: List[VarEncoderOutput]

    def __init__(self, *,
                 net: ModelType, net_path: Path = None,
                 batch_size: int,
                 dataloader: DataLoader, 
                 device: str):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.net = net
        self.net_path = net_path
        self._encouts_for_dataset = None

        image_batch, _truth = next(iter(self.dataloader))
        image = image_batch[0]
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
        nimages = len(self.dataloader.dataset)

        encouts_path: Path = None
        if self.net_path is not None:
            encouts_filename = str(self.net_path).replace(".ckpt", f"-{nimages}n_{self.image_size}x.encout-ckpt")
            encouts_path = Path(encouts_filename)
            if encouts_path.exists():
                # BUG: this doesn't pay attention to the dir the latents came from
                # so if latents came from a different dataloader/image dir that just
                # happens to have the same number of images, the cached latents will
                # be invalid.
                with open(encouts_path, "rb") as file:
                    encouts = torch.load(file)
                
                print(f"  loaded {nimages} latents from {encouts_path}")
                self._encouts_for_dataset = encouts
                return

        print(f"generating {nimages} latents...")
        self._encouts_for_dataset = list()
        dataloader_it = iter(self.dataloader)
        for _ in tqdm.tqdm(range(len(self.dataloader))):
            image_batch, _truth = next(dataloader_it)
            image_list = [image for image in image_batch]
            enc_outs = self.encouts_for_images(image_list)
            self._encouts_for_dataset.extend(enc_outs)
        
        if encouts_path is not None:
            print(f"saving {nimages} latents to {encouts_path}")
            with open(encouts_path, "wb") as file:
                torch.save(self._encouts_for_dataset, file)
    
    def get_images(self, img_idxs: List[int]) -> List[Tensor]:
        return [self.dataloader.dataset[idx][0] for idx in img_idxs]

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
        return self.latents_for_images(image_tensors)
    
    def decode(self, latents: List[Tensor]) -> List[Tensor]:
        res: List[Tensor] = list()
        for latent_batch in self._batch_gen(latents):
            images = self.net.decode(latent_batch)
            if isinstance(images, aekl.DecoderOutput):
                images = images.sample
            images = images.detach().cpu()
            res.extend([image for image in images])
        
        return res
    
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

def image_latents(*,
                  net: model_new.VarEncDec, 
                  net_path: Path = None,
                  image_dir: str,
                  batch_size: int,
                  device: str) -> ImageLatents:
    dataloader, _ = image_util.get_dataloaders(image_size=net.image_size, 
                                               image_dir=image_dir, 
                                               batch_size=batch_size,
                                               train_split=1.0,
                                               shuffle=False)
    return ImageLatents(net=net, net_path=net_path, 
                        batch_size=batch_size, 
                        dataloader=dataloader, device=device)


# NOTE: shuffle is controlled by underlying dataloader.
class EncoderDataset(Dataset):
    _encouts: List[VarEncoderOutput]

    def __init__(self, *,
                 net: model_new.VarEncDec, net_path: Path = None,
                 enc_batch_size: int,
                 dataloader: DataLoader, device: str):
        imglat = ImageLatents(net=net, net_path=net_path, batch_size=enc_batch_size,
                              dataloader=dataloader, device=device)

        imglat._load_latents()
        self._encouts = imglat._encouts_for_dataset

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        encout = self._encouts[idx]
        # return torch.stack([encout.mean, encout.std])
        return encout.sample()
        # if not isinstance(idx, slice):
        #     start, end, skip = idx
        #     return self._encouts[]
        #     raise NotImplemented(f"slice not implemented: {idx=}")
        # return self._encouts[idx]
    
    def __len__(self) -> int:
        return len(self._encouts)

