from typing import List, Literal, Union
from pathlib import Path
import tqdm
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import clip, clip.model
import torchvision.transforms

import sys
sys.path.append("..")
import image_util

#                        name                embedding length
ClipModelName = Literal["RN50",            # 1024
                        "RN101",           # 512
                        "RN50x4",          # 640
                        "RN50x16",         # 768
                        "RN50x64",         # 1024
                        "ViT-B/32",        # 512
                        "ViT-B/16",        # 512
                        "ViT-L/14",        # 768
                        "ViT-L/14@336px"]  # 768

class ClipCache:
    _embeddings: List[Tensor] = None
    model_name: ClipModelName

    # NOTE: for saving memory:
    # when using ClipCache purely for cached answers, e.g., during training, these stay as None.
    #
    # as soon as a new image/text are queries, though, they get set.
    _model: clip.model.CLIP = None
    _preprocess: torchvision.transforms.Compose = None

    def __init__(self, *, dataset: Dataset, image_dir: Path, 
                 model_name: ClipModelName = "RN50",
                 batch_size: int, device: str):

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        first_image, truth = dataset[0]
        if isinstance(first_image, list):
            first_image = first_image[0]
        image_size = first_image.shape[1]

        cache_path = Path(image_dir, f"clip-embeddings-{model_name}-{image_size}.ckpt")
        if cache_path.exists():
            print(f"loading {cache_path}")
            self._embeddings = torch.load(cache_path)
            return

        # generate the clip embeddings
        self._embeddings = list()
        model, preprocess = clip.load(model_name, device=device)

        print(f"generating {cache_path}...")
        num_images = len(dataset)
        for start_idx in tqdm.tqdm(list(range(0, num_images, batch_size))):
            end_idx = min(num_images, start_idx + batch_size)

            # (chan, height, width) -> (height, width, chan)
            base_tensors = [dataset[idx][0] for idx in range(start_idx, end_idx)]
            base_images = [image_util.tensor_to_pil(base_t) for base_t in base_tensors]

            preprocessed = [preprocess(image) for image in base_images]
            image_t = torch.stack(preprocessed).to(device)
            image_features = model.encode_image(image_t).detach().cpu()

            for one_embed in image_features:
                self._embeddings.append(one_embed)

        torch.save(self._embeddings, cache_path)
    
    def _load_model(self):
        if self._model is None:
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)

    def get_clip_emblen(self) -> int:
        return self._embeddings[0].shape[0]
    
    def encode_text(self, text: Union[str, List[str]]) -> Tensor:
        self._load_model()

        tokens = clip.tokenize(text).to(self.device)
        return self._model.encode_text(tokens)
    
    def encode_image(self, image: Image.Image) -> Tensor:
        self._load_model()
        image_t = self._preprocess(image).unsqueeze(0)
        return self._model.encode_image(image_t)

    def encode_images(self, images: List[Image.Image]) -> List[Tensor]:
        self._load_model()
        emb_list: List[Tensor] = list()
        image_t_list = [self._preprocess(image) for image in images]
        for start_idx in range(0, len(image_t_list), self.batch_size):
            end_idx = min(len(image_t_list), start_idx + self.batch_size)
            image_batch = torch.stack(image_t_list[start_idx : end_idx]).detach().to(self.device)
            emb_batch = self._model.encode_image(image_batch)
            image_batch = image_batch.cpu()

            emb_list.extend(emb_batch)
        return emb_list

    def __len__(self) -> int:
        return len(self._embeddings)
    
    def __getitem__(self, idx) -> Tensor:
        return self._embeddings[idx]
