from typing import List, Literal, Union
from pathlib import Path
import tqdm
import math
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

from nnexp.images import image_util

class ClipCache:
    _embeddings: List[Tensor] = None
    model_name: str
    text_model: CLIPTextModel = None
    tokenizer: CLIPTokenizer = None

    @torch.no_grad()
    def __init__(self, *, 
                 dataset: Dataset, image_dir: Path, 
                 model_name: str = "openai/clip-vit-large-patch14",
                 batch_size: int, device: str):

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        short_model = model_name.split("/")[-1]

        cache_path = Path(image_dir, f"clip-embeds--{short_model}.ckpt")
        if cache_path.exists():
            print(f"loading {cache_path}")
            self._embeddings = torch.load(cache_path)
            return

        # generate the clip embeddings
        self._embeddings = list()

        print(f"generating {cache_path}...")
        model = CLIPModel.from_pretrained(model_name).to(self.device)
        model.requires_grad_(False)
        text_model = model.text_model
        tokenizer = CLIPTokenizer.from_pretrained(model_name)

        caption_path = Path(image_dir, "captions.txt")
        with open(caption_path, "r") as caption_in:
            all_captions = caption_in.readlines()

        num_images = len(dataset)
        num_batch = math.ceil(num_images / batch_size)

        for start_idx in tqdm.tqdm(range(0, num_images, batch_size), total=num_batch):
            end_idx = min(len(dataset), start_idx + batch_size)
            caption_batch = all_captions[start_idx : end_idx]
            expected_len = end_idx - start_idx
            if len(caption_batch) < expected_len:
                missing = (expected_len - len(caption_batch))
                print(f"filling in {missing} embeds")
                caption_batch[len(caption_batch):] = [""] * missing

            token_ids = tokenizer(caption_batch, padding="max_length", return_tensors="pt").input_ids
            token_ids = token_ids.to(self.device)
            embeds = text_model(token_ids)[0].detach().cpu()
            self._embeddings.extend(embeds)

        print(f"len(dataset) = {len(dataset)}, len(embeds) = {len(self._embeddings)}")
        torch.save(self._embeddings, cache_path)
    
    @torch.no_grad()
    def _load_model(self):
        if self.text_model is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.text_model = self.model.text_model.to(self.device)
            self.model.vision_model.cpu()
            self.model.vision_model = None

    def get_clip_emblen(self) -> int:
        return self._embeddings[0].shape[0]

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> Tensor:
        self._load_model()

        token_ids = self.tokenizer(text, padding="max_length", return_tensors="pt").input_ids
        token_ids = token_ids.to(self.device)
        return self.text_model(token_ids)[0]
    
    # def encode_image(self, image: Image.Image) -> Tensor:
    #     self._load_model()
    #     image_t = self._preprocess(image).unsqueeze(0)
    #     return self._model.encode_image(image_t)

    # def encode_images(self, images: List[Image.Image]) -> List[Tensor]:
    #     self._load_model()
    #     emb_list: List[Tensor] = list()
    #     image_t_list = [self._preprocess(image) for image in images]
    #     for start_idx in range(0, len(image_t_list), self.batch_size):
    #         end_idx = min(len(image_t_list), start_idx + self.batch_size)
    #         image_batch = torch.stack(image_t_list[start_idx : end_idx]).detach().to(self.device)
    #         emb_batch = self._model.encode_image(image_batch)
    #         image_batch = image_batch.cpu()

    #         emb_list.extend(emb_batch)
    #     return emb_list

    def __len__(self) -> int:
        return len(self._embeddings)
    
    def __getitem__(self, idx) -> Tensor:
        return self._embeddings[idx]
