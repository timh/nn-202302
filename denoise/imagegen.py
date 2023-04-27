
import sys
from pathlib import Path
from typing import List, Union, Dict, Generator, Tuple
from PIL import Image
import random
import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset
from diffusers.schedulers import DDIMScheduler

sys.path.append("..")
from experiment import Experiment, ExpRun
import image_util
import dn_util
from models import vae, denoise, unet
from latent_cache import LatentCache
import noisegen
import clip_cache

ModelType = Union[vae.VarEncDec, denoise.DenoiseModel, unet.Unet]

class ImageGen:
    image_dir: int
    output_image_size: int
    device: str
    batch_size: int

    _dataset_by_size: Dict[int, Dataset] = dict()
    _all_image_idxs: List[int] = None
    _random_by_dim: Dict[str, Tensor] = dict()
    _cache_by_path: Dict[Path, LatentCache] = dict()

    _clip_cache: clip_cache.ClipCache

    _vae_net: vae.VarEncDec = None
    _vae_path: Path = None

    def __init__(self, *,
                 image_dir: Path, output_image_size: int, 
                 clip_model_name: clip_cache.ClipModelName = None,
                 device: str, batch_size: int):
        self.image_dir = image_dir
        self.output_image_size = output_image_size
        self.device = device
        self.batch_size = batch_size

        if clip_model_name is not None:
            ds = self.get_dataset(512)
            self._clip_cache = \
                clip_cache.ClipCache(dataset=ds, image_dir=image_dir, model_name=clip_model_name, 
                                     batch_size=batch_size, device=device)

    def _load_vae(self, exp: Experiment, run: ExpRun) -> Tuple[vae.VarEncDec, Path]:
        if exp.net_class in ['Unet', 'DenoiseModel', 'DenoiseModel2', 'DenoiseModelNew']:
            vae_path = exp.vae_path
        elif exp.net_class == 'VarEncDec':
            vae_path = run.checkpoint_path
        else:
            raise ValueError(f"can't determine vae path for {exp.shortcode}: {run.checkpoint_path.name}")

        if vae_path == self._vae_path:
            return self._vae_net, self._vae_path

        self._vae_net = dn_util.load_model(vae_path).to(self.device)
        self._vae_net.eval()
        self._vae_path = vae_path

        return self._vae_net, self._vae_path
    
    def for_run(self, exp: Experiment, run: ExpRun,
                deterministic: bool = False, scale_noise: float = 1.0) -> 'ImageGenExp':
        return ImageGenExp(gen=self, exp=exp, run=run, deterministic=deterministic, scale_noise=scale_noise)
    
    def get_dataset(self, image_size: int) -> Dataset:
        if image_size not in self._dataset_by_size:
            dataset = image_util.get_dataset(image_size=image_size, image_dir=self.image_dir)
            self._dataset_by_size[image_size] = dataset

        dataset = self._dataset_by_size[image_size]

        if self._all_image_idxs is None:
            self._all_image_idxs = list(range(len(dataset)))
            random.shuffle(self._all_image_idxs)
        
        return dataset

    def get_random(self, latent_dim: List[int], start_idx: int, end_idx: int) -> List[Tensor]:
        latent_dim_str = str(latent_dim) 
        latents = self._random_by_dim.get(latent_dim_str, None)
        if latents is None or len(latents) < end_idx:
            latent_dim_batch = [end_idx, *latent_dim]
            self._random_by_dim[latent_dim_str] = torch.randn(size=latent_dim_batch, device=self.device)
            if latents is not None:
                self._random_by_dim[latent_dim_str][:len(latents)] = latents
            
        return [lat for lat in self._random_by_dim[latent_dim_str][start_idx : end_idx]]
    
    def get_cache(self, exp: Experiment, run: ExpRun) -> LatentCache:
        self._load_vae(exp, run)
        if self._vae_path not in self._cache_by_path:
            dataset = self.get_dataset(self._vae_net.image_size)
            self._cache_by_path[self._vae_path] = \
                LatentCache(net=self._vae_net, net_path=self._vae_path,
                            batch_size=self.batch_size,
                            dataset=dataset, device=self.device)
        
        return self._cache_by_path[self._vae_path]
    
class ImageGenExp:
    _exp: Experiment
    _run: ExpRun

    _dn_net: dn_util.DNModelType = None
    _dn_path: Path = None

    _vae_net: vae.VarEncDec = None
    _vae_path: Path = None

    _sched: noisegen.NoiseSchedule = None

    def __init__(self, gen: ImageGen, exp: Experiment, run: ExpRun, 
                 deterministic: bool = False, scale_noise: float = 1.0):
        self._gen = gen
        self._exp = exp
        self._run = run

        if exp.net_class not in ['Unet', 'VarEncDec', 'DenoiseModel', 'DenoiseModel2', 'DenoiseModelNew']:
            raise ValueError(f"can't handle {exp.net_class=}")

        if self._exp.net_class in ['Unet', 'DenoiseModel', 'DenoiseModel2', 'DenoiseModelNew']:
            dn_path = run.checkpoint_path
            self._dn_path = dn_path
            self._dn_net = dn_util.load_model(dn_path).to(gen.device)

            self._sched = noisegen.make_noise_sched(exp=exp, 
                                                    deterministic=deterministic,
                                                    scale_noise=scale_noise)

        self._vae_net, self._vae_path = gen._load_vae(exp, run)
        self.cache = gen.get_cache(exp, run)

    @property
    def latent_dim(self) -> List[int]:
        return self._vae_net.latent_dim

    def interpolate_tensors(self, start: Tensor, end: Tensor, steps: int) -> Generator[Tensor, None, None]:
        for step in range(steps):
            frame = torch.lerp(input=start, end=end, weight=step / (steps - 1))
            yield frame
    
    def add_noise(self, latents: List[Tensor], timestep: int) -> List[Tensor]:
        res: List[Tensor] = list()
        for latent_in in latents:
            latent = self._sched.add_noise(orig=latent_in, timestep=timestep)[0]
            res.append(latent)
        return res
    
    def get_random_latents(self, start_idx: int, end_idx: int) -> List[Tensor]:
        return self._gen.get_random(latent_dim=self.latent_dim, 
                                    start_idx=start_idx, end_idx=end_idx)
    
    def get_image_latents(self, *, image_idxs: List[int],
                          shuffled: bool = False) -> List[Tensor]:
        if shuffled:
            image_idxs = [self._gen._all_image_idxs[idx] for idx in image_idxs]
        
        return self.cache.samples_for_idxs(image_idxs)
    
    def get_embeds_text(self, text_list: List[str]) -> List[Tensor]:
        clip_embeds = [self._gen._clip_cache.encode_text(one_text)[0] for one_text in text_list]
        return clip_embeds

    def get_embeds_image(self, image_list: List[Image.Image]) -> List[Tensor]:
        clip_embeds = self._gen._clip_cache.encode_images(image_list)
        return clip_embeds

    def gen_roundtrip(self, *,
                      image_idxs: List[int],
                      shuffled: bool = False) -> Generator[Image.Image, None, None]:
        if shuffled:
            image_idxs = [self._gen._all_image_idxs[idx] for idx in image_idxs]

        latents = self.cache.samples_for_idxs(image_idxs)
        for decoded in self.cache.decode(latents):
            yield image_util.tensor_to_pil(decoded, image_size=self._gen.output_image_size)
        
    def gen_random(self, *, start_idx: int, end_idx: int) -> Generator[Image.Image, None, None]:
        latents = self.get_random_latents(start_idx, end_idx)

        for decoded in self.cache.decode(latents):
            yield image_util.tensor_to_pil(decoded, image_size=self._gen.output_image_size)
    
    def gen_denoise_full(self, *, steps: int, max_steps: int = None,
                         yield_count: int = None,
                         latents: List[Tensor],
                         clip_embeds: Union[str, Tensor, Image.Image, List[Tensor], List[str], List[Image.Image]] = None,
                         clip_scale: Union[List[float], float] = None,
                         clip_guidance: Union[List[float], float] = None) \
                -> Generator[Image.Image, None, None]:
        """Denoise, and return (count) frames of the process. 
        
        For example, if steps=300 and count=10, this will return a frame after denoising
        for 30 steps, 60 steps, 90 steps, etc..
        
        If count is Falsey, just return the last step.
        """
        if max_steps is None and steps > self._sched.timesteps:
            max_steps = steps

        if isinstance(clip_scale, float):
            clip_scale = [clip_scale] * len(latents)
        if isinstance(clip_scale, list):
            clip_scale = torch.tensor(clip_scale)
        
        if isinstance(clip_guidance, Tensor) or isinstance(clip_guidance, float):
            clip_guidance = [clip_guidance] * len(latents)
        if isinstance(clip_guidance, list):
            clip_guidance = torch.tensor(clip_guidance)
        
        if clip_embeds is not None:
            if type(clip_embeds) in [str, Image.Image, Tensor]:
                clip_embeds = [clip_embeds] * len(latents)

            if isinstance(clip_embeds[0], str):
                clip_embeds = self.get_embeds_text(clip_embeds)
            elif isinstance(clip_embeds[0], Image.Image):
                clip_embeds = self.get_embeds_image(clip_embeds)

        for start_idx in range(0, len(latents), self._gen.batch_size):
            end_idx = min(len(latents), start_idx + self._gen.batch_size)
            latent_batch = torch.stack(latents[start_idx : end_idx]).to(self._gen.device)

            clip_embed_batch = None
            if clip_embeds is not None:
                clip_embed_batch = torch.stack(clip_embeds[start_idx : end_idx]).to(self._gen.device, dtype=latent_batch.dtype)
            
            clip_scale_batch = None
            if clip_scale is not None:
                clip_scale_batch = clip_scale[start_idx : end_idx].to(self._gen.device, dtype=latent_batch.dtype)
                clip_scale_batch = clip_scale_batch.view(end_idx - start_idx, 1, 1, 1)
            
            clip_guidance_batch = None
            if clip_guidance is not None:
                clip_guidance_batch = clip_guidance[start_idx : end_idx].to(self._gen.device, dtype=latent_batch.dtype)
                clip_guidance_batch = clip_guidance_batch.view(end_idx - start_idx, 1, 1, 1)
            
            gen_it = self._sched.gen(net=self._dn_net, inputs=latent_batch, 
                                     clip_embed=clip_embed_batch, clip_scale=clip_scale_batch, clip_guidance=clip_guidance_batch,
                                     steps=steps, max_steps=max_steps, yield_count=yield_count,
                                     progress_bar=yield_count is None)
            if yield_count is not None:
                gen_it = tqdm.tqdm(gen_it, total=yield_count)
            for denoised_latent_batch in gen_it:
                denoised_batch = self._vae_net.decode(denoised_latent_batch)
                for denoised in denoised_batch:
                    yield image_util.tensor_to_pil(denoised, image_size=self._gen.output_image_size)

    def gen_denoise_ddim(self, *, steps: int, 
                         latents: List[Tensor],
                         clip_embeds: Union[str, Tensor, Image.Image, List[Tensor], List[str], List[Image.Image]] = None,
                         clip_scale: Union[List[float], float] = None,
                         clip_guidance: Union[List[float], float] = None) \
                -> Generator[Image.Image, None, None]:
        """Denoise, and return the last frame of each process, using the diffusers
        DDIM library. 
        """
        if isinstance(clip_scale, float):
            clip_scale = [clip_scale] * len(latents)
        if isinstance(clip_scale, list):
            clip_scale = torch.tensor(clip_scale)
        
        if isinstance(clip_guidance, Tensor) or isinstance(clip_guidance, float):
            clip_guidance = [clip_guidance] * len(latents)
        if isinstance(clip_guidance, list):
            clip_guidance = torch.tensor(clip_guidance)
        
        if clip_embeds is not None:
            if type(clip_embeds) in [str, Image.Image, Tensor]:
                clip_embeds = [clip_embeds] * len(latents)

            if isinstance(clip_embeds[0], str):
                clip_embeds = self.get_embeds_text(clip_embeds)
            elif isinstance(clip_embeds[0], Image.Image):
                clip_embeds = self.get_embeds_image(clip_embeds)

        ddim = DDIMScheduler(num_train_timesteps=300)
        ddim.set_timesteps(steps)
        for start_idx in range(0, len(latents), self._gen.batch_size):
            end_idx = min(len(latents), start_idx + self._gen.batch_size)
            latent_batch = torch.stack(latents[start_idx : end_idx]).to(self._gen.device)

            clip_embed_batch = None
            if clip_embeds is not None:
                clip_embed_batch = torch.stack(clip_embeds[start_idx : end_idx]).to(self._gen.device, dtype=latent_batch.dtype)
            
            clip_scale_batch = None
            if clip_scale is not None:
                clip_scale_batch = clip_scale[start_idx : end_idx].to(self._gen.device, dtype=latent_batch.dtype)
                clip_scale_batch = clip_scale_batch.view(end_idx - start_idx, 1, 1, 1)
            
            clip_guidance_batch = None
            if clip_guidance is not None:
                clip_guidance_batch = clip_guidance[start_idx : end_idx].to(self._gen.device, dtype=latent_batch.dtype)
                clip_guidance_batch = clip_guidance_batch.view(end_idx - start_idx, 1, 1, 1)

            sample_batch = latent_batch
            for step in tqdm.tqdm(ddim.timesteps):
                amount = self._sched.noise_amount[step]
                amount = torch.stack([amount] * len(sample_batch)).to(self._gen.device)
                if clip_guidance is not None:
                    cond_out = self._dn_net.forward(sample_batch, amount,
                                                    clip_scale=clip_scale_batch,
                                                    clip_embed=clip_embed_batch)
                    uncond_out = self._dn_net.forward(sample_batch, amount)
                    out = uncond_out + clip_guidance_batch * (cond_out - uncond_out)
                else:
                    out = self._dn_net.forward(sample_batch, amount)
                sample_batch = ddim.step(out, step, sample_batch, 0.0).prev_sample

            denoised_batch = self._vae_net.decode(sample_batch)
            for denoised in denoised_batch:
                yield image_util.tensor_to_pil(denoised, image_size=self._gen.output_image_size)

    def gen_lerp(self, 
                 start: Tensor, end: Tensor, 
                 steps: int) -> Generator[Image.Image, None, None]:
        latents = list(self.interpolate_tensors(start=start, end=end, steps=steps))
        for decoded in self.cache.decode(latents):
            yield image_util.tensor_to_pil(decoded, image_size=self._gen.output_image_size)
    
