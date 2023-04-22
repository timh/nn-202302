import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import einops

"""
(1,) -> (emblen,)
"""
class SinPositionEmbedding(nn.Module):
    def __init__(self, emblen: int):
        super().__init__()
        self.emblen = emblen
    
    def forward(self, time: Tensor) -> Tensor:
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        if len(time.shape) == 2 and time.shape[-1] == 1:
            time = time.view((time.shape[0],))
        assert len(time.shape) == 1

        half_dim = self.emblen // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=time.device)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.emblen % 2 == 1:  # zero pad
            emb = F.pad(emb, (0,1,0,0))
        return emb

class SelfAttention(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, nheads: int, kernel_size: int = 3):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.scale = nheads ** -0.5
        self.nheads = nheads
        self.attn_combined = nn.Conv2d(in_chan, out_chan * 3, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)

        # nn.init.normal_(self.attn_combined.weight, mean=0, std=0.02)
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch, chan, height, width = inputs.shape

        # calc query, key, value all at the same time.
        qkv = self.attn_combined(inputs).chunk(3, dim=1)

        # note: nheads * headlen = in_chan
        #     (batch, nheads * headlen, height, width)
        #  -> (batch, nheads, headlen, height * width)
        query, key, value = map(
            lambda t: einops.rearrange(t, "b (nh hl) y x -> b nh hl (y x)", nh=self.nheads),
            qkv
        )

        out = F.scaled_dot_product_attention(query, key, value)
        out = einops.rearrange(out, "b nh hl (y x) -> b (nh hl) y x", y=height)

        out = self.norm(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, in_size: int, nheads: int, clip_emblen: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.nheads = nheads

        self.clip_emblen = clip_emblen
        self.kv_linear = nn.Linear(clip_emblen, nheads * in_size * in_size)
        self.kv_unflatten = nn.Unflatten(1, (nheads, in_size, in_size))
        self.attn_kv = nn.Conv2d(nheads, nheads * 2, kernel_size=3, padding=1, bias=False)

        self.attn_q = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)
    
    def forward(self, inputs: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tensor:
        batch, _chan, height, _width = inputs.shape
        if clip_embed is None:
            clip_embed = torch.zeros((batch, self.clip_emblen), device=inputs.device)
        if clip_scale is None:
            clip_scale = 1.0
        
        # calculate key, value on the clip embedding
        clip_flat = self.kv_linear(clip_embed)
        clip_out = self.kv_unflatten(clip_flat)
        key, value = self.attn_kv(clip_out).chunk(2, dim=1)

        # calculate query on the inputs
        query = self.attn_q(inputs)

        # note: nheads * headlen = in_chan
        #     (batch, nheads * headlen, height, width)
        #  -> (batch, nheads, headlen, height * width)
        query, key, value = map(
            lambda t: einops.rearrange(t, "b (nh hl) y x -> b nh hl (y x)", nh=self.nheads),
            [query, key, value]
        )

        out = F.scaled_dot_product_attention(query, key, value)
        out = einops.rearrange(out, "b nh hl (y x) -> b (nh hl) y x", y=height)

        out = self.norm(out)
        out = out * clip_scale
        return out

