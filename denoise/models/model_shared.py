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
    def __init__(self, chan: int, nheads: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.nheads = nheads

        self.norm_in = nn.GroupNorm(num_groups=chan, num_channels=chan)
        self.attn_combined = nn.Conv2d(chan, chan * 3, kernel_size=3, padding=1, bias=False)
        self.norm_out = nn.GroupNorm(num_groups=chan, num_channels=chan)

        # nn.init.normal_(self.attn_combined.weight, mean=0, std=0.02)
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch, chan, height, width = inputs.shape

        out = self.norm_in(inputs)

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

        out = inputs + out
        out = self.norm_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, *,
                 clip_emblen: int, 
                 chan: int, size: int, 
                 nheads: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.nheads = nheads
        self.clip_emblen = clip_emblen

        #    (clip_emblen, )
        # -> (chan, size, size)
        self.clip2flat = nn.Linear(clip_emblen, chan * size * size)
        self.clip_unflatten = nn.Unflatten(1, (chan, size, size))
        self.clip_norm = nn.GroupNorm(num_groups=chan, num_channels=chan)

        self.attn_kv = nn.Conv2d(chan, chan * 2, kernel_size=3, padding=1, bias=False)
        self.attn_q = nn.Conv2d(chan, chan, kernel_size=3, padding=1, bias=False)
        self.attn_norm = nn.GroupNorm(num_groups=chan, num_channels=chan)
    
    def forward(self, inputs: Tensor, clip_embed: Tensor, clip_scale: Tensor) -> Tensor:
        batch, _chan, height, _width = inputs.shape
        if clip_embed is None:
            clip_embed = torch.zeros((batch, self.clip_emblen), device=inputs.device)
        if clip_scale is None:
            clip_scale = torch.zeros((batch, 1, 1, 1))
        
        # calculate key, value on the clip embedding
        clip_flat = self.clip2flat(clip_embed)
        clip_out = self.clip_unflatten(clip_flat)
        clip_out = self.clip_norm(clip_out)

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

        out = inputs + out
        out = self.attn_norm(out)
        out = out * clip_scale

        return out

class CrossAttentionConv(nn.Module):
    def __init__(self, *,
                 clip_emblen: int, 
                 chan: int, size: int, 
                 nheads: int):
        super().__init__()

        self.scale = nheads ** -0.5
        self.nheads = nheads
        self.clip_emblen = clip_emblen

        #    (clip_emblen, )     (1024,)
        # -> (chan, size, size)  (4, 16, 16)
        #
        #    (1024,  1,  1)
        # -> ( 256,  2,  2)
        # -> (  64,  4,  4)
        # -> (  16,  8,  8)
        # -> (   4, 16, 16)

        self.clip2conv = nn.Sequential()
        if True:
            self.clip2conv.append(nn.Unflatten(1, (clip_emblen // 16, 4, 4)))
            in_chan = clip_emblen // 16
            in_size = 4
        else:
            in_chan = clip_emblen
            in_size = 1
        print()
        print(f"want size: {chan}, {size}")
        while in_chan > chan or in_size < size:
            out_size = min(in_size * 2, size)
            out_chan = max(in_chan // 4, chan)

            padding = 1
            output_padding = 1
            # if in_size < 4:
            #     padding = 0
            #     out_chan = in_chan

            print(f"chan {in_chan} -> {out_chan}")
            print(f"  size {in_size} -> {out_size}")
            print(f"  padding = {padding}")
            print(f"  output_padding = {output_padding}")
            conv = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=2, padding=padding, output_padding=output_padding)
            norm = nn.GroupNorm(out_chan, out_chan)
            nl = nn.SiLU(True)
            self.clip2conv.append(nn.Sequential(conv, norm, nl))

            in_chan = out_chan
            in_size = out_size

        # now we're at (chan, size, size)
        self.clip_norm = nn.GroupNorm(num_groups=chan, num_channels=chan)

        self.attn_kv = nn.Conv2d(chan, chan * 2, kernel_size=3, padding=1, bias=False)
        self.attn_q = nn.Conv2d(chan, chan, kernel_size=3, padding=1, bias=False)
        self.attn_norm = nn.GroupNorm(num_groups=chan, num_channels=chan)
    
    def forward(self, inputs: Tensor, clip_embed: Tensor = None, clip_scale: Tensor = None) -> Tensor:
        batch, _chan, height, _width = inputs.shape
        if clip_embed is None:
            clip_embed = torch.zeros((batch, self.clip_emblen), device=inputs.device)
        if clip_scale is None:
            clip_scale = torch.zeros((batch, 1, 1, 1))
        
        # calculate key, value on the clip embedding
        # clip_embed = einops.rearrange(clip_embed, "b c -> b c 1 1")
        clip_in = clip_embed
        for mod in self.clip2conv:
            clip_out = mod(clip_in)
            # print(mod)
            # print(f"{clip_in.shape} -> {clip_out.shape}")
            # print()
            clip_in = clip_out

        # clip_out = self.clip2conv(clip_embed)
        # print(f"{clip_out.shape=}")
        clip_out = self.clip_norm(clip_out)

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

        out = inputs + out
        out = self.attn_norm(out)
        out = out * clip_scale

        return out
