from typing import Any, Dict, Optional, Tuple, Union
from diffusers import UNet2DConditionModel
import torch
from torch import FloatTensor, Tensor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from nnexp.base_model import BaseModel
class UnetDiffusers(UNet2DConditionModel, BaseModel):
    _metadata_fields = "sample_size in_channels out_channels layers_per_block norm_num_groups cross_attention_dim attention_head_dim".split()
    _model_fields = _metadata_fields

    def __init__(self, 
                 sample_size: int | None = None, 
                 in_channels: int = 4, out_channels: int = 4, 
                #  center_input_sample: bool = False, 
                #  flip_sin_to_cos: bool = True, 
                #  freq_shift: int = 0, 
                #  down_block_types: Tuple[str] = ..., 
                #  mid_block_type: str | None = "UNetMidBlock2DCrossAttn", 
                #  up_block_types: Tuple[str] = ..., 
                #  only_cross_attention: bool | Tuple[bool] = False, 
                #  block_out_channels: Tuple[int] = ..., 
                 layers_per_block: int = 2, 
                #  downsample_padding: int = 1, 
                #  mid_block_scale_factor: float = 1, 
                #  act_fn: str = "silu", 
                 norm_num_groups: int | None = 32,
                #  norm_eps: float = 0.00001, 
                #  cross_attention_dim: int = 1280, 
                 cross_attention_dim: int = 768, 
                 attention_head_dim: int | Tuple[int] = 8, 
                #  dual_cross_attention: bool = False, 
                #  use_linear_projection: bool = False, 
                #  class_embed_type: str | None = None, 
                #  num_class_embeds: int | None = None, 
                #  upcast_attention: bool = False, 
                #  resnet_time_scale_shift: str = "default", 
                #  time_embedding_type: str = "positional", 
                #  timestep_post_act: str | None = None, 
                #  time_cond_proj_dim: int | None = None, 
                #  conv_in_kernel: int = 3, 
                #  conv_out_kernel: int = 3, 
                #  projection_class_embeddings_input_dim: int | None = None
                 ):
        # super().__init__(sample_size, in_channels, out_channels, center_input_sample, flip_sin_to_cos, freq_shift, down_block_types, mid_block_type, up_block_types, only_cross_attention, block_out_channels, layers_per_block, downsample_padding, mid_block_scale_factor, act_fn, norm_num_groups, norm_eps, cross_attention_dim, attention_head_dim, dual_cross_attention, use_linear_projection, class_embed_type, num_class_embeds, upcast_attention, resnet_time_scale_shift, time_embedding_type, timestep_post_act, time_cond_proj_dim, conv_in_kernel, conv_out_kernel, projection_class_embeddings_input_dim)
        super().__init__(sample_size=sample_size, in_channels=in_channels, out_channels=out_channels, 
                         layers_per_block=layers_per_block, norm_num_groups=norm_num_groups,
                         cross_attention_dim=cross_attention_dim,
                         attention_head_dim=attention_head_dim,
                         flip_sin_to_cos=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim

    def forward(self, 
                sample: FloatTensor, 
                timestep: Tensor | float | int, 
                encoder_hidden_states: Tensor, 
                # class_labels: Tensor | None = None, 
                # timestep_cond: Tensor | None = None, 
                # attention_mask: Tensor | None = None, 
                # cross_attention_kwargs: Dict[str, Any] | None = None, 
                # down_block_additional_residuals: Tuple[Tensor] | None = None, 
                # mid_block_additional_residual: Tensor | None = None, 
                # return_dict: bool = True
                ) -> UNet2DConditionOutput | Tuple:
        res = super().forward(sample, timestep, encoder_hidden_states)
        return res.sample
