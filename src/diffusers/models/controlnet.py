# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders.single_file_model import FromOriginalModelMixin
from ..utils import BaseOutput, logging
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from .unets.unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the middle block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 3,
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        # timesteps = timestep
        # if not torch.is_tensor(timesteps):
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     # This would be a good case for the `match` statement (Python 3.10+)
        #     is_mps = sample.device.type == "mps"
        #     if isinstance(timestep, float):
        #         dtype = torch.float32 if is_mps else torch.float64
        #     else:
        #         dtype = torch.int32 if is_mps else torch.int64
        #     timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        # elif len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)

        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0])

        # t_emb = self.time_proj(timesteps)

        # # timesteps does not contain any weights and will always return f32 tensors
        # # but time_embedding might actually be running in fp16. so we need to cast here.
        # # there might be better ways to encapsulate this.
        # t_emb = t_emb.to(dtype=sample.dtype)

        # emb = self.time_embedding(t_emb, timestep_cond)
        emb = torch.tensor([[
            0.001573760062456131, 0.003983119502663612, 0.004462781362235546, -0.004281631205230951, -0.00042304862290620804, -0.005237431265413761, 0.004387935623526573, -0.003057350404560566, 0.0015143556520342827, -0.0031256263609975576, 0.0018839500844478607, -0.000556538812816143, 0.003976128064095974, 0.002513561863452196, 0.0003270185552537441, -0.001813570037484169, 0.00035851821303367615, 0.0009468318894505501, 0.0017209844663739204, -0.0004064321983605623, 0.0014515905641019344, 0.000569615513086319, 0.0006995950825512409, 0.003191739320755005, 0.0002742065116763115, 0.0003253370523452759, 0.0033123816829174757, -0.004212662111967802, 0.00218002125620842, 0.001134731573984027, 0.003149289172142744, -0.0022175905760377645, -0.003896990790963173, 0.0011022526305168867, -0.0003189835697412491, -0.002463487908244133, -0.00013224361464381218, 0.0015131481923162937, 0.003953500185161829, -0.0034983335062861443, -0.0010416144505143166, 0.002219835063442588, -0.002253282815217972, 0.0005164218600839376, -0.0020651649683713913, 0.002995274029672146, 0.00038562342524528503, -0.0014354735612869263, -0.0037513375282287598, 0.00390644371509552, -0.006464649457484484, 0.0029555605724453926, -1.281495451927185, -0.0019990727305412292, -0.0024484973400831223, -0.0009931828826665878, -0.0043351175263524055, -0.0023281155154109, -0.005434863269329071, 0.0019127479754388332, 0.0030843205749988556, -0.0037640701048076153, 0.00010932702571153641, -0.0023974012583494186, -0.0009823599830269814, 0.0022071311250329018, 0.0025359923020005226, 0.0024146116338670254, -0.004546660929918289, -0.005741985980421305, -0.001721484586596489, -0.005092525854706764, 0.0036882604472339153, 0.000507475808262825, 0.001775321550667286, 0.0025191577151417732, -0.0040859924629330635, -0.0006659394130110741, 0.0049048736691474915, -0.0018546050414443016, -0.0013929088599979877, -0.006917554885149002, -0.0037578195333480835, -0.003701898269355297, 0.0029908567667007446, -0.00382615951821208, -0.0015301201492547989, -0.0037727830931544304, -0.0014529828913509846, 0.007609358057379723, 0.0006422535516321659, 0.001996539533138275, -0.005521866958588362, 0.0012976010330021381, -0.006788223050534725, -0.005281364545226097, 0.0032197360415011644, -0.004545530304312706, 0.0003637606278061867, 0.0004333518445491791, 0.001150570111349225, 0.0025102877989411354, -0.002707595005631447, 1.4468027353286743, -0.002623782493174076, -0.0028594182804226875, 0.003262449987232685, 0.0005635758861899376, -0.0007676440291106701, 0.002406049519777298, 0.0021881158463656902, -0.0032860483042895794, -0.010794845409691334, 0.000648031011223793, -0.0016486607491970062, 0.000897659920156002, 0.001105004921555519, 0.0029931701719760895, -0.0008718359749764204, 0.002969648689031601, -0.000229591503739357, 0.00010334234684705734, -1.2840321063995361, 0.00481336610391736, -0.005029212683439255, 4.883343353867531e-05, -0.00037229270674288273, -0.003632918931543827, 0.002740374766290188, -0.0012193522416055202, -0.002987159416079521, 8.001085370779037e-05, 0.0023774965666234493, 0.0028910255059599876, 0.00761066609993577, 0.0025414153933525085, 0.0003518667072057724, -0.004216345027089119, -0.00515271769836545, 0.0013072595465928316, 0.0014619100838899612, 0.0002507837489247322, 0.00013870839029550552, -0.00028861500322818756, -0.0013658218085765839, -0.0066671716049313545, -0.0021179690957069397, 0.004830261692404747, 0.003088529920205474, 0.00043837446719408035, -0.0016181059181690216, -0.006607449147850275, -0.0044359127059578896, 6.60410150885582e-05, 0.0018336246721446514, 0.0013029873371124268, -0.010800446383655071, -0.0005278047174215317, 0.0012791140470653772, 0.0005238112062215805, 0.0006494901608675718, 0.003117853309959173, 0.001921843271702528, -0.0015421854332089424, 0.0013363100588321686, -0.0012281746603548527, -0.0012898538261651993, 0.0018622875213623047, -2.7198433876037598, -0.0006378879770636559, -0.0015364945866167545, -0.0036413331981748343, -0.0013077519834041595, -0.0010831891559064388, 0.0008701649494469166, -0.0010595154017210007, 0.0003477828577160835, 0.006167642772197723, 0.0038183145225048065, 0.000238688662648201, -0.0013582934625446796, 0.0009302585385739803, 0.003870030865073204, 0.0031802989542484283, -0.0014002895914018154, 0.0013122018426656723, 0.007359057664871216, 0.0046741534024477005, -0.0010233279317617416, -0.0016618645749986172, -0.0005740784108638763, -0.0009059796575456858, 0.001242651604115963, 0.002593628130853176, -0.0027749640867114067, 0.003243710845708847, -0.00612625852227211, -0.002698092255741358, 0.0032636309042572975, 0.0020998474210500717, 0.004150474444031715, 0.0013025538064539433, 0.004807082936167717, 0.0018755621276795864, 0.0016154302284121513, 0.0034434515982866287, -0.0014972030185163021, -0.0017382721416652203, -0.000635957345366478, -0.002074766904115677, 0.0047565000131726265, 0.002718043513596058, 0.0023012771271169186, 0.0022188620641827583, 3.763008862733841e-05, 0.0013531665317714214, -0.0015252362936735153, 0.0011127330362796783, -0.001284589059650898, 3.6115990951657295e-05, 0.0031943265348672867, 0.0021157199516892433, -0.0021895524114370346, 0.042180247604846954, -0.0006518866866827011, 0.005039523355662823, -0.003038078546524048, -0.0012780565302819014, -0.0011488832533359528, -0.002280827611684799, -0.0020638033747673035, -0.000586499460041523, -0.0007883496582508087, 0.0025799237191677094, 0.00010050507262349129, -0.0018582134507596493, 0.0009967442601919174, 0.0005951030179858208, -0.0014335475862026215, -0.0006995443254709244, 0.0017255963757634163, -0.0007584551349282265, 0.003378524910658598, -0.003923366777598858, -1.6446399688720703, 0.0049955397844314575, 0.0017437906935811043, 0.0023354394361376762, 0.0017891726456582546, 0.0027156881988048553, 0.003423777874559164, -0.0022419001907110214, -0.003434761893004179, -0.005017617717385292, -0.0035867542028427124, -0.004241776652634144, 0.00840651523321867, -0.006988520734012127, -0.0018430114723742008, 0.00380491535179317, 0.0016685128211975098, -0.005709989462047815, 0.008595876395702362, -0.2618739604949951, 0.0018722694367170334, 0.002297826111316681, 0.0022490881383419037, -0.005505930632352829, 0.002256993902847171, 0.0028387601487338543, -0.0028464864008128643, -0.002752947388216853, -0.0015496471896767616, -0.011647528037428856, -0.001796056516468525, -0.002321428619325161, -0.007836826145648956, 0.0006679098587483168, -0.0020936597138643265, -0.0006143040955066681, 0.003840770572423935, -0.0004355916753411293, 0.0009083971381187439, 0.0018894951790571213, -0.0001702951267361641, 0.0007244637235999107, -0.005966695956885815, -0.0031678881496191025, 0.004658031277358532, 0.0032961098477244377, -0.00017529074102640152, 0.0037146974354982376, 0.005065157078206539, -0.00032286718487739563, -0.00010201637633144855, 0.000842507928609848, 0.0015019646380096674, 0.0016351307276636362, 0.0023873941972851753, -0.0022899205796420574, -0.005495131015777588, -0.0016921209171414375, 0.0013966262340545654, -8.303672075271606e-06, -0.004246651194989681, 0.0011001182720065117, -0.0016768891364336014, 0.00520341144874692, 0.002576081082224846, 0.008013416081666946, -0.002091445028781891, -0.01020589005202055, -0.0020319311879575253, -0.0045837764628231525, -0.004769474267959595, -0.0012807957828044891, 0.0011649522930383682, -0.005143806338310242, -0.002602573484182358, 0.0008569657802581787, 0.0016478411853313446, -0.004085839726030827, -0.008924288675189018, -0.004562027752399445, 0.00041833240538835526, 0.001741466112434864, 0.0014283908531069756, 0.0001915055327117443, 0.9168856739997864, -0.0012135356664657593, 0.0003626698162406683, -0.003538453485816717, 0.00266058836132288, -0.00042150402441620827, 0.004242387600243092, -0.0019568903371691704, -0.001958368346095085, 0.0001576144713908434, -0.0009095054119825363, -0.007110824808478355, 5.762279033660889e-05, 0.0036075301468372345, -0.00335543230175972, 2.3155555725097656, -5.282089114189148e-05, -0.0024347808212041855, -0.0020467087160795927, -0.0006498799193650484, -0.00046903640031814575, -0.0013231550110504031, 0.0014400426298379898, 0.00443670991808176, -0.0040122270584106445, 0.0032114628702402115, -0.004101053811609745, -0.0004988731816411018, -0.0014346546959131956, 0.0005926343146711588, -0.0009169634431600571, 0.006959364749491215, -0.0010797586292028427, -0.00328157190233469, -0.007339881733059883, -0.00032687000930309296, 0.0005140905268490314, 4.0269456803798676e-05, -0.003431119956076145, 0.0009855683892965317, 0.00044783297926187515, -1.4068043231964111, 0.0034143999218940735, -0.0009257793426513672, 0.0021790489554405212, 0.001458561047911644, 0.0022926144301891327, 0.0025319308042526245, -0.004655626602470875, 0.0008477112278342247, -0.006989926099777222, 0.0008376957848668098, 0.0027560684829950333, -1.8949291706085205, 0.000812336802482605, -0.0005626268684864044, 0.00476538110524416, 0.00441830325871706, 0.0037030933890491724, -0.0066874660551548, -0.008244704455137253, -0.0029975692741572857, -0.0009117024019360542, -0.007958809845149517, -0.0015100864693522453, -0.0042806752026081085, 0.008465497754514217, -0.0004926268011331558, 0.005048427730798721, 0.004449198488146067, -0.001012623542919755, -0.0003742668777704239, -0.006078293547034264, 0.00029392330907285213, 0.004031847231090069, -0.002813871018588543, -0.005575100425630808, 0.0008490118198096752, -0.0014988118782639503, 0.0020347172394394875, -0.0009627798572182655, 0.0008232803083956242, 0.0015832853969186544, 0.0048425886780023575, 0.0028991950675845146, -0.0008421251550316811, 0.001946386881172657, 0.002370106056332588, 0.0008501317352056503, 0.0020889085717499256, 0.0018355436623096466, -0.006476172246038914, 0.004663452506065369, 0.0008262074552476406, -0.0004830346442759037, -0.0020447690039873123, 0.004131851252168417, -0.0017300918698310852, -0.002857876941561699, -0.00021382980048656464, -0.002656737808138132, 0.0019742073491215706, 0.004243399947881699, 0.007720837369561195, 0.0052322885021567345, -0.0007544476538896561, -0.0006418162956833839, 0.0077160559594631195, -0.002170204184949398, -0.0008374880999326706, -0.0027047693729400635, 0.00011131074279546738, 0.009622989222407341, -0.0007555121555924416, -0.0025221500545740128, 0.0030959919095039368, -0.0007617445662617683, -0.000594334676861763, 0.0010652127675712109, 0.0011058535892516375, -0.00249468837864697, -0.0025306008756160736, 0.003857620991766453, -0.004143408499658108, -3.8419500924646854e-05, -0.005077281966805458, 0.0038945600390434265, -0.005357180256396532, 0.006497934460639954, -0.0019328389316797256, 0.0030805841088294983, -0.001107092946767807, 0.0015760880196467042, -0.0006814773660153151, -0.0005543790757656097, 0.0012729638256132603, -0.0028605302795767784, -0.006566728465259075, 0.0034328652545809746, 0.00038708001375198364, 0.002706779632717371, -0.00019780918955802917, -0.002322274260222912, 0.002062353305518627, -0.0031118281185626984, -0.000547550618648529, 0.0026848423294723034, -0.12094961851835251, -0.005694536957889795, -0.0011263079941272736, -0.00484526576474309, -1.4860886335372925, -6.948760710656643e-05, 0.001058191992342472, -0.0010365797206759453, 0.0002737146569415927, -0.005960485897958279, 0.003510179929435253, -0.001762579195201397, 0.0015929695218801498, -0.00020633172243833542, 0.003112228587269783, -0.00399877829477191, 1.639386773109436, -0.0022320072166621685, -0.00274641253054142, -0.0005401931703090668, -1.618101716041565, 0.001832813024520874, 0.0022378722205758095, -0.000643397681415081, 0.0008976217359304428, -0.0010443730279803276, 0.00018486054614186287, -0.00292200967669487, 0.0014348975382745266, 0.003784148022532463, 0.001394936814904213, -0.0011877231299877167, 0.002443718258291483, -0.000743890181183815, 0.00033687055110931396, -0.003418516833335161, 0.0013122353702783585, -1.0298254489898682, 0.0034949197433888912, 0.002301619853824377, -0.0001343321055173874, -0.002009935677051544, 0.0010001230984926224, -0.0005379975773394108, 0.004416018724441528, 0.0022373273968696594, 0.0036168843507766724, 0.0030299555510282516, -0.0013448079116642475, 0.00023363996297121048, -0.002745106816291809, 0.00046072318218648434, 0.0047026267275214195, -0.0023163114674389362, -0.0013941070064902306, -0.0019258414395153522, -0.001491079106926918, -0.002410940360277891, 0.0005715019069612026, -0.0004010056145489216, 0.005819377489387989, -0.0035830754786729813, 0.004737455397844315, 0.0042218612506985664, -0.00043945806100964546, -0.004120239522308111, 0.0012333812192082405, -0.0017643570899963379, 0.003526192158460617, 0.0002467986196279526, 0.006448457017540932, 0.0019627041183412075, 0.0006156992167234421, -0.0013647349551320076, -0.01057916134595871, -1.4918785095214844, -0.004846381023526192, -0.004311966709792614, 0.002499723806977272, 0.005356467794626951, 0.0011808648705482483, 0.0007641881238669157, -0.0015456285327672958, -0.0031174239702522755, 0.0023615583777427673, 0.0008501042611896992, -0.0014118952676653862, 0.001293649896979332, 0.001853924011811614, 0.0005669360980391502, -0.002797246677801013, -0.002297863131389022, -0.0012656152248382568, -0.0015238821506500244, -0.0012172125279903412, 0.002198481000959873, 0.002278277650475502, 0.0001421072520315647, -0.0032623824663460255, 0.0015451209619641304, -0.0015714818146079779, -0.002147005870938301, 0.0011647767387330532, -0.0005254987627267838, 0.001188592636026442, 0.004905504174530506, 0.005966876167804003, -0.0017544764559715986, 0.0012377537786960602, 0.00018718442879617214, -0.0018962661270052195, 0.001437733881175518, 0.0024882759898900986, -0.002993253991007805, 5.107407923787832e-05, -0.0022559617646038532, -0.0056568048894405365, -0.0004252055659890175, -0.001653539016842842, 0.004440532065927982, -0.0004573836922645569, -0.0005576424300670624, -0.005777172744274139, 0.0007882025092840195, -0.00042327120900154114, 0.006523096933960915, -0.0014824620448052883, 0.0012357626110315323, 0.0037946216762065887, 0.000406632199883461, 0.002366031752899289, 0.00021815451327711344, 0.002675158903002739, -0.0007092254236340523, -0.001674967585131526, 0.00225159153342247, -0.001893565058708191, -0.002088002860546112, 0.0015942249447107315, -0.003867102786898613, 0.0003978699678555131, 0.0012857289984822273, 0.0033877603709697723, 0.0010957969352602959, -0.0011560418643057346, 0.0033991942182183266, 0.0007143316324800253, -0.002073876094073057, -0.00019285408779978752, -0.00017412099987268448, 0.0004471724387258291, -0.0040366631001234055, 0.007114000618457794, -0.0015232162550091743, 0.001439741812646389, -0.0037404485046863556, 7.21001997590065e-05, 7.755914703011513e-05, 0.0004289308562874794, -0.001696727704256773, -0.0030869650654494762, 0.0018046386539936066, -0.0017079971730709076, -0.002986464649438858, 0.00031231343746185303, -0.003855002112686634, 0.005064483731985092, 0.0006083352491259575, -0.001369025558233261, 0.002161058597266674, 0.0009699780493974686, -0.0024465941824018955, 0.001185685396194458, -0.006233932450413704, -0.0007888525724411011, 0.0038322671316564083, 0.0024762842804193497, -0.004961414262652397, -0.0035335978027433157, -0.0016098041087388992, 0.004066511057317257, 0.0011343320365995169, 0.0021734992042183876, 0.0021492312662303448, 0.007374323904514313, 0.0011963974684476852, 0.0011460981331765652, -0.0013086432591080666, -0.0037141693755984306, 0.0008812267333269119, -0.0027214745059609413, 0.00036020856350660324, -0.001902431482449174, -0.0035599181428551674, 0.0010403851047158241, -0.0016584070399403572, -0.00463405717164278, -0.0044906046241521835, -0.004982098005712032, -0.0027714427560567856, -0.0008735181763768196, 0.008558740839362144, -0.0035961042158305645, -0.0013225898146629333, 0.005308445543050766, -0.0042589688673615456, -0.003493448020890355, 0.0021755159832537174, -0.00734657421708107, -0.005343990866094828, -0.006614945828914642, -0.0001998113002628088, 0.004023966379463673, 0.0008875573985278606, 0.00238843634724617, 0.005394498817622662, 0.0021166340447962284, 0.0015096813440322876, 0.006221773102879524, 0.0003690016455948353, 0.006201388780027628, 0.0007003238424658775, 0.0033250320702791214, -0.003188507631421089, 0.0013225693255662918, 0.0008990960195660591, 0.0045664929784834385, 0.001383165828883648, 0.0031069982796907425, 0.0018772035837173462, 0.0026900488883256912, 0.00325818732380867, -0.0008720774203538895, -0.001051061786711216, -0.0008806396508589387, -0.0007171910256147385, 0.006759069859981537, 8.042342960834503e-05, -0.0024415114894509315, 0.00041000964120030403, -0.0023328433744609356, -0.001885436475276947, 0.0004613678902387619, -0.005041848868131638, 0.003930213861167431, 0.004293358884751797, 0.0012252144515514374, -0.003117151325568557, -0.001955214887857437, 0.006754918489605188, 0.004392855800688267, -0.0048887780867516994, 0.005726806819438934, 0.004945304244756699, -0.0005626031197607517, -0.0013753632083535194, -0.0071588121354579926, -0.0008491247426718473, -0.0038863522931933403, -0.002258771099150181, 3.0239217281341553, -0.00251003447920084, -0.00408544484525919, -0.0022957748733460903, -0.006423422135412693, 0.0012284654658287764, 0.001780893886461854, 6.966758519411087e-05, -0.0036941785365343094, -0.0063496907241642475, -0.004460230469703674, 0.003197777085006237, 0.0030860770493745804, 7.306528277695179e-05, 0.0007587573491036892, 0.0002026837319135666, -0.0004748925566673279, 0.0035630250349640846, -0.0020335260778665543, 0.007223442196846008, 0.0010587787255644798, 0.0024817087687551975, 0.0035464689135551453, 0.003677068278193474, -0.7253103256225586, -0.0006245188415050507, 3.302842378616333e-05, 0.0010428675450384617, 0.0005070003680884838, 0.0018898025155067444, -0.000709263258613646, 0.0015119186136871576, -0.0036627098452299833, 0.0016123878303915262, -0.0002939412370324135, 0.0031020785681903362, -0.00074005126953125, -0.004039250314235687, -0.00024979666341096163, -0.0036016860976815224, -0.0020781958010047674, 0.0009094085544347763, 0.0029314355924725533, -0.0029435809701681137, -0.005923964083194733, 0.003304737154394388, -0.0018496755510568619, 0.0030334945768117905, 8.25691968202591e-05, 0.0004921448417007923, -0.0009002527222037315, -1.633897304534912, -0.00015310384333133698, 0.0037671588361263275, 0.0016135051846504211, -0.00011366792023181915, -0.001864435849711299, 0.005045438185334206, -0.0003604530356824398, 0.002680456265807152, -0.0015722918324172497, 0.0023080362007021904, -0.0024516191333532333, -1.7451725006103516, 0.0016792765818536282, 0.0020986171439290047, 0.0008678464218974113, 0.0031220444943755865, 0.003935010172426701, -0.0012288354337215424, 0.0013154372572898865, -0.0020919316448271275, 0.0029172198846936226, -0.0007597170770168304, -0.005329126492142677, -7.3188915848732e-05, -0.0011559110134840012, 0.0006760729011148214, 0.0007421951740980148, -0.005462390370666981, 0.004615028388798237, -0.0017914376221597195, -0.003460240550339222, -0.0008385586552321911, 0.0021267198026180267, 0.0011241231113672256, 0.0029126256704330444, 0.0030372850596904755, 0.0029235035181045532, 0.32123592495918274, -0.00628615589812398, -0.0003090398386120796, -0.001110333250835538, -0.004590494558215141, 0.004380230791866779, 0.0007834043353796005, -0.005684261675924063, 0.0017709156963974237, 0.002021440304815769, -0.001956153428182006, 0.0006242482922971249, -0.004898661747574806, 0.0020476942881941795, -0.0002321125939488411, 0.000334068201482296, 0.0008440464735031128, 0.004206049256026745, 0.009479492902755737, 0.0015855520032346249, 0.0017925789579749107, -0.004714589100331068, -0.002132670022547245, 0.0050640711560845375, 0.0027477885596454144, -0.002532149665057659, -0.0025305040180683136, -0.0029056305065751076, -0.0036113662645220757, 0.0014385688118636608, -0.005048342049121857, 0.0010290630161762238, -0.0027932426892220974, 0.00601715175434947, 0.0002436924260109663, 0.0024283691309392452, -0.0009125033393502235, -0.0030069705098867416, -0.0016991370357573032, 0.0022056270390748978, -0.001784052699804306, 0.0022624218836426735, -0.002716373186558485, -0.41468432545661926, -0.0025408780202269554, 0.0036822576075792313, 0.004576297476887703, -0.003090783953666687, -0.0024545546621084213, 0.004092009272426367, 0.0028380611911416054, -0.0008670417591929436, 0.0004910863935947418, 0.0029047681018710136, -0.005248703062534332, 0.00398904737085104, -0.00012462027370929718, -0.0014782510697841644, 0.00554899126291275, 0.0003595435991883278, -0.00041473470628261566, -0.0006935740821063519, -0.0008803494274616241, -0.000593909528106451, 0.00263320654630661, -2.32793390750885e-05, 0.0039300499483942986, -0.001353536732494831, 0.001343493815511465, -0.004830924794077873, -0.0008937530219554901, -0.0026782117784023285, -0.0005197133868932724, 0.002332039177417755, -0.000991358421742916, -0.0007039839401841164, -0.0013416027650237083, -0.0006264676339924335, -0.0005540670827031136, 0.0010761301964521408, -0.0008522779680788517, 0.0008130935020744801, -0.0020078150555491447, -0.004241798538714647, 0.005864625796675682, 0.0011827284470200539, -0.003298233263194561, -0.0007616886869072914, 0.0006372854113578796, -0.004849070683121681, 0.0006641910877078772, 0.0035660010762512684, 0.00033114291727542877, -0.0005867527797818184, 0.0018633943982422352, -0.00036505982279777527, 0.002763839904218912, -0.004819529131054878, 0.008333471603691578, 0.0003508441150188446, 0.002395779360085726, 0.0021870583295822144, 0.00253940187394619, -0.003009064821526408, -0.0037221135571599007, 0.001971549354493618, 0.0012405402958393097, -0.00015866197645664215, -0.002934104762971401, -0.0015861503779888153, 1.5812348127365112, 0.004401747137308121, 0.0009417757391929626, -0.004960566759109497, -0.0012387835886329412, -0.000548649113625288, 0.004420945420861244, 0.00269148638471961, 0.004815464839339256, -0.005988169927150011, 2.16728076338768e-05, 0.00043149059638381004, -0.003960667178034782, 0.001562969759106636, -0.005089595913887024, -0.004418808966875076, -0.0010860838228836656, 0.0018424857407808304, -0.0008966922760009766, -0.00028818147256970406, -0.00032126111909747124, 0.0005008559674024582, -0.003966665826737881, -0.0023323222994804382, 0.00013964297249913216, -0.0025547947734594345, 0.00120452418923378, 0.005071369931101799, -0.00044418941251933575, 0.002048116410151124, 0.0029920944944024086, -0.0011471696197986603, -0.0023097973316907883, -0.003475458826869726, 0.00015943869948387146, -0.0022622868418693542, -0.00190043356269598, 0.00014438806101679802, 0.0013883155770599842, -0.0035430402494966984, -0.0008174190297722816, 2.3085391148924828e-05, 0.0011450834572315216, -0.006895215716212988, -0.0005761473439633846, 0.003001634031534195, 0.0054661184549331665, -0.0068648336455225945, 0.004911451600492001, 3.786478191614151e-05, 0.0019904840737581253, 0.004156948067247868, -0.0033011040650308132, 3.8747675716876984e-05, -0.0018689371645450592, 0.005017763003706932, 2.8114280700683594, 0.00078623928129673, 0.00016822898760437965, -0.0005959784612059593, 0.002179596573114395, 0.00018699048087000847, -0.004196321591734886, 0.004341773688793182, 0.007171729113906622, -0.0016133328899741173, -0.0017857067286968231, -0.0027435850352048874, 0.0019112159498035908, 0.0006300467066466808, 0.0047654034569859505, -0.005713992286473513, -0.00047552958130836487, -0.0003591049462556839, -0.0025159139186143875, -0.00016899383626878262, -0.0063021620735526085, 0.002350982278585434, 0.0018456317484378815, 0.006085352972149849, -0.00032450631260871887, -0.003962242975831032, -0.0012691696174442768, -0.0008511766791343689, -0.00023767584934830666, -0.002027584472671151, 0.004480898380279541, 0.002916550263762474, 0.0005348473787307739, 0.005142136011272669, 0.002954286988824606, 0.00033515505492687225, -0.0019863881170749664, 0.00041757524013519287, -0.0075147696770727634, -0.0003546690568327904, 0.009084317833185196, 0.003674534149467945, -0.0007253922522068024, 0.0002051754272542894, -0.002833779901266098, 0.0022399863228201866, 0.0006759236566722393, -0.00046778004616498947, -0.0037448452785611153, -0.0063897850923240185, 0.0011677350848913193, 0.0012337015941739082, 0.004640596453100443, -1.4281222820281982, 0.0014177057892084122, 0.19433896243572235, -0.0006621431093662977, -0.0008017572108656168, 0.0004889518022537231, -2.10341215133667, 0.0025174422189593315, 0.0004610437899827957, 0.004700290504842997, -0.001080143265426159, 0.0058074332773685455, 0.0006607584655284882, -0.00024177972227334976, -0.0016557546332478523, -1.2840347290039062, -0.002236573025584221, -0.0021272972226142883, 0.001634489744901657, 0.0022295068483799696, 0.0011627376079559326, -0.005332889501005411, -0.0033645490184426308, -0.0036730491556227207, 0.0016279635019600391, -0.0032469360157847404, 0.002352361334487796, 0.0009300047531723976, -0.000680902972817421, -0.0031165790278464556, -0.0019067542161792517, -0.0032868199050426483, -0.006642715074121952, 0.002466166391968727, 0.0014772703871130943, -0.003884074278175831, 0.00025007128715515137, -0.0036653350107371807, -0.004342014901340008, -0.002221371978521347, -0.0016593467444181442, -0.0009377803653478622, 0.0006422828882932663, 0.001671796664595604, -0.005785600747913122, -0.0006269048899412155, 0.0015235599130392075, 0.010439109057188034, -0.0015283161774277687, -0.0014796582981944084, 0.0015669837594032288, -0.000427482184022665, -0.0017491262406110764, -0.000563659006729722, -0.0032539372332394123, 0.0009276107884943485, -0.004631875082850456, -0.0013774840626865625, -0.0021644816733896732, -0.0008814241737127304, 0.0008815629407763481, 0.001688716933131218, 0.0010842280462384224, 0.001281999982893467, -0.003832554444670677, 0.0019688010215759277, 0.0003179148770868778, 0.00015615951269865036, 0.0031380001455545425, -0.0003474820405244827, 0.001381240668706596, 0.0010844040662050247, 0.003846457228064537, -0.006254042033106089, 0.003845561295747757, 0.00035628490149974823, -0.0031035300344228745, 0.0020300778560340405, -0.0018826201558113098, 0.0016375640407204628, 6.017833948135376e-05, 0.005620122421532869, 0.0007369155064225197, -0.004857498221099377, -0.00021863123401999474, 0.001321721589192748, -0.001956176944077015, 0.003959510009735823, -0.0038497699424624443, -0.0035636876709759235, 0.0028892895206809044, 0.006793946027755737, -0.0002987366169691086, -0.0020075938664376736, 0.0037815808318555355, 0.0043587395921349525, -0.005607419181615114, -0.004988698288798332, -0.00020177476108074188, 0.000867379829287529, 0.0007666358724236488, -0.006854435428977013, -0.004148400388658047, 0.0002255771541967988, 0.007598418276757002, -0.0005946159362792969, -0.007864024490118027, -0.002120738849043846, -0.00020080525428056717, -0.004447177052497864, 0.007061447016894817, -0.002036137506365776, 0.0011654924601316452, 0.0007649455219507217, -0.00012221187353134155, 0.005908178165555, -0.004090787842869759, 0.0020263725891709328, 0.0017749574035406113, 0.0018324078992009163, -0.0026239329017698765, -7.185712456703186e-05, 0.013181917369365692, 0.0018460694700479507, 0.0004098387435078621, 6.654812023043633e-05, 0.0006736405193805695, 0.001035214401781559, -0.0018508760258555412, -0.009155718609690666, 0.0011359232012182474, -0.0014096652157604694, -1.28099524974823, -0.001628592610359192, -0.0005769133567810059, -0.004236030392348766, -7.69816106185317e-05, -0.0007387716323137283, -0.0017938707023859024, 0.00043406616896390915, -0.0013707843609154224, 0.003578394651412964, 0.0063819680362939835, 0.00417726906016469, 0.0013478565961122513, 0.0022441167384386063, -0.0022446014918386936, 0.0015756767243146896, -0.001039890805259347, -0.009393153712153435, 0.0003429371863603592, -1.544225960969925e-05, 0.006139843259006739, 0.004910267889499664, -0.0016936594620347023, -0.0007370105013251305, -5.592871457338333e-05, 0.007193633355200291, -0.0045380727387964725, 0.009887877851724625, 0.0023380154743790627, 0.00027130916714668274, -0.0003590146079659462, 0.003195621073246002, -0.0018983976915478706, 0.002257922198623419, -0.002265665680170059, -0.00792757049202919, -0.002597050741314888, -0.0021384074352681637, 0.0012137978337705135, 0.0004513794556260109, -0.0045631565153598785, -0.001179361715912819, -0.0005247863009572029, -0.0010946174152195454, 0.00330214761197567, -0.002049025148153305, 0.0001325160264968872, 0.00034994748421013355, 0.0007294844835996628, 0.001200290396809578, 0.0043980577029287815, 0.0007217568345367908, 0.002625221386551857, 0.00038312096148729324, 0.0024133874103426933, 0.0018376028165221214, -1.4436577558517456, -0.00011195940896868706, 0.0006154407747089863, -0.000667286105453968, 0.004642015788704157, 0.0022163698449730873, -0.0020090239122509956, -0.003975590690970421, 0.0038032345473766327, -0.006407415494322777, -0.002446576487272978, 0.0010132926981896162, 0.007177361287176609, 0.004488252103328705, 0.003701772540807724, -0.006817928049713373, -0.00357226375490427, 0.0007550165755674243, -0.006563796196132898, -0.003407437354326248, -0.009274870157241821, -0.0003452617675065994, -0.0043473923578858376, 0.002632269635796547, 0.003757092170417309, -0.001883177668787539, -0.002972327172756195, 0.00177762471139431, -0.0014464594423770905, -0.0032688407227396965, -0.004790119826793671, -0.00018445774912834167, 0.00041343457996845245, -0.0022937178146094084, 0.0018343795090913773, -0.0017355559393763542, 0.0012868968769907951, 0.0021854154765605927, -0.0067925844341516495, -0.0003222622908651829, -0.0014073187485337257, 0.003934781067073345, 0.005610184744000435, 0.005196422804147005, -0.0009837128454819322, 0.0003563077189028263, 0.004645019769668579, -6.27022236585617e-05, 0.00475663598626852
        ]], dtype=torch.float32, requires_grad=False)
        # print("LOLOLOLOL CTRLNET")

        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
