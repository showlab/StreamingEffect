# SPDX-FileCopyrightText: Copyright (c) 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Any, Dict, List, Optional, Tuple, Union, Set
import types
import os
import torch
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.utils import deprecate, USE_PEFT_BACKEND, unscale_lora_layers
from transformers import CLIPTextModel, CLIPTokenizer
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


def classify_forward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
):
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
        return_features_early: If true, the forward pass returns the features once the set is complete.
            This means the forward pass will not finish completely and no final output is returned.
        return_logvar: If true, the foward pass returns the logvar.

    Returns:
        [A `list` is returned where the first element is the sample tensor.
    """

    # By default, samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb_timestep = self.time_embedding(t_emb, timestep_cond)

    emb = emb_timestep

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
    # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                    and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                    for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    down_block_res_samples = (sample,)
    idx, features = 0, []
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        if idx in feature_indices:
            features.append(sample)
        idx += 1

        down_block_res_samples += res_samples

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    if idx in feature_indices:
        features.append(sample)

    # If we have all the features, we can exit early
    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if len(feature_indices) == 0:
        # no features requested, return only the model output
        out = sample
    else:
        # score and features； score, features
        out = [sample, features]

    if return_logvar:
        logvar = self.unet.logvar_linear(emb_timestep)
        return out, logvar

    return out


class StableDiffusionTextEncoder:
    def __init__(
        self,
        model_id: str,
    ):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        # We never update the encoder, so freeze it
        self.text_encoder.eval().requires_grad_(False)

        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )

        # pre-compute the unconditioned prompt embeddings
        uncond_tokenized = self.tokenizer(
            [""], max_length=self.tokenizer.model_max_length, return_tensors="pt", padding="max_length", truncation=True
        )
        self.uncond_prompt_mask = uncond_tokenized.attention_mask
        self.uncond_prompt_embeddings = self.text_encoder(uncond_tokenized.input_ids)[0]

    def encode(self, conditioning: Optional[Any] = None, precision: dtype = torch.float32) -> Tuple[Any, Any]:
        # Get prompt embeddings
        if "" in conditioning:
            prompt_embeddings = torch.repeat_interleave(self.uncond_prompt_embeddings, len(conditioning), dim=0).to(
                self.text_encoder.device, dtype=precision
            )
            prompt_mask = torch.repeat_interleave(self.uncond_prompt_mask, len(conditioning), dim=0).to(
                self.text_encoder.device, dtype=precision
            )
        else:
            tokenized = self.tokenizer(
                conditioning,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenized.input_ids
            prompt_mask = tokenized.attention_mask.to(self.text_encoder.device, dtype=precision)
            prompt_embeddings = self.text_encoder(tokens.to(self.text_encoder.device))[0]

        return prompt_embeddings, prompt_mask

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.text_encoder.to(*args, **kwargs)
        return self


class StableDiffusionImageEncoder:
    def __init__(
        self,
        model_id: str,
    ):
        super().__init__()
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        # We never update the encoder, so freeze it
        self.vae.eval().requires_grad_(False)

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        latent_images = self.vae.encode(real_images, return_dict=False)[0].sample()

        return latent_images * self.vae.config.scaling_factor

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        latents = 1 / self.vae.config.scaling_factor * latent_images
        # print(f'vae.config.scaling_factor: {vae.config.scaling_factor}')
        images = self.vae.decode(latents, return_dict=False)[0].clip_(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.vae.to(*args, **kwargs)
        return self


class StableDiffusion15(FastGenNetwork):
    """A StableDiffusion teacher model for text-to-image diffusion distillation."""

    MODEL_ID_15 = "runwayml/stable-diffusion-v1-5"
    MODEL_ID_21 = "stabilityai/stable-diffusion-2-1"

    def __init__(
        self,
        model_id: str = MODEL_ID_15,
        net_pred_type="eps",
        schedule_type="sd",
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """StableDiffusion score model constructor.

        Args:
            model_id (str, optional): The huggingface model ID to load.
                Defaults to "runwayml/stable-diffusion-v1-5".
            net_pred_type (str, optional): Prediction type. Defaults to "eps".
            schedule_type (str, optional): Schedule type. Defaults to "sd".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
        """
        # Initialize FastGenNetwork with SD15-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id, load_pretrained)

        # Rewrite the unet forward to get bottleneck feature
        self.unet.forward = types.MethodType(classify_forward, self.unet)

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the UNet network.

        Args:
            model_id: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading SD15 UNet from pretrained")
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                cache_dir=os.environ["HF_HOME"],
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = UNet2DConditionModel.load_config(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="unet",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info("Initializing SD15 UNet on meta device (zero memory, will receive weights via FSDP sync)")
            else:
                logger.info("Initializing SD15 UNet from config (no pretrained weights)")
                logger.warning("SD15 UNet being initialized from config. No weights are loaded!")
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_config(config)

        self.unet.logvar_linear = torch.nn.Linear(1280, 1)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear and convolutional layers.
        """
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

        logger.debug("Reinitialized SD15 parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the SD15 network for FSDP.

        Note: We shard `self.unet` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.

        SD15 uses a U-Net architecture with:
        - down_blocks: Encoder blocks (CrossAttnDownBlock2D, DownBlock2D)
        - mid_block: Middle block (UNetMidBlock2DCrossAttn)
        - up_blocks: Decoder blocks (CrossAttnUpBlock2D, UpBlock2D)
        """
        unet_block_types = (
            CrossAttnDownBlock2D,
            CrossAttnUpBlock2D,
            DownBlock2D,
            UpBlock2D,
            UNetMidBlock2DCrossAttn,
        )

        # Note: Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if hasattr(self.unet, "gradient_checkpointing") and self.unet.gradient_checkpointing:
            self.unet.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(self.unet, check_fn=lambda block: isinstance(block, unet_block_types))
            logger.info("Applied FSDP activation checkpointing to SD15 U-Net blocks")

        # Shard down blocks
        for block in self.unet.down_blocks:
            fully_shard(block, **kwargs)

        # Shard mid block
        if hasattr(self.unet, "mid_block") and self.unet.mid_block is not None:
            fully_shard(self.unet.mid_block, **kwargs)

        # Shard up blocks
        for block in self.unet.up_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.unet, **kwargs)

    def init_preprocessors(self):
        """Initialize the text and image encoders for the Stable Diffusion model."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Stable Diffusion model."""
        self.text_encoder = StableDiffusionTextEncoder(model_id=self.model_id)

    def init_vae(self):
        """Initialize the image encoder for Stable Diffusion model."""
        self.vae = StableDiffusionImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def forward(
        self,
        x_t: torch.FloatTensor,
        t: torch.Tensor,
        condition: Optional[List[torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ):
        """Forward pass of the StableDiffusion latent diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (list[torch.Tensor]): Optional conditioning information.  Defaults to None.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_logvar: If true, the foward pass returns the logvar.
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.

        Returns:
            torch.Tensor: The score model output.
        """
        if r is not None:
            # TODO: add support for SD15
            raise NotImplementedError("r is not yet supported for SD15")
        if feature_indices is None:
            feature_indices = {}
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        self.unet.enable_gradient_checkpointing()

        # Note: Don't cast timestep to x_t.dtype here - the UNet's time_embedding expects
        # float32 input, and autocast will handle dtype conversions inside the UNet
        model_outputs = self.unet(
            x_t,
            self.noise_scheduler.rescale_t(t),
            encoder_hidden_states=condition[0],
            encoder_attention_mask=condition[1],
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[List[torch.Tensor]] = None,
        neg_condition: Optional[List[torch.Tensor]] = None,
        guidance_scale: Optional[float] = 7.5,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using DDIM-style sampling.

        Args:
            noise: Initial noise tensor [B, C, H, W] (should be scaled by max sigma).
            condition: List of [encoder_hidden_states, encoder_attention_mask].
            neg_condition: List of negative [encoder_hidden_states, encoder_attention_mask] for CFG.
            guidance_scale: CFG guidance scale. None disables guidance.
            num_steps: Number of sampling steps.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples in latent space.
        """
        # Get timestep schedule (SD uses t in [0, 1], mapped to [0, 1000])
        t_list = self.noise_scheduler.get_t_list(num_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=t_list[0])

        for t, t_next in zip(t_list[:-1], t_list[1:]):
            # Expand t for batch
            t_batch = t.expand(x.shape[0])

            # Get noise prediction with optional CFG
            if guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None:
                # CFG: predict with both conditions
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = [
                    torch.cat([neg_condition[0], condition[0]], dim=0),
                    torch.cat([neg_condition[1], condition[1]], dim=0),
                ]

                eps_pred = self(x_input, t_input, condition=cond_input, fwd_pred_type="eps")
                eps_uncond, eps_cond = eps_pred.chunk(2)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self(x, t_batch, condition=condition, fwd_pred_type="eps")

            # Convert to x0 prediction
            x0_pred = self.noise_scheduler.eps_to_x0(x, eps_pred, t_batch)

            # DDIM step: x_next = alpha_next * x0 + sigma_next * eps
            t_next_batch = t_next.expand(x.shape[0])
            alpha_next = self.noise_scheduler.alpha(t_next_batch).view(-1, 1, 1, 1).to(x.dtype)
            sigma_next = self.noise_scheduler.sigma(t_next_batch).view(-1, 1, 1, 1).to(x.dtype)
            x = alpha_next * x0_pred + sigma_next * eps_pred

        return x
