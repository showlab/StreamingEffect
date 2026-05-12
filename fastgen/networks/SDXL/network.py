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

import os
from typing import Any, Optional, List, Set
import types
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
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.SD15.network import classify_forward
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


class SDXLTextEncoder:
    def __init__(self, model_id):
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder_one.eval().requires_grad_(False)

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder_two.eval().requires_grad_(False)

        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_2",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )

    def encode(self, conditioning: Optional[Any] = None, precision: dtype = torch.float32) -> List[torch.Tensor]:
        text_input_ids_one = self.tokenizer_one(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            conditioning,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_list = []

        for text_input_ids, text_encoder in zip(
            [text_input_ids_one, text_input_ids_two], [self.text_encoder_one, self.text_encoder_two]
        ):
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            pooled_prompt_embeds = prompt_embeds[0].to(precision)

            prompt_embeds = prompt_embeds.hidden_states[-2].to(precision)
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            len(text_input_ids_one), -1
        )  # use the second text encoder's pooled prompt embeds (overwrite in for loop)

        return [prompt_embeds, pooled_prompt_embeds]

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.text_encoder_one.to(*args, **kwargs)
        self.text_encoder_two.to(*args, **kwargs)
        return self


class SDXLImageEncoder:
    def __init__(
        self,
        model_id: str,
    ):
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
        images = self.vae.decode(latents, return_dict=False)[0].clip_(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.vae.to(*args, **kwargs)
        return self


class StableDiffusionXL(FastGenNetwork):
    """A StableDiffusion XL teacher model for text-to-image diffusion distillation."""

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type="eps",
        schedule_type="sdxl",
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """StableDiffusion score model constructor.

        Args:
            model_id (str, optional): The huggingface model ID to load.
                Defaults to "stabilityai/stable-diffusion-xl-base-1.0".
            net_pred_type (str, optional): Prediction type. Defaults to "eps".
            schedule_type (str, optional): Schedule type. Defaults to "sdxl".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
        """
        # Initialize FastGenNetwork with SDXL-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id, load_pretrained)

        self.add_time_ids = self.build_condition_input(1024)

        self.unet.forward = types.MethodType(classify_forward, self.unet)
        self.unet.enable_gradient_checkpointing()

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
            logger.info("Loading SDXL UNet from pretrained")
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
                logger.info("Initializing SDXL UNet on meta device (zero memory, will receive weights via FSDP sync)")
            else:
                logger.info("Initializing SDXL UNet from config (no pretrained weights)")
                logger.warning("SDXL UNet being initialized from config. No weights are loaded!")
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_config(config)

        # Add logvar_linear layer
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

        logger.debug("Reinitialized SDXL parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the SDXL network for FSDP.

        Note: We shard `self.unet` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.

        SDXL uses a U-Net architecture with:
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
            logger.info("Applied FSDP activation checkpointing to SDXL U-Net blocks")

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
        """Initialize the text and image encoders for the SDXL model."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for SDXL model."""
        self.text_encoder = SDXLTextEncoder(model_id=self.model_id)

    def init_vae(self):
        """Initialize the image encoder for SDXL model."""
        self.vae = SDXLImageEncoder(model_id=self.model_id)

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

    def build_condition_input(self, resolution):
        original_size = (resolution, resolution)
        target_size = (resolution, resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32)
        return add_time_ids

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
            # TODO: add support for SDXL
            raise NotImplementedError("r is not yet supported for SDXL")
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # Get prompt embeddings
        text_embedding, pooled_text_embedding = condition[0], condition[1]
        add_time_ids = self.add_time_ids.repeat(x_t.shape[0], 1)
        unet_added_conditions = {"time_ids": add_time_ids.to(x_t.device), "text_embeds": pooled_text_embedding}

        # Note: Don't cast timestep to x_t.dtype here - the UNet's time_embedding expects
        # float32 input, and autocast will handle dtype conversions inside the UNet
        model_outputs = self.unet(
            x_t,
            self.noise_scheduler.rescale_t(t),
            text_embedding,
            added_cond_kwargs=unet_added_conditions,
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
            condition: List of [text_embedding, pooled_text_embedding].
            neg_condition: List of negative [text_embedding, pooled_text_embedding] for CFG.
            guidance_scale: CFG guidance scale. None disables guidance.
            num_steps: Number of sampling steps.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples in latent space.
        """
        # Get timestep schedule (SDXL uses t in [0, 1], mapped to [0, 1000])
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
            alpha_next = self.noise_scheduler.alpha(t_next.expand(x.shape[0])).view(-1, 1, 1, 1).to(x.dtype)
            sigma_next = self.noise_scheduler.sigma(t_next.expand(x.shape[0])).view(-1, 1, 1, 1).to(x.dtype)
            x = alpha_next * x0_pred + sigma_next * eps_pred

        return x
