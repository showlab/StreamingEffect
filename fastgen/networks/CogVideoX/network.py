# SPDX-FileCopyrightText: Copyright (c) 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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
import types
from typing import Any, Optional, Tuple, Union, Dict, Set, List
from einops import rearrange
import math
import torch
import torch.nn as nn
from torch import dtype

from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from torch.distributed.fsdp import fully_shard
from transformers import T5EncoderModel, T5Tokenizer

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import str2bool


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    encoder_hidden_states: torch.Tensor,
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb_timestep = self.time_embedding(t_emb, timestep_cond)
    emb = emb_timestep

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = self.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    # 3. Transformer blocks
    features = []
    for i, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # TODO: need to rewrite this since attention_kwargs is a dict. Can be removed in newer versions of diffusers.
            def create_custom_forward(module, attention_kwargs):
                def custom_forward(hidden_states, encoder_hidden_states, emb, image_rotary_emb):
                    return module(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,  # Pass as a keyword argument
                    )

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block, attention_kwargs),
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        if i in feature_indices:
            features.append(hidden_states)

        # If we have all the features, we can exit early
        if return_features_early and len(features) == len(feature_indices):
            return features

    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # 4. Final block
    # adalayernorm_forward_2d() and time_embedding_forward_2d() are used in Causvid; they handles heterogeneous timesteps
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    p = self.config.patch_size
    p_t = self.config.patch_size_t
    if p_t is None:
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if len(feature_indices) == 0:
        # no features requested, return only the model output
        out = output
    else:
        # score and features； score, features
        out = [output, features]

    if return_logvar:
        logvar = self.transformer.logvar_linear(emb_timestep)
        return out, logvar

    return out


# TODO: can be removed in newer versions of diffusers
def block_forward_kwargs(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_seq_length = encoder_hidden_states.size(1)
    attention_kwargs = attention_kwargs or {}

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        hidden_states, encoder_hidden_states, temb
    )

    # attention
    attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **attention_kwargs,
    )

    hidden_states = hidden_states + gate_msa * attn_hidden_states
    encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
        hidden_states, encoder_hidden_states, temb
    )

    # feed-forward
    norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
    ff_output = self.ff(norm_hidden_states)

    hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
    encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

    return hidden_states, encoder_hidden_states


class CogVideoXTextEncoder:
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_id_or_local_path,
            subfolder="text_encoder",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        # We never update the encoder, so freeze it
        self.text_encoder.eval().requires_grad_(False)

        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_id_or_local_path,
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
        else:
            tokenized = self.tokenizer(
                conditioning,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            tokens = tokenized.input_ids
            prompt_embeddings = self.text_encoder(tokens.to(self.text_encoder.device))[0]

        return prompt_embeddings

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.text_encoder.to(*args, **kwargs)
        return self


class CogVideoXVideoEncoder:
    def __init__(
        self,
        model_id_or_local_path: str,
    ):
        self.vae: AutoencoderKLCogVideoX = AutoencoderKLCogVideoX.from_pretrained(
            model_id_or_local_path,
            subfolder="vae",
            cache_dir=os.environ["HF_HOME"],
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        # We never update the encoder, so freeze it
        self.vae.eval().requires_grad_(False)

    def encode(self, real_videos: torch.Tensor) -> torch.Tensor:
        """Encode raw videos to latent space.

        Args:
            real_videos: Input video tensor of shape (B, C, T, H, W).

        Returns:
            Latent tensor of shape (B, C, T, H, W) scaled by VAE config.
        """
        latent_videos = self.vae.encode(real_videos, return_dict=False)[0].sample()
        return latent_videos * self.vae.config.scaling_factor

    def decode(self, latent_videos: torch.Tensor) -> torch.Tensor:
        """Decode latents back to video space.

        Args:
            latent_videos: Latent tensor of shape (B, C, T, H, W).

        Returns:
            Decoded video tensor of shape (B, C, T, H, W).
        """
        latents = 1 / self.vae.config.scaling_factor * latent_videos
        return self.vae.decode(latents).sample

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.vae.to(*args, **kwargs)
        return self


class CogVideoX(FastGenNetwork):
    """A CogVideoX teacher model for text-to-video diffusion distillation."""

    MODEL_PATH_5B = "THUDM/CogVideoX-5b"
    MODEL_PATH_2B = "THUDM/CogVideoX-2b"

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_PATH_2B,
        net_pred_type="v",
        schedule_type="cogvideox",
        load_pretrained: bool = True,
        disable_grad_ckpt: bool = False,
        **model_kwargs,
    ):
        """CogVideoX score model constructor.

        Args:
            model_id_or_local_path (str, optional): The huggingface model ID or local path to load.
                Defaults to "THUDM/CogVideoX-2b".
            net_pred_type (str, optional): Prediction type. Defaults to "v".
            schedule_type (str, optional): Schedule type. Defaults to "cogvideox".
            load_pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
            disable_grad_ckpt (bool, optional): Whether to disable gradient checkpointing.
                Defaults to False (checkpointing enabled).
        """
        # Pass model_id to the noise schedule for correct snr_shift_scale
        self.model_id = model_id_or_local_path
        model_kwargs["model_id"] = self.model_id

        # Initialize FastGenNetwork with CogVideoX-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id_or_local_path
        self.vae_scale_factor_spatial = 8

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id_or_local_path, load_pretrained)

        # Overwrite the transformer forward to get bottleneck feature
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        for block in self.transformer.transformer_blocks:
            block.forward = types.MethodType(block_forward_kwargs, block)

        # Enable/disable gradient checkpointing (controlled at init, like Wan)
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id_or_local_path: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading CogVideoX transformer from pretrained")
            self.transformer: CogVideoXTransformer3DModel = CogVideoXTransformer3DModel.from_pretrained(
                model_id_or_local_path,
                subfolder="transformer",
                cache_dir=os.environ["HF_HOME"],
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = CogVideoXTransformer3DModel.load_config(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing CogVideoX transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing CogVideoX transformer from config (no pretrained weights)")
                logger.warning("CogVideoX transformer being initialized from config. No weights are loaded!")
            self.transformer: CogVideoXTransformer3DModel = CogVideoXTransformer3DModel.from_config(config)

        # Add logvar_linear layer
        self.transformer.logvar_linear = torch.nn.Linear(1280, 1)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear layers and embeddings.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

        logger.debug("Reinitialized CogVideoX parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the CogVideoX network for FSDP.

        Note: We shard `self.transformer` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.
        FSDP2's fully_shard works by dynamically modifying __class__, so we apply it only
        to the transformer submodule which is a standard torch.nn.Module.
        """
        # Note: Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing:
            # Disable the built-in gradient checkpointing (which uses torch.utils.checkpoint)
            self.transformer.disable_gradient_checkpointing()
            # Apply FSDP-compatible activation checkpointing to the transformer
            apply_fsdp_checkpointing(self.transformer, check_fn=lambda block: isinstance(block, CogVideoXBlock))
            logger.info("Applied FSDP activation checkpointing to CogVideoX transformer blocks")

        # Apply FSDP sharding to each transformer block
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize the text and image encoders for the CogVideoX model."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for CogVideoX model."""
        self.text_encoder = CogVideoXTextEncoder(model_id_or_local_path=self.model_id)

    def init_vae(self):
        """Initialize the video encoder for CogVideoX model for visualization."""
        self.vae = CogVideoXVideoEncoder(model_id_or_local_path=self.model_id)

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.transformer.to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _prepare_rotary_positional_embeddings(
        self, height: int, width: int, num_frames: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dtype)
        return freqs_cos, freqs_sin

    def _unpatchify_features(self, x_t: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        B, T, C, H, W = x_t.shape
        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t
        feats = []
        for feat in features:
            if p_t is None:
                feat = feat.reshape(feat.shape[0], T, H // p, W // p, -1, p, p)
                # (B, D, T, H, W)
                feat = rearrange(feat, "B T Hp Wp D p1 p2 -> B D T (Hp p1) (Wp p2)")
            else:
                feat = feat.reshape(feat.shape[0], (T + p_t - 1) // p_t, H // p, W // p, -1, p_t, p, p)
                # (B, D, T, H, W)
                feat = rearrange(feat, "B Tp Hp Wp D pt p1 p2 -> B D (Tp pt) (Hp p1) (Wp p2)")
            feats.append(feat)
        return feats

    def _get_variables(self, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back=None):
        lamb = ((alpha_prod_t / (1 - alpha_prod_t)) ** 0.5).log()
        lamb_next = ((alpha_prod_t_prev / (1 - alpha_prod_t_prev)) ** 0.5).log()
        h = lamb_next - lamb

        if alpha_prod_t_back is not None:
            lamb_previous = ((alpha_prod_t_back / (1 - alpha_prod_t_back)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def _get_mult(self, h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back):
        mult1 = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * alpha_prod_t_prev**0.5

        if alpha_prod_t_back is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    @staticmethod
    def _compute_local_guidance_scale(
        t: torch.Tensor, guidance_scale: Optional[float] = None, use_dynamic_cfg: bool = False, num_steps: int = 50
    ) -> Optional[float]:
        if use_dynamic_cfg:
            guidance_scale = guidance_scale or 1.0
            return 1 + guidance_scale * ((1 - math.cos(math.pi * ((num_steps - t.item()) / num_steps) ** 5.0)) / 2)
        return guidance_scale

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        use_dynamic_cfg: Optional[bool] = False,
        num_steps: Optional[int] = 50,
        timestep_spacing: Optional[str] = "trailing",
        **kwargs,
    ) -> torch.Tensor:
        """Multistep sample using the DPM solver ++ method

        Args:
            noise (torch.Tensor): The noisy latents to start from.
            condition (torch.Tensor, optional): The condition
            neg_condition (torch.Tensor, optional): The negative condition
            guidance_scale: the guidance scale. None means no guidance.
            use_dynamic_cfg (bool, optional): Whether to use dynamic config
            num_steps (int, optional): The number of steps to sample
            timestep_spacing (str, optional): The time step spacing
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: The sample output.
        """
        assert self.schedule_type == "cogvideox", f"{self.schedule_type} is not supported"

        num_train_timesteps = self.noise_scheduler.config["num_train_timesteps"]
        if timestep_spacing == "trailing":
            timesteps = (torch.linspace(1000, 1000 // num_steps, num_steps) - 1).to(torch.int64)
        else:
            raise NotImplementedError

        # Initialize latents with proper scaling based on the initial timestep
        t_init = timesteps[0].to(noise.device) / num_train_timesteps
        x_t = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        prev_pred_xhat = None
        for i, t in enumerate(timesteps):
            # 0. shape conversion
            timestep = t.expand(x_t.shape[0]).to(x_t.device)

            # 1. get previous step value (=t-1)
            prev_timestep = timesteps[i + 1] if i < num_steps - 1 else -1
            timestep_back = timesteps[i - 1] if i > 0 else None

            # 2. compute alphas, betas
            alpha_prod_t = self.noise_scheduler._alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.noise_scheduler._alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else torch.tensor(
                    1.0,
                    dtype=self.noise_scheduler._alphas_cumprod.dtype,
                    device=self.noise_scheduler._alphas_cumprod.device,
                )
            )
            alpha_prod_t_back = (
                self.noise_scheduler._alphas_cumprod[timestep_back] if timestep_back is not None else None
            )

            pred_xhat = self(x_t, timestep / num_train_timesteps, condition=condition, fwd_pred_type="x0")

            # apply guidance
            local_guidance_scale = CogVideoX._compute_local_guidance_scale(
                t, guidance_scale=guidance_scale, use_dynamic_cfg=use_dynamic_cfg, num_steps=num_steps
            )
            if local_guidance_scale is not None and local_guidance_scale > 1:
                pred_xhat_neg = self(x_t, timestep / num_train_timesteps, condition=neg_condition, fwd_pred_type="x0")
                pred_xhat = pred_xhat + (local_guidance_scale - 1) * (pred_xhat - pred_xhat_neg)

            h, r, _, _ = self._get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
            mult = list(self._get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
            mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

            noise = torch.randn_like(x_t)
            prev_sample = mult[0] * x_t - mult[1] * pred_xhat + mult_noise * noise
            if prev_pred_xhat is None or prev_timestep < 0:
                # Save a network evaluation if all noise levels are 0 or on the first step
                x_t = prev_sample
            else:
                denoised_d = mult[2] * pred_xhat - mult[3] * prev_pred_xhat
                noise = torch.randn_like(x_t)
                x_advanced = mult[0] * x_t - mult[1] * denoised_d + mult_noise * noise
                x_t = x_advanced

            prev_pred_xhat = pred_xhat

        return x_t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the StableDiffusion latent diffusion score model.

        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (torch.Tensor): The condition information.
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_logvar: If true, the forward pass returns the logvar.
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.

        Returns:
            torch.Tensor: The score model output.
        """
        if r is not None:
            # TODO: add support for CogVideoX
            raise NotImplementedError("r is not yet supported for CogVideoX")
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # rearrange x_t from B, C, T, H, W to B, T, C, H, W since CogVideoX expects time first, then channels
        x_t = rearrange(x_t, "b c t h w -> b t c h w")

        if "5b" in self.model_id:
            latent_height, latent_width = x_t.shape[3], x_t.shape[4]
            height = latent_height * self.vae_scale_factor_spatial
            width = latent_width * self.vae_scale_factor_spatial

            image_rotary_emb = self._prepare_rotary_positional_embeddings(
                height, width, x_t.shape[1], device=x_t.device, dtype=x_t.dtype
            )
        else:
            image_rotary_emb = None

        model_outputs = self.transformer(
            x_t,
            self.noise_scheduler.rescale_t(t).to(dtype=x_t.dtype),
            condition,
            image_rotary_emb=image_rotary_emb,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t, model_outputs)

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            out = rearrange(out, "b t c h w -> b c t h w")
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            out[0] = rearrange(out[0], "b t c h w -> b c t h w")
            out[1] = self._unpatchify_features(x_t, out[1])  # already in B, C, T, H, W

        if return_logvar:
            return out, logvar
        return out
