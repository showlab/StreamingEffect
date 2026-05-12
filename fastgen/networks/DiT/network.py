# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transformer-based diffusion model architecture for FastGen.
This implementation provides a clean, modular DiT-style architecture with
adaptive normalization and patch-based processing for latent diffusion.
"""

from typing import List, Set, Optional, Union, Tuple, Mapping, Any
from functools import partial

import numpy as np
import math
import timm
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
import fastgen.utils.logging_utils as logger


def apply_adaptive_modulation(features, shift, scale):
    """
    Apply adaptive modulation to features using shift and scale parameters.

    Args:
        features: Input features of shape (B, N, D)
        shift: Shift parameters of shape (B, D)
        scale: Scale parameters of shape (B, D)

    Returns:
        Modulated features of shape (B, N, D)
    """
    return features * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#          Time and Condition Embedding Modules for Adaptive Conditioning        #
#################################################################################


class FourierTimeEmbedding(nn.Module):
    """
    Fourier-based time embedding module that projects scalar timesteps
    to high-dimensional vectors using sinusoidal frequency encoding.
    """

    def __init__(self, embed_dim, frequency_dim=256, max_freq=10000):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_freq = max_freq

        # Projection network with residual connection
        self.proj_net = nn.Sequential(
            nn.Linear(frequency_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

    def encode_timesteps(self, timesteps, dim):
        """
        Generate Fourier features from timesteps using sinusoidal encoding.

        Args:
            timesteps: (B, ) tensor of timestep values
            dim: Dimension of the frequency encoding

        Returns:
            (B, dim) tensor of Fourier features
        """
        original_dtype = timesteps.dtype
        half_dim = dim // 2

        # Compute frequency bands
        freq_bands = torch.exp(
            -math.log(self.max_freq)
            * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )

        # Compute sinusoidal encodings
        angles = timesteps[:, None].float() * freq_bands[None, :]
        fourier_features = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        # Handle odd dimensions
        if dim % 2 == 1:
            fourier_features = torch.cat([fourier_features, torch.zeros_like(fourier_features[:, :1])], dim=-1)

        return fourier_features.to(original_dtype)

    def forward(self, t):
        fourier_feats = self.encode_timesteps(t, self.frequency_dim)
        embeddings = self.proj_net(fourier_feats)
        return embeddings


class ConditionalEmbedding(nn.Module):
    """
    Conditional embedding module for class labels with built-in classifier-free guidance support.
    Provides learnable embeddings for discrete class labels and handles stochastic masking
    for classifier-free guidance during training.
    """

    def __init__(self, num_classes, embed_dim, cfg_enabled, cfg_dropout_rate):
        super().__init__()
        self.num_classes = num_classes
        self.cfg_dropout_rate = cfg_dropout_rate if cfg_enabled else 0.0

        # Add one extra embedding for null/unconditional token when CFG is enabled
        num_embeddings = num_classes + (1 if cfg_dropout_rate > 0 else 0)
        self.class_embeddings = nn.Embedding(num_embeddings, embed_dim)

    def apply_cfg_mask(self, class_ids, force_mask=None):
        """
        Apply classifier-free guidance masking to class labels.

        Args:
            class_ids: (B, ) tensor of class indices
            force_mask: Optional (B, ) tensor to force specific samples to be masked

        Returns:
            (B, ) tensor with some labels replaced by unconditional token
        """
        if force_mask is not None:
            mask = force_mask == 1
        else:
            mask = torch.rand(class_ids.shape[0], device=class_ids.device) < self.cfg_dropout_rate

        # Replace masked labels with unconditional token index
        masked_ids = torch.where(mask, self.num_classes, class_ids)
        return masked_ids

    def forward(self, class_labels, is_training, force_mask=None):
        # Apply masking if in training mode or forced
        if (is_training and self.cfg_dropout_rate > 0) or (force_mask is not None):
            class_labels = self.apply_cfg_mask(class_labels, force_mask)

        return self.class_embeddings(class_labels)


#################################################################################
#           Transformer Block with Adaptive Normalization                       #
#################################################################################


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer normalization and gated residuals.

    This block implements a two-stage architecture:
    1. Multi-head self-attention with adaptive normalization
    2. Feed-forward MLP with adaptive normalization

    Both stages use gated residual connections conditioned on the input conditioning signal.
    """

    def __init__(self, model_dim, num_attention_heads, ffn_expansion=4.0, **kwargs):
        super().__init__()
        # Attention branch with parameter-free normalization
        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(model_dim, num_heads=num_attention_heads, qkv_bias=True, **kwargs)

        # FFN branch with parameter-free normalization
        self.ffn_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        ffn_hidden_dim = int(model_dim * ffn_expansion)
        self.feed_forward = Mlp(
            in_features=model_dim,
            hidden_features=ffn_hidden_dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=0,
        )

        # Adaptive modulation network: generates 6 conditioning parameters
        # (scale/shift/gate for attention, scale/shift/gate for FFN)
        self.conditioning_net = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 6 * model_dim, bias=True))

    def forward(self, hidden_states, conditioning):
        # Generate adaptive parameters from conditioning
        params = self.conditioning_net(conditioning).chunk(6, dim=1)
        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = params

        # Attention branch with adaptive modulation and gated residual
        normed_hidden = self.attn_norm(hidden_states)
        modulated_hidden = apply_adaptive_modulation(normed_hidden, attn_shift, attn_scale)
        attn_output = self.attention(modulated_hidden)
        hidden_states = hidden_states + attn_gate.unsqueeze(1) * attn_output

        # FFN branch with adaptive modulation and gated residual
        normed_hidden = self.ffn_norm(hidden_states)
        modulated_hidden = apply_adaptive_modulation(normed_hidden, ffn_shift, ffn_scale)
        ffn_output = self.feed_forward(modulated_hidden)
        hidden_states = hidden_states + ffn_gate.unsqueeze(1) * ffn_output

        return hidden_states


class OutputProjection(nn.Module):
    """
    Final output projection layer with adaptive normalization.
    Projects transformer hidden states back to patch-space predictions.
    """

    def __init__(self, hidden_dim, patch_dim, output_channels):
        super().__init__()
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.projection = nn.Linear(hidden_dim, patch_dim * patch_dim * output_channels, bias=True)

        # Conditioning network for adaptive normalization
        self.adaptive_params = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))

    def forward(self, hidden_states, conditioning):
        # Apply adaptive normalization
        shift, scale = self.adaptive_params(conditioning).chunk(2, dim=1)
        normalized = apply_adaptive_modulation(self.output_norm(hidden_states), shift, scale)

        # Project to output space
        output = self.projection(normalized)
        return output


class DiT(FastGenNetwork):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        enable_class_dropout=False,
        num_classes=1000,
        learn_sigma=False,
        r_timestep=False,
        scale_t=True,  # whether to rescale the timesteps
        time_cond_type="abs",  # "abs" or "diff"
        net_pred_type="flow",  # Prediction type for FastGenNetwork
        schedule_type="rf",  # Schedule type for FastGenNetwork
        enable_fused_attn=False,  # disable fused attention for jvp with DiT
        use_sit_convention=False,  # Use SiT convention: t -> 1-t and v -> -v
        **model_kwargs,
    ):
        timm.layers.set_fused_attn(enable_fused_attn)
        self.use_sit_convention = use_sit_convention

        # Initialize FastGenNetwork with DiT-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.img_resolution = input_size * 8

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = FourierTimeEmbedding(hidden_size)

        self.scale_t = scale_t
        self.r_timestep = r_timestep
        self.time_cond_type = time_cond_type
        if self.r_timestep:
            self.r_embedder = FourierTimeEmbedding(hidden_size)
        else:
            self.r_embedder = None

        self.y_embedder = ConditionalEmbedding(num_classes, hidden_size, enable_class_dropout, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Positional embeddings (persistent=True to save in state_dict)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size), persistent=True)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, ffn_expansion=mlp_ratio) for _ in range(depth)])
        self.final_layer = OutputProjection(hidden_size, patch_size, self.out_channels)
        self.logvar_linear = nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier initialization for linear layers
        def _xavier_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_xavier_init)

        # Initialize positional embeddings with sinusoidal encoding
        grid_size = int(self.x_embedder.num_patches**0.5)
        pos_encoding = compute_sinusoidal_2d_embeddings(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_encoding).float().unsqueeze(0))

        # Initialize patch embedding projection
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize class embeddings
        nn.init.normal_(self.y_embedder.class_embeddings.weight, std=0.02)

        # Initialize time embedding networks
        nn.init.normal_(self.t_embedder.proj_net[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.proj_net[2].weight, std=0.02)

        if self.r_timestep:
            nn.init.normal_(self.r_embedder.proj_net[0].weight, std=0.02)
            nn.init.normal_(self.r_embedder.proj_net[2].weight, std=0.02)

        # Zero-initialize adaptive conditioning layers for residual path
        for block in self.blocks:
            nn.init.constant_(block.conditioning_net[-1].weight, 0)
            nn.init.constant_(block.conditioning_net[-1].bias, 0)

        # Zero-initialize output projection layers
        nn.init.constant_(self.final_layer.adaptive_params[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaptive_params[-1].bias, 0)
        nn.init.constant_(self.final_layer.projection.weight, 0)
        nn.init.constant_(self.final_layer.projection.bias, 0)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2. Non-persistent
        buffers (like positional embeddings) must be recomputed on all ranks after
        materializing from meta device.
        """
        self.initialize_weights()

        super().reset_parameters()

        logger.debug("Reinitialized DiT parameters")

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Load state dict with automatic conversion for Facebook DiT checkpoints.

        Converts key names from Facebook's DiT checkpoint format to our naming convention:
            - t_embedder.mlp -> t_embedder.proj_net
            - blocks.X.attn -> blocks.X.attention
            - blocks.X.mlp -> blocks.X.feed_forward
            - blocks.X.adaLN_modulation -> blocks.X.conditioning_net
            - y_embedder.embedding_table -> y_embedder.class_embeddings
            - final_layer.linear -> final_layer.projection
            - final_layer.adaLN_modulation -> final_layer.adaptive_params
        """
        # Check if this looks like a Facebook DiT checkpoint
        is_facebook_format = any(
            k.startswith("t_embedder.mlp") or ".attn." in k or ".adaLN_modulation" in k for k in state_dict.keys()
        )
        if is_facebook_format:
            logger.info("Detected Facebook DiT checkpoint format, converting keys...")
            new_state_dict = {}
            pos_embed_value = None

            for key, value in state_dict.items():
                new_key = key

                # Store pos_embed to load directly into buffer
                if key == "pos_embed":
                    pos_embed_value = value
                    continue

                # Apply key conversions (order matters - more specific rules first)
                new_key = new_key.replace("t_embedder.mlp", "t_embedder.proj_net")
                new_key = new_key.replace(".attn.", ".attention.")
                new_key = new_key.replace(".mlp.", ".feed_forward.")
                new_key = new_key.replace("y_embedder.embedding_table", "y_embedder.class_embeddings")
                new_key = new_key.replace("final_layer.linear", "final_layer.projection")
                # Handle final_layer.adaLN_modulation BEFORE the generic .adaLN_modulation. rule
                new_key = new_key.replace("final_layer.adaLN_modulation", "final_layer.adaptive_params")
                new_key = new_key.replace(".adaLN_modulation.", ".conditioning_net.")

                new_state_dict[new_key] = value

            logger.info(f"Converted {len(new_state_dict)} keys from Facebook DiT format")
            state_dict = new_state_dict

            # Load pos_embed directly into buffer (important for SiT!)
            if pos_embed_value is not None:
                if pos_embed_value.shape == self.pos_embed.shape:
                    self.pos_embed.data.copy_(pos_embed_value)
                    logger.info("Loaded pos_embed from checkpoint")
                else:
                    logger.warning(
                        f"pos_embed shape mismatch: checkpoint {pos_embed_value.shape} vs model {self.pos_embed.shape}"
                    )

        return super().load_state_dict(state_dict, **kwargs)

    def fully_shard(self, **kwargs):
        """
        Apply FSDP sharding to DiT network components.

        Shards each transformer block and embedding module independently
        for optimal memory efficiency in distributed training.
        """
        # Shard each transformer block
        for block in self.blocks:
            fully_shard(block, **kwargs)

        # Shard embedding and projection modules
        fully_shard(self.x_embedder, **kwargs)
        fully_shard(self.t_embedder, **kwargs)
        fully_shard(self.y_embedder, **kwargs)
        fully_shard(self.final_layer, **kwargs)

        if self.r_embedder is not None:
            fully_shard(self.r_embedder, **kwargs)

    def init_preprocessors(self):
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_vae(self):
        """Initialize the VAE for visualization/decoding."""
        self.vae = SDVAE()

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        super().to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def unpatchify(self, x):
        """
        Reconstruct spatial features from patch tokens.

        Args:
            x: (N, T, patch_size^2 * C) tensor of patch tokens

        Returns:
            imgs: (N, C, H, W) tensor of spatial features
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def prepare_t(self, t: Optional[torch.Tensor] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor | None:
        if t is None:
            return t
        elif self.scale_t:
            t = self.noise_scheduler.rescale_t(t)
        return t.to(dtype=dtype)

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
        """
        Forward pass of DiT.
        x_t: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        condition: (N, C) tensor of class labels
        r: (N, ) tensor of another diffusion timesteps
        """
        if feature_indices is None:
            feature_indices = {}
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"
        x_in, t_in = x_t, t

        # condition is passed in as one-hot encodings
        if condition.ndim == 2:
            # this mimics the behavior of LabelEmbedder.token_drop, mapping a zero-vector to `self.num_classes`
            # note that this requires `class_dropout_prob > 0`
            # such that LabelEmbedder.embedding_table is initialized with one embedding more
            mask = torch.any(condition, dim=1)
            condition = torch.where(~mask, self.num_classes, condition.argmax(dim=1))

        t = self.prepare_t(t, dtype=x_t.dtype)
        r = self.prepare_t(r, dtype=x_t.dtype)

        # SiT convention: t=0 is noise, t=1 is data (opposite of RF)
        # Transform timestep so SiT-trained network sees its expected convention
        if self.use_sit_convention:
            t = 1 - t

        # Ensure pos_embed buffer is on the same device as input
        if self.pos_embed.device != x_t.device:
            self.pos_embed = self.pos_embed.to(x_t.device)

        # Embed spatial patches and add positional information
        x = self.x_embedder(x_t) + self.pos_embed

        # Embed time conditioning
        t_emb = self.t_embedder(t)

        # Handle secondary time conditioning (for two-stage models)
        if self.r_embedder is not None and r is not None:
            if self.time_cond_type == "diff":
                r_emb = self.r_embedder(t - r)
            elif self.time_cond_type == "abs":
                r_emb = self.r_embedder(r)
            else:
                raise ValueError(f"Invalid time_cond_type: {self.time_cond_type}")
        else:
            r_emb = torch.zeros_like(t_emb)

        # Embed class conditioning
        y = self.y_embedder(condition, self.training)

        # Combine all conditioning signals
        c = t_emb + y + r_emb

        # Process through transformer blocks
        features = []
        for idx, block in enumerate(self.blocks):
            x = block(x, c)
            if idx in feature_indices:
                features.append(x)

            if return_features_early and len(features) == len(feature_indices):
                return features

        # Project to output space
        x = self.final_layer(x, c)

        # Reconstruct spatial output
        x = self.unpatchify(x)

        # legacy issue from SiT, where the second half of the channels are unused
        if self.learn_sigma:
            x = x.chunk(2, dim=1)[0]

        # SiT convention: v_sit = x_0 - eps, but RF expects v_rf = eps - x_0
        # Note: "eps" and "x0" predictions don't need negation
        if self.use_sit_convention and self.net_pred_type == "flow":
            x = -x

        x = self.noise_scheduler.convert_model_output(
            x_in, x, t_in, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
        )

        # Prepare output
        if len(feature_indices) == 0:
            out = x
        else:
            out = [x, features]

        # Optionally compute log variance
        if return_logvar:
            logvar = self.logvar_linear(t_emb)
            return out, logvar

        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using appropriate sampler based on schedule_type.

        Args:
            noise: Initial noise tensor [B, C, H, W].
            condition: Class label conditioning (class indices or one-hot).
            neg_condition: Negative conditioning for CFG.
            guidance_scale: CFG guidance scale. None disables guidance.
            num_steps: Number of sampling steps.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples in latent space.
        """
        if self.schedule_type in ("sd", "sdxl", "alphas"):
            return self._sample_ddim(noise, condition, neg_condition, guidance_scale, num_steps)
        elif self.schedule_type == "rf":
            return self._sample_flow(noise, condition, neg_condition, guidance_scale, num_steps)
        else:
            raise ValueError(f"Invalid schedule_type: {self.schedule_type}")

    def _sample_flow(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Generate samples using Euler method for flow matching.

        Supports both RF and RF2 noise schedules via the noise_scheduler:
        - RF:  x_t = (1-t)*x_0 + t*eps, t: max_t → min_t (descending)
        - RF2: x_t = t*x_0 + (1-t)*eps, t: min_t → max_t (ascending)

        The ODE is integrated using: x = x + dt * v
        where dt and v signs are determined by the noise schedule convention.

        Args:
            noise: Initial noise tensor.
            condition: Class conditioning (one-hot or indices).
            neg_condition: Negative/null conditioning for CFG.
            guidance_scale: CFG scale. >1.0 enables guidance.
            num_steps: Number of sampling steps.
        """
        t_list = self.noise_scheduler.get_t_list(num_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=t_list[0])
        for t, t_next in zip(t_list[:-1], t_list[1:]):
            t_batch = t.expand(x.shape[0])
            dt = (t_next - t).to(x.dtype)

            # Get velocity prediction with optional CFG
            if guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None:
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([neg_condition, condition], dim=0)

                v = self(x_input, t_input, condition=cond_input, fwd_pred_type="flow")
                v_uncond, v_cond = v.chunk(2)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self(x, t_batch, condition=condition, fwd_pred_type="flow")

            # Euler step: x = x + dt * v
            x = x + dt * v

        return x

    def _sample_ddim(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Generate samples using DDIM (for DDPM schedule with epsilon prediction).

        Uses diffusers DDIMScheduler which provides integer timesteps (0-999).
        We normalize them to [0, 1] before passing to forward().
        """
        num_train_timesteps = 1000
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_steps, device=noise.device)

        # Initialize latents with proper scaling based on the initial timestep
        t_init = torch.as_tensor(
            scheduler.timesteps[0].float() / num_train_timesteps,
            device=noise.device,
            dtype=noise.dtype,
        )
        x = self.noise_scheduler.latents(noise=noise, t_init=t_init)
        for t in scheduler.timesteps:
            # Normalize timestep from [0, 999] to [0, 1]
            t_normalized = t.float() / num_train_timesteps
            t_batch = t_normalized.expand(x.shape[0])

            if guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None:
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([neg_condition, condition], dim=0)

                eps_pred = self(x_input, t_input, condition=cond_input)
                eps_uncond, eps_cond = eps_pred.chunk(2)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self(x, t_batch, condition=condition)

            # DDIM step (uses original integer timestep)
            x = scheduler.step(eps_pred, t, x, return_dict=False)[0]

        return x


#################################################################################
#              2D Sinusoidal Positional Encoding Generation                     #
#################################################################################


def compute_sinusoidal_2d_embeddings(
    embedding_dim: int,
    spatial_size: int,
    include_cls_token: bool = False,
    num_extra_tokens: int = 0,
    max_wavelength: float = 10000.0,
) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings for a square spatial grid.

    Args:
        embedding_dim: Dimension of the output embeddings (must be even)
        spatial_size: Height and width of the spatial grid
        include_cls_token: Whether to prepend zeros for classification token
        num_extra_tokens: Number of extra token positions to prepend
        max_wavelength: Maximum wavelength for frequency bands

    Returns:
        pos_embeddings: Array of shape (num_positions, embedding_dim)
    """
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"

    # Generate coordinate grids
    y_coords = np.arange(spatial_size, dtype=np.float32)
    x_coords = np.arange(spatial_size, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Flatten spatial coordinates
    y_flat = y_grid.reshape(-1)  # (H*W,)
    x_flat = x_grid.reshape(-1)  # (H*W,)

    # Split embedding dimension between x and y coordinates
    dim_per_axis = embedding_dim // 2

    # Encode x and y coordinates separately
    x_embeddings = _encode_1d_positions(x_flat, dim_per_axis, max_wavelength)
    y_embeddings = _encode_1d_positions(y_flat, dim_per_axis, max_wavelength)

    # Concatenate x and y embeddings
    pos_embeddings = np.concatenate([x_embeddings, y_embeddings], axis=1)

    # Optionally prepend zeros for extra tokens
    if include_cls_token and num_extra_tokens > 0:
        extra_token_embeddings = np.zeros((num_extra_tokens, embedding_dim), dtype=np.float32)
        pos_embeddings = np.concatenate([extra_token_embeddings, pos_embeddings], axis=0)

    return pos_embeddings


def _encode_1d_positions(positions: np.ndarray, output_dim: int, max_wavelength: float = 10000.0) -> np.ndarray:
    """
    Encode 1D positions using sinusoidal functions with varying frequencies.

    Args:
        positions: Array of position values, shape (num_positions,)
        output_dim: Output dimension (must be even)
        max_wavelength: Maximum wavelength for frequency scaling

    Returns:
        embeddings: Array of shape (num_positions, output_dim)
    """
    assert output_dim % 2 == 0, "Output dimension must be even"

    # Compute frequency bands
    half_dim = output_dim // 2
    freq_bands = np.arange(half_dim, dtype=np.float64)
    freq_bands = freq_bands / float(half_dim)
    freq_bands = 1.0 / (max_wavelength**freq_bands)

    # Compute position-frequency products
    angles = np.outer(positions, freq_bands)  # (num_positions, half_dim)

    # Apply sinusoidal functions
    sin_embeddings = np.sin(angles)
    cos_embeddings = np.cos(angles)

    # Interleave sin and cos embeddings
    embeddings = np.concatenate([sin_embeddings, cos_embeddings], axis=1)

    return embeddings


class SDVAE:
    """
    The Stable Diffusion VAE for DiT.

    Handles both:
    - Raw RGB images (3 channels) -> encodes to latents
    - Pre-encoded latents (8 channels = mean + std) -> samples and scales
    """

    def __init__(self):
        self._vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().requires_grad_(False)
        self.scaling_factor = self._vae.config.scaling_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents.

        Args:
            x: Either RGB images [B, 3, H, W] or pre-encoded [B, 8, H, W] (mean + std).

        Returns:
            Scaled latents [B, 4, H//8, W//8].
        """
        if x.shape[1] == 3:
            # Raw RGB images - encode with VAE
            d = self._vae.encode(x)["latent_dist"]
            latents = d.mean + torch.randn_like(d.mean) * d.std
        elif x.shape[1] == 8:
            # Pre-encoded latents (mean + std) - sample from distribution
            mean, std = x.chunk(2, dim=1)
            latents = mean + torch.randn_like(mean) * std
        else:
            raise ValueError(f"Expected 3 (RGB) or 8 (mean+std) channels, got {x.shape[1]}")
        return latents * self.scaling_factor

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        latents = x / self.scaling_factor
        images = self._vae.decode(latents, return_dict=False)[0].clamp(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self._vae.to(*args, **kwargs)
        return self
