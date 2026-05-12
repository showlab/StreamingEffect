# SPDX-FileCopyrightText: Copyright (c) 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Optional, List, Set, Dict, Union, Tuple
import torch
from einops import rearrange
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models import WanVACETransformer3DModel

from fastgen.networks.Wan.network import Wan, flatten_timestep, unflatten_timestep_proj
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.VaceWan.modules.vace_depth_annotator import VACEDepthExtractor

import fastgen.utils.logging_utils as logger
from fastgen.utils.basic_utils import str2bool


def vace_classify_forward_prepare(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
]:
    """Prepare inputs for VACE transformer forward pass."""

    del attention_kwargs  # unused for base VACE path

    batch_size, _, _, _, _ = hidden_states.shape

    if control_hidden_states is not None and control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))

    rotary_emb = self.rope(hidden_states)

    # Ensure RoPE buffers are on the same device as input, useful for cpu_offloading
    if isinstance(rotary_emb, (tuple, list)):
        rotary_emb = tuple(emb.to(device=hidden_states.device, dtype=hidden_states.dtype) for emb in rotary_emb)
    elif isinstance(rotary_emb, torch.Tensor):
        rotary_emb = rotary_emb.to(device=hidden_states.device, dtype=hidden_states.dtype)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    processed_control_states: Optional[torch.Tensor] = None
    processed_control_scales: Optional[List[torch.Tensor]] = None
    if control_hidden_states is not None:
        processed_control_states = self.vace_patch_embedding(control_hidden_states)
        processed_control_states = processed_control_states.flatten(2).transpose(1, 2)
        control_hidden_states_padding = processed_control_states.new_zeros(
            batch_size,
            hidden_states.size(1) - processed_control_states.size(1),
            processed_control_states.size(2),
        )
        processed_control_states = torch.cat([processed_control_states, control_hidden_states_padding], dim=1)
        processed_control_scales = (
            list(torch.unbind(control_hidden_states_scale)) if control_hidden_states_scale is not None else None
        )

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if self.r_embedder is not None and r_timestep is not None:
        if self.time_cond_type == "abs":
            # Keep the r_timestep as is
            pass
        elif self.time_cond_type == "diff":
            r_timestep = timestep - r_timestep
        else:
            raise ValueError(f"Invalid time condition: {self.time_cond_type}")

        r_timestep, rs_seq_len = flatten_timestep(r_timestep)
        r_timestep = self.r_embedder.timesteps_proj(r_timestep)
        time_embedder_dtype = next(iter(self.r_embedder.time_embedder.parameters())).dtype
        if r_timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            r_timestep = r_timestep.to(time_embedder_dtype)

        remb = self.r_embedder.time_embedder(r_timestep).type_as(encoder_hidden_states)
        r_timestep_proj = self.r_embedder.time_proj(self.r_embedder.act_fn(remb))
        r_timestep_proj = unflatten_timestep_proj(r_timestep_proj, rs_seq_len)

        if getattr(self, "encoder_depth", None) is None:
            timestep_proj = timestep_proj + r_timestep_proj
            temb = temb + remb
        else:
            temb = remb
    elif r_timestep is not None:
        # Raise an error here, otherwise we silently ignore the r_timestep
        raise ValueError("r_timestep provided but no r_embedder is present")
    else:
        r_timestep_proj = None

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
        processed_control_states,
        processed_control_scales,
    )


def vace_classify_forward_block_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    r_timestep_proj: Optional[torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[List[torch.Tensor]] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: bool = False,
    lora_scale: float = 1.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Execute transformer blocks with optional VACE control."""

    del attention_kwargs  # unused for base VACE path

    features: List[torch.Tensor] = []
    feature_indices = feature_indices or set()
    idx = 0

    gradient_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
    has_control = control_hidden_states is not None and hasattr(self, "vace_blocks")

    if gradient_ckpt:
        control_hidden_states_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        current_control_states = control_hidden_states
        if has_control:
            assert control_hidden_states_scale is not None
            for layer_idx, block in enumerate(self.vace_blocks):
                conditioning_states, current_control_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                )  # type: ignore
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[layer_idx]))
            control_hidden_states_list = control_hidden_states_list[::-1]

        for block_idx, block in enumerate(self.blocks):
            if skip_layers is not None and block_idx in skip_layers:
                continue

            _encoder_depth = getattr(self, "encoder_depth", None)
            timestep_proj_to_use = (
                r_timestep_proj
                if _encoder_depth is not None and block_idx == _encoder_depth and r_timestep_proj is not None
                else timestep_proj
            )

            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj_to_use,
                rotary_emb,
            )  # type: ignore

            if has_control and block_idx in self.config.vace_layers:
                assert control_hidden_states_list, "Control states depleted before applying to configured layers"
                control_hint, scale = control_hidden_states_list.pop()
                hidden_states = hidden_states + control_hint * scale

            if idx in feature_indices:
                features.append(hidden_states)
            idx += 1
    else:
        control_hidden_states_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        current_control_states = control_hidden_states
        if has_control:
            assert control_hidden_states_scale is not None
            for layer_idx, block in enumerate(self.vace_blocks):
                conditioning_states, current_control_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    current_control_states,
                    timestep_proj,
                    rotary_emb,
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[layer_idx]))
            control_hidden_states_list = control_hidden_states_list[::-1]

        for block_idx, block in enumerate(self.blocks):
            if skip_layers is not None and block_idx in skip_layers:
                continue

            _encoder_depth = getattr(self, "encoder_depth", None)
            timestep_proj_to_use = (
                r_timestep_proj
                if _encoder_depth is not None and block_idx == _encoder_depth and r_timestep_proj is not None
                else timestep_proj
            )

            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj_to_use, rotary_emb)

            if has_control and block_idx in self.config.vace_layers:
                assert control_hidden_states_list, "Control states depleted before applying to configured layers"
                control_hint, scale = control_hidden_states_list.pop()
                hidden_states = hidden_states + control_hint * scale

            if idx in feature_indices:
                features.append(hidden_states)
            idx += 1

    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return hidden_states, features

    return hidden_states, features


def vace_classify_forward(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: Optional[torch.Tensor] = None,
    control_hidden_states_scale: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward pass with feature collection and logvar support for VACE model."""
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        temb,
        rotary_emb,
        processed_control_states,
        processed_control_scales,
    ) = self.classify_forward_prepare(
        hidden_states,
        timestep,
        encoder_hidden_states,
        r_timestep,
        encoder_hidden_states_image,
        control_hidden_states,
        control_hidden_states_scale,
        attention_kwargs,
    )

    hidden_states, features = self.classify_forward_block_forward(
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        rotary_emb,
        processed_control_states,
        processed_control_scales,
        skip_layers,
        feature_indices,
        bool(return_features_early),
        lora_scale,
        attention_kwargs,
    )

    if return_features_early:
        return features

    if temb.dim() == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if shift.shape[1] == hidden_states.shape[1]:
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        else:
            frame_seqlen = post_patch_height * post_patch_width
            hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (post_patch_num_frames, frame_seqlen))
            hidden_states = (
                (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
            )
    else:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if feature_indices is None or len(feature_indices) == 0:
        out: Union[torch.Tensor, List[torch.Tensor]] = output
    else:
        out = [output, features]

    if return_logvar:
        logvar = self.transformer.logvar_linear(temb)
        return out, logvar
    return out


class VACEWan(Wan):
    """VACE-WAN model following the exact VACE pattern."""

    MODEL_ID_1_3B = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    MODEL_ID_14B = "Wan-AI/Wan2.1-VACE-14B-diffusers"

    def __init__(
        self,
        depth_model_path: str = None,
        context_scale: float = 1.0,
        model_id_or_local_path: str = MODEL_ID_1_3B,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "abs",  # default to absolute for back compatibility
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        use_fsdp_checkpoint: bool = True,
        **model_kwargs,
    ):
        """Initialize VACE-WAN model.

        Args:
            depth_model_path: Path to depth model
            context_scale: Scale factor for context influence
            model_id_or_local_path: HuggingFace model ID or local path for base WAN
            r_timestep (bool): Whether to support meanflow-like models with r timestep. Defaults to False.
            disable_efficient_attn (bool, optional): Whether to disable efficient attention. Defaults to False.
            disable_grad_ckpt (bool, optional): Whether to disable checkpoints during training. Defaults to False.
            enable_logvar_linear (bool, optional): Whether to enable logvar linear prediction. Defaults to True.
            r_embedder_init (str, optional): Initialization method for the r embedder. Defaults to "pretrained".
            time_cond_type (str, optional): Time condition type. Defaults to "abs".
            norm_temb (bool, optional): Unused, but included for API compatibility.
            net_pred_type (str, optional): Prediction type. Defaults to "flow".
            schedule_type (str, optional): Schedule type. Defaults to "rf".
            encoder_depth (int, optional): The depth of the encoder (i.e. the number of blocks taking in t embeddings).
                Defaults to None, meaning all blocks take in [t embeddings + r embeddings].
            load_pretrained (bool, optional): Whether to load the model from pretrained.
                If False, the model will be initialized from the config.
                Defaults to True.
            use_fsdp_checkpoint (bool, optional): Whether to use FSDP gradient checkpointing. Defaults to True.
            **model_kwargs: Additional keyword arguments to pass to the FastGenNetwork constructor.
        """
        if norm_temb:
            logger.warning("norm_temb is not supported for VACE-WAN")
        super().__init__(
            model_id_or_local_path=model_id_or_local_path,
            r_timestep=r_timestep,
            disable_efficient_attn=disable_efficient_attn,
            disable_grad_ckpt=disable_grad_ckpt,
            enable_logvar_linear=enable_logvar_linear,
            r_embedder_init=r_embedder_init,
            time_cond_type=time_cond_type,
            norm_temb=norm_temb,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            encoder_depth=encoder_depth,
            load_pretrained=load_pretrained,
            use_fsdp_checkpoint=use_fsdp_checkpoint,
            **model_kwargs,
        )
        self.context_scale = context_scale

        # Depth annotator (lazy loading)
        self.depth_annotator = None
        self.depth_model_path = depth_model_path
        self._all_zero_latent = None

        # Flag to indicate this is a video-to-video model (mainly used for data processing)
        self.is_vid2vid = True

        torch.cuda.empty_cache()
        logger.info("Initialized VACE-WAN model")

    def override_transformer_forward(self, inner_dim: int) -> None:
        """Override the transformer forward method to use VACE-specific methods."""
        self.transformer.classify_forward_prepare = types.MethodType(vace_classify_forward_prepare, self.transformer)
        self.transformer.classify_forward_block_forward = types.MethodType(
            vace_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(vace_classify_forward, self.transformer)

    def _initialize_network(self, model_id_or_local_path: str, load_pretrained: bool) -> Tuple[str, int]:
        """Initialize the network.
        Args:
            model_id_or_local_path (str): The model ID or local path to load. This can be either a HuggingFace
                model ID or a local directory holding the model weights.
            load_pretrained (bool): Whether to load the model from pretrained. If False, the model will be initialized from the config.

        Returns:
            Tuple[str, int]: The model ID and the inner dimension.
                model_id (str): The model ID (extracted from local path if needed).
                inner_dim (int): The inner dimension.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading VaceWan transformer")
            self.transformer: WanVACETransformer3DModel = WanVACETransformer3DModel.from_pretrained(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = WanVACETransformer3DModel.load_config(
                model_id_or_local_path,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing VaceWan transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing VaceWan transformer from config (no pretrained weights)")
                logger.warning("VaceWan Transformer being initializated from config. No weights are loaded!")
            self.transformer: WanVACETransformer3DModel = WanVACETransformer3DModel.from_config(config)
        model_id = Wan.get_model_id(model_id_or_local_path)

        # Add logvar linear
        if model_id == self.MODEL_ID_14B:
            inner_dim = 5120
        else:
            inner_dim = 1536
        return model_id, inner_dim

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state dict excluding depth_annotator weights (loaded separately)."""
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith("depth_annotator.")}

    def prepare_vid_conditioning(
        self,
        video: torch.Tensor,
        condition_latents: torch.Tensor = None,
    ) -> torch.Tensor:
        """Prepare VACE conditioning by extracting depth and creating latents.

        Args:
            video: [B, C, T, H, W] video frames in range [-1, 1]
            condition_latents: [B, C, T, H, W] condition latents in range [-1, 1]

        Returns:
            conditioning: [B, 96, T_latent, H_latent, W_latent] (32 depth latents + 64 mask)
        """

        if condition_latents is None:
            if not hasattr(self, "vae") or self.vae is None:
                raise ValueError("VAE must be initialized to prepare video latents")

            # Initialize depth annotator if needed
            if self.depth_annotator is None:
                self.depth_annotator = VACEDepthExtractor(
                    model_path=self.depth_model_path,
                    device=video.device.type,
                )

            # Extract depth maps from video
            with torch.no_grad():  # Ensure no gradients flow through depth extraction
                inp_depth_annotator = rearrange(video, "b c t h w -> (b t) c h w")
                inp_depth_annotator = (inp_depth_annotator + 1) * 0.5  # from [-1, 1] to [0, 1]
                depth_video = self.depth_annotator(inp_depth_annotator.float())  # Returns  in [0, 1]
                depth_video = rearrange(depth_video, "(b t) c h w -> b c t h w", b=video.shape[0])

            # Convert depth to proper range for VAE encoding [-1, 1]
            depth_video = 2.0 * depth_video - 1.0

            # For VACE, we need to encode depth with a mask to get 32 channels
            # Create an all-ones mask to indicate the entire frame contains depth information
            mask = torch.ones_like(depth_video)

            # Prepare video latents WITH mask to get proper 32-channel format
            video_latents = self._prepare_video_latents(depth_video, mask=mask)  # Shape: [B, 32, T, H, W]
        else:
            # get masked all zero latents
            if self._all_zero_latent is None:
                condition_latents = condition_latents.to(self.vae.vae.device, self.vae.vae.dtype)
                self._all_zero_latent = self.vae.encode(
                    torch.zeros_like(self.vae.decode(video)).to(dtype=self.vae.vae.dtype, device=self.vae.vae.device),
                )
                # offload depth extractor and vae to cpu -- we don't need it anymore
                del self.depth_annotator
                del self.vae
                torch.cuda.empty_cache()
            inactive_latents = self._all_zero_latent.to(video.device, video.dtype)
            reactive_latents = condition_latents.to(video.device, video.dtype)
            video_latents = torch.cat([inactive_latents, reactive_latents], dim=1)

        # Prepare mask for latent space (always all ones for full frame generation)
        b, _, t, h, w = video_latents.shape
        mask_latents = torch.ones(b, 64, t, h, w, device=video_latents.device, dtype=video_latents.dtype)

        # Concatenate video latents and mask
        conditioning = torch.cat([video_latents, mask_latents], dim=1)  # Shape: [B, 96, T, H, W]

        return conditioning

    def _prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """General method to prepare video latents by encoding with VAE.

        Args:
            video: [B, C, T, H, W] video frames in range [-1, 1]
            mask: [B, C, T, H, W] optional mask tensor
            generator: Random generator for sampling

        Returns:
            latents: [B, C_latent*2, T_latent, H_latent, W_latent] if mask provided,
                        else [B, C_latent, T_latent, H_latent, W_latent]
        """
        if not hasattr(self, "vae") or self.vae is None:
            raise ValueError("VAE must be initialized to prepare video latents")

        # Encode video with VAE
        video = video.to(self.vae.vae.dtype)

        if mask is None:
            # Encode video directly with argmax mode for deterministic results
            latents = self.vae.encode(
                video,
            )
        else:
            # Apply mask to separate inactive/reactive regions
            mask = (mask > 0.5).to(dtype=video.dtype, device=video.device)
            inactive = video * (1 - mask)
            reactive = video * mask

            # Encode separately with argmax mode
            inactive_latents = self.vae.encode(inactive)
            reactive_latents = self.vae.encode(reactive)

            # Concatenate
            latents = torch.cat([inactive_latents, reactive_latents], dim=1)

        return latents

    def sample(
        self,
        noise: torch.Tensor,
        guidance_scale: float = 5.0,
        num_steps: int = 50,
        shift: float = 16.0,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the model."""
        return super().sample(
            noise,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            shift=shift,
            **kwargs,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        unpatchify_features: bool = True,
        **fwd_kwargs,
    ):
        """Forward pass with VACE context support.
        Args:
            x_t (torch.Tensor): The diffused data sample.
            t (torch.Tensor): The current timestep.
            condition (dict[str, torch.Tensor]): dict including: text_embeds, real_raw
             text_embeds: Text embeddings from text encoder
             vid_context: [B, 96, T, H, W] - used to extract depth for conditioning
            r (torch.Tensor): Another timestep mainly used by meanflow.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_logvar: If true, the foward pass returns the logvar.
            fwd_pred_type: Update the network prediction type, must be in ['x0', 'eps', 'v', 'flow'].
                None means using the original net_pred_type.
            skip_layers: Apply skip-layer guidance by skipping layers of the unconditional network during forward pass.
            unpatchify_features: If true, the features will be unpatchified and returned in shape of [B, T, H, W, C].

        Returns:
            torch.Tensor: The score model output.
        """
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "vid_context" in condition, "condition must contain 'vid_context'"

        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # Handle VACE mode
        text_embeds = condition["text_embeds"]
        control_hidden_states = condition["vid_context"]

        # Prepare control_hidden_states_scale
        control_hidden_states_scale = self.context_scale
        if isinstance(control_hidden_states_scale, (int, float)):
            control_hidden_states_scale = [control_hidden_states_scale] * len(self.transformer.config.vace_layers)
        if isinstance(control_hidden_states_scale, list):
            if len(control_hidden_states_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} does not "
                    f"match number of layers {len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = torch.tensor(control_hidden_states_scale)
        if isinstance(control_hidden_states_scale, torch.Tensor):
            if control_hidden_states_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `control_hidden_states_scale` {control_hidden_states_scale.size(0)} does not "
                    f"match number of layers {len(self.transformer.config.vace_layers)}."
                )
            control_hidden_states_scale = control_hidden_states_scale.to(device=x_t.device, dtype=x_t.dtype)

        # Call transformer
        encoder_hidden_states = torch.stack(text_embeds) if isinstance(text_embeds, list) else text_embeds
        model_outputs = self.transformer(
            hidden_states=x_t,
            timestep=self.noise_scheduler.rescale_t(t).to(dtype=x_t.dtype),
            encoder_hidden_states=encoder_hidden_states,
            r_timestep=None if r is None else self.noise_scheduler.rescale_t(r).to(dtype=x_t.dtype),
            encoder_hidden_states_image=None,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_hidden_states_scale,
            attention_kwargs=None,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
            skip_layers=skip_layers,
        )

        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            return self._unpatchify_features(x_t, model_outputs) if unpatchify_features else model_outputs

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
            out[1] = self._unpatchify_features(x_t, out[1]) if unpatchify_features else out[1]

        if return_logvar:
            return out, logvar
        return out
