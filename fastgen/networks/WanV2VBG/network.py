# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
WanV2VBG: Bidirectional Wan wrapper for V2V background replacement.

This is an OPTIONAL verification-only wrapper. The primary network for all 3 distillation
stages is CausalWanV2VBG in network_causal.py.

Replicates the V2V_background CustomWanTransformer3DModel forward logic:
  - Temporal concat: [ref(N) | cond(F) | target(F)] along frame dim
  - Collaborative RoPE: each ref copies target's RoPE at ref_frame_indices[i]
  - Output slice: only return target frames from the output
"""

import types
from typing import Optional, List, Set, Union, Tuple, Dict

import torch

from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.Wan.network import (
    classify_forward,
    classify_forward_prepare as _base_classify_forward_prepare,
    classify_forward_block_forward,
    flatten_timestep,
    unflatten_timestep_proj,
)
from fastgen.networks.noise_schedule import NET_PRED_TYPES
import fastgen.utils.logging_utils as logger


def v2v_bg_classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
):
    """
    V2VBG-specific classify_forward_prepare that handles [ref|cond|target] concat
    and collaborative RoPE before patch embedding.

    The attention_kwargs must contain:
        - encoder_condition_states: [B, C, F, H, W] foreground condition latent
        - encoder_ref_states: [B, C, N, H, W] or None (ref images, N=0..3)
        - ref_frame_indices: List[int] latent-space frame indices for collaborative RoPE
        - ref_mask: [B, N] bool mask for valid refs (optional, for dropout)
    """
    attention_kwargs = attention_kwargs or {}
    encoder_condition_states = attention_kwargs.get("encoder_condition_states", None)
    encoder_ref_states = attention_kwargs.get("encoder_ref_states", None)
    ref_frame_indices = attention_kwargs.get("ref_frame_indices", None)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tokens_per_frame = post_patch_height * post_patch_width

    # Ensure RoPE buffers are on the same device as input
    if self.rope.freqs_cos.device != hidden_states.device:
        self.rope.freqs_cos = self.rope.freqs_cos.to(hidden_states.device)
        self.rope.freqs_sin = self.rope.freqs_sin.to(hidden_states.device)

    if encoder_ref_states is not None and ref_frame_indices is not None:
        num_refs = encoder_ref_states.shape[2]

        # Collaborative RoPE: compute RoPE for target, then copy for refs
        rotary_emb_hidden = self.rope(hidden_states)  # RoPE for target frames

        # Each ref copies the RoPE of its corresponding target frame
        # RoPE shape: [batch, seq_len, 1, head_dim] — concat on dim=1 (seq)
        ref_rotary_embs_cos = []
        ref_rotary_embs_sin = []
        # Normalize ref_frame_indices to a flat list of per-ref ints.
        # Accepts: Python list of ints, tensor (N,) / (B, N), or list of per-ref
        # tensors (PyTorch default_collate transposes list-valued fields, so a
        # batch of [[24], [24]] becomes [tensor([24, 24])]).
        if isinstance(ref_frame_indices, list):
            if len(ref_frame_indices) > 0 and torch.is_tensor(ref_frame_indices[0]):
                ref_frame_indices = [int(t.flatten()[0].item()) for t in ref_frame_indices]
            else:
                ref_frame_indices = [int(x) for x in ref_frame_indices]
        else:  # tensor
            if ref_frame_indices.dim() > 1:
                ref_frame_indices = ref_frame_indices[0]
            ref_frame_indices = [int(x.item()) for x in ref_frame_indices]

        for frame_idx in ref_frame_indices:
            start = frame_idx * tokens_per_frame
            end = (frame_idx + 1) * tokens_per_frame
            ref_rotary_embs_cos.append(rotary_emb_hidden[0][:, start:end, :, :])
            ref_rotary_embs_sin.append(rotary_emb_hidden[1][:, start:end, :, :])

        # Final RoPE: [refs | cond | target]
        rotary_emb = (
            torch.cat([
                torch.cat(ref_rotary_embs_cos, dim=1),
                rotary_emb_hidden[0],  # cond reuses target's RoPE
                rotary_emb_hidden[0],  # target's RoPE
            ], dim=1),
            torch.cat([
                torch.cat(ref_rotary_embs_sin, dim=1),
                rotary_emb_hidden[1],
                rotary_emb_hidden[1],
            ], dim=1),
        )

        # Concat: [ref | cond | target] along frame dim
        hidden_states = torch.cat([encoder_ref_states, encoder_condition_states, hidden_states], dim=2)
        fake_nums = 2 * post_patch_num_frames + num_refs
    elif encoder_condition_states is not None:
        # No refs, just cond + target
        rotary_emb_cond = self.rope(encoder_condition_states)
        rotary_emb_target = self.rope(hidden_states)
        rotary_emb = tuple(
            torch.cat([emb_c, emb_t], dim=1)
            for emb_c, emb_t in zip(rotary_emb_cond, rotary_emb_target)
        )
        hidden_states = torch.cat([encoder_condition_states, hidden_states], dim=2)
        fake_nums = 2 * post_patch_num_frames
    else:
        rotary_emb = self.rope(hidden_states)
        fake_nums = post_patch_num_frames

    # Store for output slicing
    self._v2v_bg_fake_nums = fake_nums
    self._v2v_bg_post_patch_num_frames = post_patch_num_frames

    # Patch embedding
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # Timestep processing
    timestep, ts_seq_len = flatten_timestep(timestep)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    # r_timestep handling (for meanflow-like models)
    r_timestep_proj = None
    if self.r_embedder is not None and r_timestep is not None:
        if self.time_cond_type == "abs":
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

        if self.encoder_depth is None:
            timestep_proj = timestep_proj + r_timestep_proj
            temb = temb + remb
        else:
            temb = remb

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        temb,
        rotary_emb,
    )


def v2v_bg_classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
):
    """V2VBG classify_forward that handles [ref|cond|target] concat and output slicing."""
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        temb,
        rotary_emb,
    ) = self.classify_forward_prepare(
        hidden_states,
        timestep,
        encoder_hidden_states,
        r_timestep,
        encoder_hidden_states_image,
        attention_kwargs,
    )

    hidden_states, features = self.classify_forward_block_forward(
        hidden_states,
        timestep_proj,
        encoder_hidden_states,
        rotary_emb,
        r_timestep_proj,
        skip_layers,
        feature_indices,
        return_features_early,
        lora_scale,
        attention_kwargs,
    )

    # Slice features to target-only tokens (matching the output slice for hidden_states).
    # Intermediate features contain tokens for the full [ref|cond|target] sequence,
    # but downstream consumers (discriminator, _unpatchify_features) expect target-only.
    fake_nums = getattr(self, "_v2v_bg_fake_nums", post_patch_num_frames)
    if fake_nums > post_patch_num_frames and len(features) > 0:
        target_tokens = post_patch_num_frames * post_patch_height * post_patch_width
        features = [feat[:, -target_tokens:, :] for feat in features]

    if return_features_early:
        assert len(features) == len(feature_indices)
        return features

    # Output norm + projection
    if temb.dim() == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
        if shift.shape[1] == hidden_states.shape[1]:
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        else:
            frame_seqlen = post_patch_height * post_patch_width
            hs_norm_out = self.norm_out(hidden_states.float()).unflatten(1, (-1, frame_seqlen))
            hidden_states = (
                (hs_norm_out * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).flatten(1, 2).type_as(hidden_states)
            )
    else:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    # Slice to only return target frames (like V2V_background's CustomWanTransformer3DModel)
    fake_nums = getattr(self, "_v2v_bg_fake_nums", post_patch_num_frames)
    if fake_nums > post_patch_num_frames:
        hidden_states = hidden_states[:, -hidden_states.shape[-2] // fake_nums * post_patch_num_frames:]

    # Unpatchify
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features]

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar
    return out


class WanV2VBG(WanI2V):
    """Bidirectional Wan wrapper for V2V background replacement (verification only).

    Uses the same Wan2.2 TI2V-5B base model but replaces the forward logic to
    handle [ref|cond|target] temporal concatenation with collaborative RoPE,
    matching V2V_background's CustomWanTransformer3DModel.
    """

    def __init__(
        self,
        model_id_or_local_path: str = WanI2V.MODEL_ID_VER_2_2_TI2V_5B_720P,
        r_timestep: bool = False,
        disable_efficient_attn: bool = False,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = False,
        r_embedder_init: str = "pretrained",
        time_cond_type: str = "diff",
        norm_temb: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        encoder_depth: int | None = None,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
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
            **model_kwargs,
        )
        self.is_v2v_bg = True
        # Disable I2V-specific first_frame_cond logic
        self.concat_mask = False
        self.use_image_encoder = False

    def override_transformer_forward(self, inner_dim: int) -> None:
        """Override the transformer forward to use V2VBG [ref|cond|target] logic."""
        from fastgen.networks.Wan.network import block_forward

        # Patch each block's forward to accept norm_temb (same as Wan base)
        for block in self.transformer.blocks:
            block.forward = types.MethodType(block_forward, block)

        self.transformer.classify_forward_prepare = types.MethodType(
            v2v_bg_classify_forward_prepare, self.transformer
        )
        self.transformer.classify_forward_block_forward = types.MethodType(
            classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(
            v2v_bg_classify_forward, self.transformer
        )

    def _compute_timestep_inputs(self, timestep: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute expanded timestep for V2VBG.

        For V2VBG, timestep is expanded to [B, T] where T is per-token.
        Ref and cond tokens get timestep=0, target tokens get the actual timestep.
        """
        timestep = self.noise_scheduler.rescale_t(timestep)
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        if mask is not None:
            p_t, _, _ = self.transformer.config.patch_size
            timestep = mask[:, ::p_t, 0, 0] * timestep
        return timestep

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
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "foreground_latent" in condition, "condition must contain 'foreground_latent'"

        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES

        text_embeds = condition["text_embeds"]
        foreground_latent = condition["foreground_latent"]
        ref_latents = condition.get("ref_latents", None)
        ref_frame_indices = condition.get("ref_frame_indices", None)
        ref_mask = condition.get("ref_mask", None)

        text_embeds = torch.stack(text_embeds, dim=0) if isinstance(text_embeds, list) else text_embeds

        # Build expanded timestep: [ref(0) | cond(0) | target(t)]
        bsz, _, num_frames, H, W = x_t.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        ppf = num_frames // p_t
        tokens_per_frame = (H // p_h) * (W // p_w)
        t_rescaled = self.noise_scheduler.rescale_t(t).view(bsz, 1)

        # Target timestep: per-token
        target_ts = t_rescaled.expand(bsz, ppf * tokens_per_frame)

        # Build full timestep with zeros for ref + cond
        ts_parts = []
        if ref_latents is not None and ref_mask is not None:
            num_refs = ref_latents.shape[2]
            ts_parts.append(torch.zeros(bsz, num_refs * tokens_per_frame, device=x_t.device, dtype=x_t.dtype))
        ts_parts.append(torch.zeros(bsz, ppf * tokens_per_frame, device=x_t.device, dtype=x_t.dtype))
        ts_parts.append(target_ts.to(dtype=x_t.dtype))
        timestep = torch.cat(ts_parts, dim=1)

        # Build attention_kwargs with V2VBG-specific data
        attention_kwargs = {
            "encoder_condition_states": foreground_latent,
            "encoder_ref_states": ref_latents,
            "ref_frame_indices": ref_frame_indices,
        }

        model_outputs = self.transformer(
            hidden_states=x_t,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            r_timestep=r,
            attention_kwargs=attention_kwargs,
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
