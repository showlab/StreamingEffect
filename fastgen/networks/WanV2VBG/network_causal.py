# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CausalWanV2VBG: Causal AR network for V2V background distillation.

Used across all 3 distillation stages:
  Stage 1: Causal SFT (teacher-forcing, 50-step DSM)
  Stage 2: ODE Init (trajectory regression, MSE loss)
  Stage 3: Self-Forcing (4-step AR rollout, VSD + GAN / DMD2 loss)

Key design:
  - Interleaved super-chunks: [ref(N) | cond_0 | tgt_0 | cond_1 | tgt_1 | ...] with collaborative RoPE
  - Ref tokens form a "sink" prefix; cond_i + tgt_i form super-chunk i
  - Training: full interleaved sequence with block-wise causal mask per super-chunk
  - AR chunk 0: [ref | cond_0 | tgt_0], AR chunk i>0: [cond_i | tgt_i]
"""

import math
import types
from typing import Optional, List, Set, Union, Tuple, Dict, Any

import torch
from tqdm.auto import tqdm

from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.Wan.network import (
    classify_forward,
    flatten_timestep,
    unflatten_timestep_proj,
)
from fastgen.networks.Wan.network_causal import (
    FLEX_ATTENTION_AVAILABLE,
    _rope_forward_with_time_offset,
    _prepare_blockwise_causal_attn_mask,
    _create_external_caches,
    _wan_set_attn_processor,
    _wan_block_forward_inline_cache,
    classify_forward_block_forward as _causal_classify_forward_block_forward,
    CausalWanAttnProcessor,
)
from fastgen.networks.WanV2VBG.network import WanV2VBG
from fastgen.networks.noise_schedule import NET_PRED_TYPES
import fastgen.utils.logging_utils as logger

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import create_block_mask


def _prepare_v2v_bg_blockwise_causal_attn_mask(
    device: torch.device,
    ref_frames: int,
    num_super_chunks: int,
    chunk_size: int,
    frame_seqlen: int,
):
    """Block-wise causal attention mask for V2VBG with interleaved super-chunks.

    The sequence layout is [ref_tokens | sc_0(cond+tgt) | sc_1(cond+tgt) | ...].
    The mask enforces:
      - All tokens can attend to ref tokens (ref is "sink")
      - Super-chunk i can attend to ref + all super-chunks 0..i (block-wise causal)
      - Ref tokens attend only within ref (bidirectional)
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return None

    ref_tokens = ref_frames * frame_seqlen
    sc_tokens = 2 * chunk_size * frame_seqlen  # cond + target per super-chunk
    total_tokens = ref_tokens + num_super_chunks * sc_tokens

    logger.info(
        f"creating V2VBG blockwise causal attn mask: "
        f"ref_frames={ref_frames}, num_super_chunks={num_super_chunks}, chunk_size={chunk_size}"
    )

    # Right-pad to multiple of 128 for FlexAttention
    padded_length = math.ceil(total_tokens / 128) * 128 - total_tokens

    ends = torch.zeros(total_tokens + padded_length, device=device, dtype=torch.long)

    # Ref tokens: bidirectional within ref
    ends[:ref_tokens] = ref_tokens

    # Super-chunks: block-wise causal
    for i in range(num_super_chunks):
        sc_start = ref_tokens + i * sc_tokens
        sc_end = ref_tokens + (i + 1) * sc_tokens
        ends[sc_start:sc_end] = sc_end  # see ref + all super-chunks 0..i

    def attention_mask(b, h, q_idx, kv_idx) -> torch.Tensor:
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens + padded_length,
        KV_LEN=total_tokens + padded_length,
        _compile=True,  # Must be True: avoids materializing dense (seq_len x seq_len) mask
        device=device,
    )

    return block_mask


def _slice_interleaved_to_target(
    tensors: List[torch.Tensor],
    transformer,
    post_patch_height: int,
    post_patch_width: int,
    target_num_frames: int,
    total_concat_frames: int,
) -> List[torch.Tensor]:
    """Slice interleaved [ref|cond_0|tgt_0|...|cond_N|tgt_N] tensors to target-only tokens."""
    tokens_per_frame = post_patch_height * post_patch_width
    if getattr(transformer, "_v2v_bg_is_interleaved", False):
        num_refs = getattr(transformer, "_v2v_bg_num_refs", 0)
        num_sc = getattr(transformer, "_v2v_bg_num_super_chunks", 1)
        ref_tokens = num_refs * tokens_per_frame
        sc_tokens = (total_concat_frames - num_refs) * tokens_per_frame // num_sc
        cond_chunk_tokens = sc_tokens // 2
        result = []
        for t in tensors:
            parts = []
            for i in range(num_sc):
                tgt_start = ref_tokens + i * sc_tokens + cond_chunk_tokens
                tgt_end = ref_tokens + (i + 1) * sc_tokens
                parts.append(t[:, tgt_start:tgt_end])
            result.append(torch.cat(parts, dim=1))
        return result
    else:
        target_tokens = target_num_frames * tokens_per_frame
        return [t[:, -target_tokens:] for t in tensors]


def _v2v_bg_causal_classify_forward_prepare(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    V2VBG causal classify_forward_prepare.

    Handles [ref|cond|target] concatenation with collaborative RoPE in causal mode.
    For chunk 0, ref+cond are prepended. For chunk i>0, only target tokens are processed
    (prefix KV is in cache).

    Expects attention_kwargs to contain:
        - encoder_condition_states: [B, C, F, H, W] foreground condition latent
        - encoder_ref_states: [B, C, N, H, W] or None
        - ref_frame_indices: List[int]
        - chunk_size, cache_tag, cur_start_frame, total_num_frames, is_ar
    """
    attention_kwargs = attention_kwargs or {}
    encoder_condition_states = attention_kwargs.get("encoder_condition_states", None)
    encoder_ref_states = attention_kwargs.get("encoder_ref_states", None)
    ref_frame_indices = attention_kwargs.get("ref_frame_indices", None)

    # Normalize ref_frame_indices to plain list of ints.
    # After default_collate, JSON [2,12,22] becomes [tensor([2]), tensor([12]), tensor([22])]
    # (list of 1D tensors, inner dim = batch_size). Extract first sample.
    if ref_frame_indices is not None:
        if isinstance(ref_frame_indices, (list, tuple)) and len(ref_frame_indices) > 0:
            first = ref_frame_indices[0]
            if isinstance(first, torch.Tensor):
                ref_frame_indices = [int(x[0]) for x in ref_frame_indices]
        elif isinstance(ref_frame_indices, torch.Tensor):
            if ref_frame_indices.dim() > 1:
                ref_frame_indices = ref_frame_indices[0].tolist()
            else:
                ref_frame_indices = ref_frame_indices.tolist()

    chunk_size = attention_kwargs.get("chunk_size", 3)
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    cur_start_frame = attention_kwargs.get("cur_start_frame", 0)
    total_num_frames = attention_kwargs.get("total_num_frames", 13)
    store_kv = attention_kwargs.get("store_kv", False)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tokens_per_frame = post_patch_height * post_patch_width

    # Determine if this is the first chunk (needs prefix concat)
    is_first_chunk = (cur_start_frame == 0)

    # Check if we already have cached prefix tokens
    has_cached_prefix = False
    if not is_first_chunk:
        kv_container = getattr(self, "_external_self_kv_list", None)
        if isinstance(kv_container, list) and len(kv_container) > 0:
            for per_block in kv_container:
                if isinstance(per_block, dict):
                    tag_entry = per_block.get(cache_tag, None)
                    if isinstance(tag_entry, dict) and int(tag_entry.get("len", 0)) > 0:
                        has_cached_prefix = True
                        break

    is_ar = attention_kwargs.get("is_ar", False)

    if is_first_chunk and encoder_condition_states is not None:
        if not is_ar:
            # === Path A: Training — interleaved super-chunks ===
            # Layout: [ref | cond_0 | tgt_0 | cond_1 | tgt_1 | ... | cond_N | tgt_N]
            num_super_chunks = total_num_frames // chunk_size

            # Full RoPE for all total_num_frames
            rotary_emb_full = self.rope(encoder_condition_states, start_frame=0)

            # Build interleaved hidden_states
            parts = []
            if encoder_ref_states is not None:
                parts.append(encoder_ref_states)
            for i in range(num_super_chunks):
                cs_start = i * chunk_size * p_t
                cs_end = (i + 1) * chunk_size * p_t
                parts.append(encoder_condition_states[:, :, cs_start:cs_end])
                parts.append(hidden_states[:, :, cs_start:cs_end])
            hidden_states = torch.cat(parts, dim=2)

            # Build interleaved RoPE
            rope_cos, rope_sin = [], []
            if encoder_ref_states is not None and ref_frame_indices is not None:
                num_refs = encoder_ref_states.shape[2] // p_t
                for fidx in ref_frame_indices:
                    start = fidx * tokens_per_frame
                    end = (fidx + 1) * tokens_per_frame
                    rope_cos.append(rotary_emb_full[0][:, start:end, :])
                    rope_sin.append(rotary_emb_full[1][:, start:end, :])
            else:
                num_refs = 0

            # Super-chunks: cond_i + tgt_i share same RoPE positions
            for i in range(num_super_chunks):
                s = i * chunk_size * tokens_per_frame
                e = (i + 1) * chunk_size * tokens_per_frame
                chunk_cos = rotary_emb_full[0][:, s:e, :]
                chunk_sin = rotary_emb_full[1][:, s:e, :]
                rope_cos.extend([chunk_cos, chunk_cos])
                rope_sin.extend([chunk_sin, chunk_sin])

            rotary_emb = (torch.cat(rope_cos, dim=1), torch.cat(rope_sin, dim=1))

            # Store state
            target_num_frames = num_frames // p_t
            total_concat_frames = hidden_states.shape[2] // p_t
            self._v2v_bg_target_num_frames = target_num_frames
            self._v2v_bg_total_concat_frames = total_concat_frames
            self._v2v_bg_num_refs = num_refs
            self._v2v_bg_is_interleaved = True
            self._v2v_bg_num_super_chunks = num_super_chunks
            attention_kwargs["frame_seqlen"] = tokens_per_frame

        else:
            # === Path B: AR chunk 0 — [ref | cond_0 | target_0] ===
            # encoder_condition_states already sliced to chunk_size by forward()

            # Full RoPE table for ref collaborative positions
            dummy = torch.empty(
                1, 1, total_num_frames * p_t, height, width,
                device=hidden_states.device,
            )
            rotary_emb_full = self.rope(dummy, start_frame=0)

            # Chunk 0 RoPE (for cond_0 and target_0)
            chunk_rope = self.rope(hidden_states, start_frame=0)

            parts = []
            rope_cos, rope_sin = [], []

            if encoder_ref_states is not None and ref_frame_indices is not None:
                parts.append(encoder_ref_states)
                num_refs = encoder_ref_states.shape[2] // p_t
                for fidx in ref_frame_indices:
                    start = fidx * tokens_per_frame
                    end = (fidx + 1) * tokens_per_frame
                    rope_cos.append(rotary_emb_full[0][:, start:end, :])
                    rope_sin.append(rotary_emb_full[1][:, start:end, :])
            else:
                num_refs = 0

            parts.append(encoder_condition_states)
            parts.append(hidden_states)
            hidden_states = torch.cat(parts, dim=2)

            # cond_0 + target_0 share same RoPE
            rope_cos.extend([chunk_rope[0], chunk_rope[0]])
            rope_sin.extend([chunk_rope[1], chunk_rope[1]])
            rotary_emb = (torch.cat(rope_cos, dim=1), torch.cat(rope_sin, dim=1))

            # Store state
            target_num_frames = num_frames // p_t
            total_concat_frames = hidden_states.shape[2] // p_t
            self._v2v_bg_target_num_frames = target_num_frames
            self._v2v_bg_total_concat_frames = total_concat_frames
            self._v2v_bg_num_refs = num_refs
            self._v2v_bg_is_interleaved = False
            attention_kwargs["frame_seqlen"] = tokens_per_frame

    elif not is_first_chunk and encoder_condition_states is not None:
        # === Path C: AR chunk i>0 — [cond_i | target_i] ===
        # encoder_condition_states already sliced to chunk_size by forward()
        hidden_states_orig = hidden_states
        hidden_states = torch.cat([encoder_condition_states, hidden_states], dim=2)

        # cond_i + target_i share same RoPE at offset
        chunk_rope = self.rope(hidden_states_orig, start_frame=cur_start_frame)
        rotary_emb = (
            chunk_rope[0].repeat(1, 2, 1, 1),
            chunk_rope[1].repeat(1, 2, 1, 1),
        )

        # Compute cache_start_tokens for KV cache positioning
        num_refs = getattr(self, "_v2v_bg_num_refs", 0)
        ref_tokens = num_refs * tokens_per_frame
        chunk_idx = cur_start_frame // chunk_size
        super_chunk_tokens = 2 * chunk_size * tokens_per_frame
        attention_kwargs["cache_start_tokens"] = ref_tokens + chunk_idx * super_chunk_tokens

        target_num_frames = num_frames // p_t
        self._v2v_bg_target_num_frames = target_num_frames
        self._v2v_bg_total_concat_frames = hidden_states.shape[2] // p_t
        self._v2v_bg_is_interleaved = False
        attention_kwargs["frame_seqlen"] = tokens_per_frame

    else:
        # No condition: just target (legacy/fallback path)
        rotary_emb = self.rope(hidden_states, start_frame=cur_start_frame)
        target_num_frames = num_frames // p_t
        self._v2v_bg_target_num_frames = target_num_frames
        self._v2v_bg_total_concat_frames = target_num_frames
        self._v2v_bg_is_interleaved = False
        attention_kwargs["frame_seqlen"] = tokens_per_frame

    # Patch embedding
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # Timestep processing
    timestep, ts_seq_len = flatten_timestep(timestep)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = unflatten_timestep_proj(timestep_proj, ts_seq_len)

    # r_timestep handling
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

    # Construct block mask for teacher-forcing (full sequence training)
    # V2VBG needs a prefix-aware mask: prefix tokens (ref+cond) are always visible,
    # target tokens use block-wise causal masking within themselves.
    total_target_frames = total_num_frames
    actual_num_frames = hidden_states.shape[1] // tokens_per_frame if tokens_per_frame > 0 else 1
    is_ar = attention_kwargs.get("is_ar", False)
    if getattr(self, "block_mask", None) is None and is_first_chunk and not is_ar:
        if FLEX_ATTENTION_AVAILABLE:
            num_refs = getattr(self, "_v2v_bg_num_refs", 0)
            num_sc = getattr(self, "_v2v_bg_num_super_chunks", total_num_frames // chunk_size)
            if num_refs > 0 or num_sc > 0:
                self.block_mask = _prepare_v2v_bg_blockwise_causal_attn_mask(
                    hidden_states.device,
                    ref_frames=num_refs,
                    num_super_chunks=num_sc,
                    chunk_size=chunk_size,
                    frame_seqlen=tokens_per_frame,
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    hidden_states.device,
                    num_frames=actual_num_frames,
                    frame_seqlen=tokens_per_frame,
                    chunk_size=chunk_size,
                )

    # Create external KV caches for AR mode
    # For V2VBG interleaved: total capacity = num_refs + total_num_frames * 2 (cond + target)
    if is_first_chunk and encoder_condition_states is not None:
        num_refs = getattr(self, "_v2v_bg_num_refs", 0)
        total_capacity_frames = num_refs + total_num_frames * 2
    else:
        total_capacity_frames = total_target_frames
        if has_cached_prefix:
            kv_container = self._external_self_kv_list
            if len(kv_container) > 0:
                tag_entry = kv_container[0].get(cache_tag, {})
                k_buf = tag_entry.get("k", None)
                if k_buf is not None:
                    total_capacity_frames = k_buf.shape[1] // tokens_per_frame

    if actual_num_frames < total_capacity_frames:
        self._create_external_caches(
            hidden_states,
            encoder_hidden_states,
            num_frames=actual_num_frames,
            total_num_frames=total_capacity_frames,
            cache_tag=cache_tag,
            cur_start_frame=0 if is_first_chunk else cur_start_frame,
        )

    return (
        hidden_states,
        timestep_proj,
        r_timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        temb,
        rotary_emb,
    )


def _v2v_bg_causal_classify_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    r_timestep: Optional[torch.LongTensor] = None,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_features_early: Optional[bool] = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: Optional[bool] = False,
    skip_layers: Optional[List[int]] = None,
    **kwargs,
):
    """V2VBG causal classify_forward with output slicing for [ref|cond|target] concat."""
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

    # Output norm + projection
    target_num_frames = getattr(self, "_v2v_bg_target_num_frames", post_patch_num_frames)
    total_concat_frames = getattr(self, "_v2v_bg_total_concat_frames", post_patch_num_frames)

    if return_features_early:
        assert len(features) == len(feature_indices)
        # Slice features to target-only tokens (same logic as hidden_states below)
        if total_concat_frames > target_num_frames:
            features = _slice_interleaved_to_target(
                features, self, post_patch_height, post_patch_width,
                target_num_frames, total_concat_frames,
            )
        return features

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

    # Slice to only return target frames
    if total_concat_frames > target_num_frames:
        tokens_per_frame = post_patch_height * post_patch_width
        if getattr(self, "_v2v_bg_is_interleaved", False):
            # Training: target tokens are scattered — second half of each super-chunk
            num_refs = getattr(self, "_v2v_bg_num_refs", 0)
            num_sc = getattr(self, "_v2v_bg_num_super_chunks", 1)
            ref_tokens = num_refs * tokens_per_frame
            sc_tokens = (total_concat_frames - num_refs) * tokens_per_frame // num_sc
            cond_chunk_tokens = sc_tokens // 2
            target_parts = []
            for i in range(num_sc):
                tgt_start = ref_tokens + i * sc_tokens + cond_chunk_tokens
                tgt_end = ref_tokens + (i + 1) * sc_tokens
                target_parts.append(hidden_states[:, tgt_start:tgt_end])
            hidden_states = torch.cat(target_parts, dim=1)
        else:
            # AR mode: target is at the end
            target_tokens = target_num_frames * tokens_per_frame
            hidden_states = hidden_states[:, -target_tokens:]

    # Unpatchify
    hidden_states = hidden_states.reshape(
        batch_size, target_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
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


class CausalWanV2VBG(CausalFastGenNetwork, WanV2VBG):
    """Causal AR network for V2V background distillation.

    Used by all 3 distillation stages. Handles:
    - [ref|cond|target] temporal concat with collaborative RoPE
    - Prefix KV caching (ref+cond as sink tokens)
    - Chunk-by-chunk AR generation
    - Feature extraction for discriminator
    """

    def __init__(
        self,
        model_id_or_local_path: str = WanV2VBG.MODEL_ID_VER_2_2_TI2V_5B_720P,
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
        use_fsdp_checkpoint: bool = True,
        chunk_size: int = 3,
        total_num_frames: int = 13,
        delete_cache_on_clear: bool = False,
        ref_dropout_prob: float = 0.0,
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
            use_fsdp_checkpoint=use_fsdp_checkpoint,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **model_kwargs,
        )
        self._delete_cache_on_clear = delete_cache_on_clear
        self.ref_dropout_prob = ref_dropout_prob

    def override_transformer_forward(self, inner_dim: int) -> None:
        """Install causal attention processor and V2VBG-specific forward logic."""
        # Patch rope for time offset
        if hasattr(self.transformer, "rope"):
            self.transformer.rope.forward = types.MethodType(
                _rope_forward_with_time_offset, self.transformer.rope
            )

        # Patch helper methods
        self.transformer._prepare_blockwise_causal_attn_mask = types.MethodType(
            _prepare_blockwise_causal_attn_mask, self.transformer
        )
        self.transformer._create_external_caches = types.MethodType(
            _create_external_caches, self.transformer
        )
        self.transformer.set_attn_processor = types.MethodType(
            _wan_set_attn_processor, self.transformer
        )

        # V2VBG-specific prepare and forward
        self.transformer.classify_forward_prepare = types.MethodType(
            _v2v_bg_causal_classify_forward_prepare, self.transformer
        )
        self.transformer.classify_forward_block_forward = types.MethodType(
            _causal_classify_forward_block_forward, self.transformer
        )
        self.transformer.forward = types.MethodType(
            _v2v_bg_causal_classify_forward, self.transformer
        )

        # Patch block forward to inline-cache path
        for block in self.transformer.blocks:
            block.forward = types.MethodType(_wan_block_forward_inline_cache, block)

        # Install causal attention processor
        self.transformer.set_attn_processor(CausalWanAttnProcessor())
        self.transformer.block_mask = None

    def clear_caches(self) -> None:
        """Clear KV caches."""
        if self._delete_cache_on_clear:
            raw = self.transformer
            raw.__dict__.pop("_external_self_kv_list", None)
            raw.__dict__.pop("_external_cross_kv_list", None)
            torch.cuda.empty_cache()
            return

        if hasattr(self.transformer, "_external_self_kv_list"):
            for kvb in self.transformer._external_self_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        if hasattr(self.transformer, "_external_cross_kv_list"):
            for kvb in self.transformer._external_cross_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])

    def _apply_ref_dropout(self, condition: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply per-ref independent dropout during training (Stage 1 only)."""
        if self.ref_dropout_prob <= 0 or not self.training:
            return condition

        ref_latents = condition.get("ref_latents", None)
        ref_mask = condition.get("ref_mask", None)
        if ref_latents is None or ref_mask is None:
            return condition

        condition = dict(condition)
        ref_latents = ref_latents.clone()
        ref_mask = ref_mask.clone()

        num_refs = ref_mask.shape[1]
        for i in range(num_refs):
            if ref_mask[0, i].item() > 0:  # only dropout real refs
                if torch.rand(1).item() < self.ref_dropout_prob:
                    ref_latents[:, :, i] = 0
                    ref_mask[:, i] = 0

        condition["ref_latents"] = ref_latents
        condition["ref_mask"] = ref_mask
        return condition

    def _build_v2v_bg_timestep(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        cur_start_frame: int = 0,
        is_ar: bool = False,
    ) -> torch.Tensor:
        """Build per-frame timestep for V2VBG with interleaved super-chunk layout.

        Training: [ref_0 | cond_0_0 | tgt_0_ts | cond_1_0 | tgt_1_ts | ...]
        AR chunk 0: [ref_0 | cond_0_0 | tgt_0_ts]
        AR chunk i>0: [cond_i_0 | tgt_i_ts]
        """
        bsz, _, num_frames, H, W = x_t.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        ppf = num_frames // p_t  # post-patch target frames in this chunk

        t_rescaled = self.noise_scheduler.rescale_t(t)

        if t_rescaled.ndim == 1:
            target_ts = t_rescaled.view(bsz, 1).expand(bsz, ppf)
        else:
            if t_rescaled.shape[1] > ppf:
                target_ts = t_rescaled[:, cur_start_frame:cur_start_frame + ppf]
            else:
                target_ts = t_rescaled

        target_ts = target_ts.to(dtype=x_t.dtype)

        ref_latents = condition.get("ref_latents", None)
        ref_mask = condition.get("ref_mask", None)

        if cur_start_frame == 0 and not is_ar:
            # Training: interleaved timestep
            ts_parts = []
            if ref_latents is not None and ref_mask is not None:
                num_refs = ref_latents.shape[2]
                ts_parts.append(torch.zeros(bsz, num_refs, device=x_t.device, dtype=x_t.dtype))

            cs = self.chunk_size
            num_super_chunks = ppf // cs
            for i in range(num_super_chunks):
                ts_parts.append(torch.zeros(bsz, cs, device=x_t.device, dtype=x_t.dtype))
                ts_parts.append(target_ts[:, i * cs:(i + 1) * cs])
            timestep = torch.cat(ts_parts, dim=1)

        elif cur_start_frame == 0 and is_ar:
            # AR chunk 0: [ref_zeros | cond_0_zeros | target_0_ts]
            ts_parts = []
            if ref_latents is not None and ref_mask is not None:
                num_refs = ref_latents.shape[2]
                ts_parts.append(torch.zeros(bsz, num_refs, device=x_t.device, dtype=x_t.dtype))
            ts_parts.append(torch.zeros(bsz, ppf, device=x_t.device, dtype=x_t.dtype))
            ts_parts.append(target_ts)
            timestep = torch.cat(ts_parts, dim=1)

        else:
            # AR chunk i>0: [cond_i_zeros | target_i_ts]
            ts_parts = [
                torch.zeros(bsz, ppf, device=x_t.device, dtype=x_t.dtype),
                target_ts,
            ]
            timestep = torch.cat(ts_parts, dim=1)

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
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
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

        # Apply ref dropout (only in training mode for Stage 1)
        condition = self._apply_ref_dropout(condition)

        text_embeds = condition["text_embeds"]
        foreground_latent = condition["foreground_latent"]
        ref_latents = condition.get("ref_latents", None)
        ref_frame_indices = condition.get("ref_frame_indices", None)
        ref_mask = condition.get("ref_mask", None)

        text_embeds = torch.stack(text_embeds, dim=0) if isinstance(text_embeds, list) else text_embeds

        # Build timestep
        timestep = self._build_v2v_bg_timestep(x_t, t, condition, cur_start_frame, is_ar=is_ar)

        # Build attention_kwargs
        attention_kwargs = {
            "cache_tag": cache_tag,
            "chunk_size": self.chunk_size,
            "store_kv": store_kv,
            "cur_start_frame": cur_start_frame,
            "total_num_frames": self.total_num_frames,
            "is_ar": is_ar,
        }

        # Always pass cond (full for training, sliced for AR)
        if is_ar:
            p_t = self.transformer.config.patch_size[0]
            chunk_idx = cur_start_frame // self.chunk_size
            cond_start = chunk_idx * self.chunk_size * p_t
            cond_end = (chunk_idx + 1) * self.chunk_size * p_t
            attention_kwargs["encoder_condition_states"] = foreground_latent[:, :, cond_start:cond_end]
        else:
            # Training: pass full foreground for interleaving in prepare()
            attention_kwargs["encoder_condition_states"] = foreground_latent

        # Ref only on first chunk (ref is a prefix, cached after chunk 0)
        if cur_start_frame == 0:
            attention_kwargs["encoder_ref_states"] = ref_latents
            attention_kwargs["ref_frame_indices"] = ref_frame_indices
        else:
            attention_kwargs["encoder_ref_states"] = None
            attention_kwargs["ref_frame_indices"] = None

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

    def preserve_conditioning(self, x: torch.Tensor, condition: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """No first-frame preservation needed for V2VBG (unlike I2V)."""
        return x

    def sample(
        self,
        noise: torch.FloatTensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        sample_steps: Optional[int] = 50,
        shift: float = 5.0,
        context_noise: float = 0,
        solver: str = "unipc",
        sample_type: str = "ode",
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive sampling with CFG for V2VBG.

        Args:
            solver: "unipc" for teacher/SFT multi-step ODE, "euler" for distilled few-step x0-prediction.
            sample_type: For euler solver only — "ode" (deterministic) or "sde" (stochastic).
        """
        assert self.schedule_type == "rf"
        self.clear_caches()

        batch_size = noise.shape[0]
        num_frames = noise.shape[2]

        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size
        time_rescale_factor = self.unipc_scheduler.config.num_train_timesteps

        self.unipc_scheduler.config.flow_shift = shift
        self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
        timesteps = self.unipc_scheduler.timesteps
        t_init = timesteps[0] / time_rescale_factor
        x = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        if solver == "euler":
            # Build t_list: scheduler timesteps scaled to [0, 1] + terminal 0.0
            t_list = (timesteps / time_rescale_factor).to(dtype=x.dtype, device=x.device)
            t_list = torch.cat([t_list, t_list.new_zeros(1)])  # append t=0
            return self._sample_euler(x, t_list, condition, neg_condition,
                                      guidance_scale, context_noise, sample_type)

        return self._sample_unipc(x, timesteps, time_rescale_factor, sample_steps,
                                  condition, neg_condition, guidance_scale, context_noise)

    def _sample_unipc(
        self, x, timesteps, time_rescale_factor, sample_steps,
        condition, neg_condition, guidance_scale, context_noise,
    ) -> torch.Tensor:
        """Multi-step UniPC ODE sampling (for teacher / SFT models)."""
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]
            self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=x.device)

            for timestep in tqdm(timesteps, total=sample_steps, desc=f"chunk {i}"):
                t = (timestep / time_rescale_factor).expand(batch_size)
                x_cur = x_next
                flow_pred = self(
                    x_cur, t, condition,
                    cache_tag="pos", cur_start_frame=start, store_kv=False, is_ar=True,
                )
                if guidance_scale is not None:
                    flow_uncond = self(
                        x_cur, t, neg_condition,
                        cache_tag="neg", cur_start_frame=start, store_kv=False, is_ar=True,
                    )
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
                x_next = self.unipc_scheduler.step(flow_pred, timestep, x_next, return_dict=False)[0]

            x[:, :, start:end] = x_next

            # Store KV for this chunk
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=x.dtype)
            if context_noise > 0:
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(x_cache, t_cache, condition, cache_tag="pos", cur_start_frame=start, store_kv=True, is_ar=True)
            if guidance_scale is not None:
                _ = self(x_cache, t_cache, neg_condition, cache_tag="neg", cur_start_frame=start, store_kv=True, is_ar=True)

        self.clear_caches()
        return x

    def _sample_euler(
        self, x, t_list, condition, neg_condition,
        guidance_scale, context_noise, sample_type,
    ) -> torch.Tensor:
        """Few-step Euler x0-prediction sampling (for distilled models).

        Matches CausVidModel._student_sample_loop but with CFG support.
        """
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]

            for step in tqdm(range(len(t_list) - 1), desc=f"chunk {i}"):
                t_cur = t_list[step].expand(batch_size)
                x_cur = x_next

                # Predict x0 with CFG
                x0_pred = self(
                    x_cur, t_cur, condition,
                    fwd_pred_type="x0",
                    cache_tag="pos", cur_start_frame=start, store_kv=False, is_ar=True,
                )
                if guidance_scale is not None:
                    x0_uncond = self(
                        x_cur, t_cur, neg_condition,
                        fwd_pred_type="x0",
                        cache_tag="neg", cur_start_frame=start, store_kv=False, is_ar=True,
                    )
                    x0_pred = x0_uncond + guidance_scale * (x0_pred - x0_uncond)

                # Forward process to next timestep (skip if last step → t=0)
                t_next = t_list[step + 1]
                if t_next > 0:
                    t_next_batch = t_next.expand(batch_size)
                    if sample_type == "sde":
                        eps = torch.randn_like(x0_pred)
                    else:  # ode
                        eps = self.noise_scheduler.x0_to_eps(xt=x_cur, x0=x0_pred, t=t_cur)
                    x_next = self.noise_scheduler.forward_process(x0_pred, eps, t_next_batch)
                else:
                    x_next = x0_pred

            x[:, :, start:end] = x_next

            # Store KV for this chunk (clean denoised frames)
            x_cache = x_next
            t_cache = t_list[-1].expand(batch_size)
            if context_noise > 0:
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(x_cache, t_cache, condition, fwd_pred_type="x0",
                     cache_tag="pos", cur_start_frame=start, store_kv=True, is_ar=True)
            if guidance_scale is not None:
                _ = self(x_cache, t_cache, neg_condition, fwd_pred_type="x0",
                         cache_tag="neg", cur_start_frame=start, store_kv=True, is_ar=True)

        self.clear_caches()
        return x
