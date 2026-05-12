# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist
import torch.utils.checkpoint

from fastgen.methods import CausVidModel

import fastgen.utils.logging_utils as logger
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.utils.basic_utils import convert_cfg_to_dict
from fastgen.utils.distributed import is_rank0, world_size

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


def _chunked_pointwise(fn, *tensors, chunk_size, dim=1):
    """Apply a pointwise function to chunked tensors along dim, then concatenate."""
    split_tensors = [t.split(chunk_size, dim=dim) for t in tensors]
    return torch.cat([fn(*chunk_group) for chunk_group in zip(*split_tensors)], dim=dim)


def _mem_efficient_block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    norm_temb: bool,
    _chunk_size: int = 25000,
):
    """Memory-efficient block_forward for long-sequence bidirectional models.

    The bidirectional teacher/fake_score processes ~104,940 tokens (53 frames at 1056×1920).
    The original block_forward has these peak allocations:
      - temb.float(): 7.21 GiB  (104,940 × 6 × 3072 × 4B)
      - hidden_states.float(): 1.22 GiB each (×5 occurrences)
      - FFN GELU intermediate: 2.45 GiB

    This version:
      1. Computes AdaLN modulation in bf16 (saves 7.21 GiB by NOT casting temb to float32)
      2. Chunks all pointwise norm/modulate/residual ops along token dim (caps float32 temps)
      3. Keeps attention full-sequence (flash attention is already memory-efficient)
      4. FFN is chunked at module level (installed separately)
    """
    from fastgen.networks.Wan.network import normalize

    compute_dtype = hidden_states.dtype
    cs = _chunk_size

    # --- AdaLN modulation in bf16 (avoids 7.21 GiB float32 copy of temb) ---
    if temb.ndim == 4:
        # temb: [B, N, 6, D] bf16.  Cast the tiny table (72 KB) down instead.
        table = self.scale_shift_table.unsqueeze(0).to(compute_dtype)  # [1, 1, 6, D]
        modulation = table + temb  # [B, N, 6, D] bf16, ~3.6 GiB (vs 7.21 GiB float32)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation.chunk(6, dim=2)
        del modulation
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    if norm_temb:
        shift_msa = normalize(shift_msa)
        scale_msa = normalize(scale_msa)
        c_shift_msa = normalize(c_shift_msa)
        c_scale_msa = normalize(c_scale_msa)

    # --- 1. Self-attention ---
    # Chunked norm+modulate (peak float32 temp: ~0.3 GiB per chunk instead of 1.22 GiB)
    norm_hidden_states = _chunked_pointwise(
        lambda hs, sc, sh: (self.norm1(hs.float()) * (1 + sc) + sh).type_as(hs),
        hidden_states, scale_msa, shift_msa, chunk_size=cs,
    )
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
    del norm_hidden_states
    # Chunked residual
    hidden_states = _chunked_pointwise(
        lambda hs, ao, g: (hs.float() + ao * g).type_as(hs),
        hidden_states, attn_output, gate_msa, chunk_size=cs,
    )
    del attn_output, shift_msa, scale_msa, gate_msa

    # --- 2. Cross-attention ---
    norm_hidden_states = _chunked_pointwise(
        lambda hs: self.norm2(hs.float()).type_as(hs),
        hidden_states, chunk_size=cs,
    )
    attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
    hidden_states = hidden_states + attn_output
    del norm_hidden_states, attn_output

    # --- 3. Feed-forward (FFN forward is chunked at module level) ---
    norm_hidden_states = _chunked_pointwise(
        lambda hs, sc, sh: (self.norm3(hs.float()) * (1 + sc) + sh).type_as(hs),
        hidden_states, c_scale_msa, c_shift_msa, chunk_size=cs,
    )
    ff_output = self.ffn(norm_hidden_states)
    del norm_hidden_states
    hidden_states = _chunked_pointwise(
        lambda hs, ff, g: (hs.float() + ff.float() * g).type_as(hs),
        hidden_states, ff_output, c_gate_msa, chunk_size=cs,
    )
    del ff_output, c_shift_msa, c_scale_msa, c_gate_msa

    return hidden_states


def _install_mem_efficient_blocks(transformer, token_chunk_size: int = 25000):
    """Install memory-efficient block_forward and chunked FFN on all transformer blocks.

    Patches:
    1. Block forward: bf16 AdaLN + chunked pointwise ops (saves ~10 GiB peak)
    2. FFN forward: chunked token processing (saves ~1.8 GiB peak)
    """
    import types

    for block in transformer.blocks:
        # Patch block forward
        block.forward = types.MethodType(
            lambda self, hs, temb, enc_hs, rot_emb, norm_temb, _cs=token_chunk_size: _mem_efficient_block_forward(
                self, hs, temb, enc_hs, rot_emb, norm_temb, _chunk_size=_cs
            ),
            block,
        )

        # Patch FFN forward
        original_ffn_forward = block.ffn.forward

        def _make_chunked_ffn(fwd, cs):
            def chunked_forward(hidden_states, *args, **kwargs):
                if hidden_states.shape[1] <= cs:
                    return fwd(hidden_states, *args, **kwargs)
                return torch.cat([fwd(c, *args, **kwargs) for c in hidden_states.split(cs, dim=1)], dim=1)
            return chunked_forward

        block.ffn.forward = _make_chunked_ffn(original_ffn_forward, token_chunk_size)


class SelfForcingModel(CausVidModel):
    """Self-Forcing model for distribution matching distillation
    Inheritance hierarchy:
    SelfForcingModel -> CausVidModel -> DMD2Model -> FastGenModel

    The major difference between SelfForcingModel and DMD2Model is how we get
    the gen_data in the single_train_step() function.  In SelfForcingModel, we
    use self.rollout_with_gradient() to get the gen_data, which
    does the rollout with gradient tracking at the last denoising step.  The
    number of denoising steps is stochastic, and is sampled from the
    denoising_step_list.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()
        # The bidirectional teacher and fake_score process [ref|cond|target] (53 frames =
        # 104,940 tokens). The original block_forward creates massive float32 temporaries:
        #   - temb.float(): 7.21 GiB
        #   - FFN GELU:     2.45 GiB
        # Install memory-efficient block forward + chunked FFN to cap peak at ~1.7 GiB.
        _install_mem_efficient_blocks(self.teacher.transformer, token_chunk_size=25000)
        _install_mem_efficient_blocks(self.fake_score.transformer, token_chunk_size=25000)

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            batch_size: Batch size
            real_data: Real data tensor for dtype/device reference

        Returns:
            input_student: Random noise used by the student
            t_max: Time step used by the student
            t: Time step for distribution matching
            eps: Random noise used by a forward process
        """
        batch_size = real_data.shape[0]

        # Multi-resolution ("bucket") support: trust real_data.shape over the static
        # config input_shape. Invalidate cached FlexAttention block_masks (tied to
        # spatial size/frame_seqlen) on the student, teacher, and fake_score
        # transformers whenever the per-step shape changes.
        input_shape = list(real_data.shape[1:])
        last_shape = getattr(self, "_last_input_shape", None)
        if last_shape != input_shape:
            for net in (getattr(self, "net", None),
                        getattr(self, "teacher", None),
                        getattr(self, "fake_score", None)):
                transformer = getattr(net, "transformer", None) if net is not None else None
                if transformer is not None and hasattr(transformer, "block_mask"):
                    transformer.block_mask = None
            self._last_input_shape = input_shape

        eps_student = torch.randn(batch_size, *input_shape, device=self.device, dtype=real_data.dtype)
        t_student = torch.full(
            (batch_size,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        input_student = self.net.noise_scheduler.latents(noise=eps_student)

        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )

        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)

        return input_student, t_student, t, eps

    def _sample_denoising_end_steps(self, num_blocks: int) -> List[int]:
        """Sample a list of denoising end indices for each block"""
        sample_steps = self.config.student_sample_steps

        if is_rank0():
            if self.config.last_step_only:
                indices = torch.full((num_blocks,), sample_steps - 1, dtype=torch.long, device=self.device)
            else:
                indices = torch.randint(low=0, high=sample_steps, size=(num_blocks,), device=self.device)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=self.device)

        # Broadcast the random indices to all ranks
        if world_size() > 1:
            dist.broadcast(indices, src=0)

        return indices.tolist()

    def rollout_with_gradient(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        enable_gradient: bool = True,
        start_gradient_frame: int = 0,
    ) -> torch.Tensor:
        """
        Perform self-forcing rollout with gradient tracking at the last step of each block.

        No external KV cache is used. Instead, we update the model's internal caches
        once per completed block using `store_kv=True` under no_grad.

        Args:
            noise: Initial noise tensor [B, C, T, H, W]
            condition: Conditioning (dict with 'text_embeds'/'prompt_embeds' or a tensor)
            enable_gradient: Whether to enable gradients at the exit step
            start_gradient_frame: Frame index to start gradient tracking

        Returns:
            generated_frames: Generated video frames, same shape as noise [B, C, T, H, W]
        """
        assert isinstance(self.net, CausalFastGenNetwork), f"{self.net} must be a CausalFastGenNetwork"
        self.net.clear_caches()

        # Reset peak memory stats for per-rollout VRAM monitoring
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        batch_size, C, num_frames, H, W = noise.shape
        chunk_size = self.net.chunk_size
        num_blocks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size
        sample_steps = self.config.student_sample_steps
        dtype = noise.dtype

        # Sample denoising end steps
        denoising_end_steps = self._sample_denoising_end_steps(num_blocks)
        logger.debug(f"denoising_end_steps: {denoising_end_steps}")

        # t_list
        t_list = self.config.sample_t_cfg.t_list
        if t_list is None:
            t_list = self.net.noise_scheduler.get_t_list(sample_steps, device=self.device)
        else:
            assert (
                len(t_list) - 1 == sample_steps
            ), f"t_list length (excluding zero) != student_sample_steps: {len(t_list) - 1} != {sample_steps}"
            t_list = torch.tensor(t_list, device=self.device, dtype=self.net.noise_scheduler.t_precision)

        # Collect denoised blocks and concatenate to preserve autograd graph
        denoised_blocks = []
        for block_idx in range(max(1, num_blocks)):
            if num_blocks == 0:
                # Handle case where num_frames < chunk_size
                cur_start_frame, cur_end_frame = 0, remaining_size
            else:
                # Normal chunking logic
                cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx + remaining_size
                cur_end_frame = chunk_size * (block_idx + 1) + remaining_size

            noisy_input = noise[:, :, cur_start_frame:cur_end_frame]

            # Denoising steps for current block
            for step, t_cur in enumerate(t_list):
                if self.config.same_step_across_blocks:
                    exit_flag = step == denoising_end_steps[0]
                else:
                    exit_flag = step == denoising_end_steps[block_idx]

                t_chunk_cur = t_cur.expand(batch_size)

                if not exit_flag:
                    # Non-exit steps: no grads, no cache updates
                    with torch.no_grad():
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )

                    # update to the next timestep for forward process
                    t_next = t_list[step + 1]
                    t_chunk_next = t_next.expand(batch_size)
                    if self.config.student_sample_type == "sde":
                        eps_infer = torch.randn_like(x0_pred_chunk)
                    elif self.config.student_sample_type == "ode":
                        eps_infer = self.net.noise_scheduler.x0_to_eps(xt=noisy_input, x0=x0_pred_chunk, t=t_chunk_cur)
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {self.config.student_sample_type}"
                        )
                    noisy_input = self.net.noise_scheduler.forward_process(x0_pred_chunk, eps_infer, t_chunk_next)
                else:
                    # Exit step: allow gradient if enabled
                    enable_grad = (
                        enable_gradient and torch.is_grad_enabled() and (cur_start_frame >= start_gradient_frame)
                    )
                    with torch.set_grad_enabled(enable_grad):
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )
                    break

            # Save denoised block; keep autograd path by collecting and concatenating later
            denoised_blocks.append(x0_pred_chunk)

            # Update internal KV cache for this finished block using t=0 or context noise (no grads)
            with torch.no_grad():
                if self.config.context_noise > 0:
                    # Add context noise to denoised frames before caching
                    t_cache = torch.full((batch_size,), self.config.context_noise, device=self.device, dtype=dtype)
                    x0_pred_cache = self.net.noise_scheduler.forward_process(
                        x0_pred_chunk,
                        torch.randn_like(x0_pred_chunk),
                        t_cache,
                    )
                else:
                    x0_pred_cache = x0_pred_chunk
                    t_cache = torch.zeros(batch_size, device=self.device, dtype=dtype)

                # update kv-cache with generated frames
                _ = self.net(
                    x0_pred_cache,
                    t_cache,
                    condition=condition,
                    cache_tag="pos",
                    store_kv=True,
                    cur_start_frame=cur_start_frame,
                    fwd_pred_type="x0",
                    is_ar=True,
                )

        # Concatenate blocks along the temporal dimension to form full output with gradients
        output = torch.cat(denoised_blocks, dim=2) if len(denoised_blocks) > 0 else torch.empty_like(noise)

        self.net.clear_caches()
        return output

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        del t_student
        enable_gradient = self.config.enable_gradient_in_rollout
        start_gradient_frame = self.config.start_gradient_frame

        if torch.is_grad_enabled() and enable_gradient:
            # Wrap rollout in gradient checkpoint: intermediate student activations
            # (~1.7 GB block-level checkpoints) are NOT saved during forward, freeing
            # memory for the subsequent bidirectional teacher/fake_score forwards.
            # They are recomputed during backward.
            def rollout_fn(noise):
                return self.rollout_with_gradient(
                    noise=noise,
                    condition=condition,
                    enable_gradient=enable_gradient,
                    start_gradient_frame=start_gradient_frame,
                )

            gen_data = torch.utils.checkpoint.checkpoint(
                rollout_fn,
                input_student,
                use_reentrant=False,
            )
        else:
            gen_data = self.rollout_with_gradient(
                noise=input_student,
                condition=condition,
                enable_gradient=enable_gradient,
                start_gradient_frame=start_gradient_frame,
            )
        return gen_data

