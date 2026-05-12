# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Causal Consistency Distillation (Causal CD) for V2VBG.

Alternative to ODE Init (Stage 2) that eliminates pre-computed trajectory data.
The teacher generates a single ODE step on-the-fly during training using only
ground-truth data.

Algorithm:
1. Sample timestep_idx from [0, N-2]; look up t_cur, t_next from discrete schedule
2. Add uniform noise: x_t = (1-t_cur)*x0 + t_cur*eps
3. Teacher ODE step (no grad): chunk-by-chunk AR with CFG, single Euler step → x_{t_next}
4. EMA: chunk-by-chunk AR with teacher forcing (clean prefill), predict x0 from x_{t_next}
5. Student: per-chunk backward — for each chunk: clean prefill → predict (with grad) → loss → backward
6. Loss: MSE(student_x0_pred, ema_x0_pred)

Teacher forcing: for chunk i, KV cache is prefilled with clean x0 chunks 0..i-1 at t=0,
then noisy chunk i is predicted. This matches Causal-Forcing's clean_x mechanism.

Optimization: incremental O(n) prefill — only prefill chunk i-1 before predicting chunk i
(cache accumulates), instead of clearing and re-prefilling 0..i-1 from scratch each time.
Per-chunk backward limits activation memory to 1 chunk at a time.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
import fastgen.utils.logging_utils as logger
from fastgen.methods.knowledge_distillation.KD import CausalKDModel
from fastgen.methods.model import FastGenModel

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class _ForwardTimer:
    """Lightweight per-forward CUDA event timer. Records start/end events per forward,
    synchronizes once at the end to compute all timings without stalling the GPU pipeline."""

    def __init__(self):
        self._events: List[tuple] = []  # [(start, end, label), ...]

    def record_start(self, label: str = ""):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, label)

    def record_end(self, token):
        start, label = token
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((start, end, label))

    def summarize(self) -> str:
        """Synchronize and return a timing summary string."""
        if not self._events:
            return ""
        self._events[-1][1].synchronize()
        times = {}
        for start, end, label in self._events:
            ms = start.elapsed_time(end)
            times.setdefault(label, []).append(ms / 1000.0)
        parts = []
        for label, vals in times.items():
            total = sum(vals)
            parts.append(f"{label}={total:.1f}s({len(vals)}x, avg={total/len(vals):.1f}s)")
        return "  ".join(parts)


class CausalCDModel(CausalKDModel):
    """Causal Consistency Distillation model.

    Inherits from CausalKDModel for _get_outputs (AR visualization) and
    _prepare_training_data (V2VBG conditioning). Overrides build_model to
    add a frozen teacher and discrete CD schedule, and single_train_step
    to implement the CD training loop with per-chunk backward.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        # Instantiate student network (from FastGenModel.build_model)
        FastGenModel.build_model(self)

        # Build frozen teacher from pretrained checkpoint
        self.build_teacher()

        # Init student weights from teacher + setup EMA
        self.load_student_weights_and_ema()

        # Enable true cache deletion for memory management.
        # KV caches are ~47GB each; only one network's cache should exist at a time.
        self.net._delete_cache_on_clear = True
        self.teacher._delete_cache_on_clear = True
        for ema_name in self.use_ema:
            getattr(self, ema_name)._delete_cache_on_clear = True

        # Compute discrete CD schedule: shifted sigmas, clamped to noise scheduler range
        cd_N = getattr(self.config, "cd_num_steps", 48)
        cd_shift = getattr(self.config, "cd_shift", 5.0)
        sigmas = torch.linspace(1.0, 0.0, cd_N + 1)[:-1]  # N values from 1.0 to ~0
        t_schedule = cd_shift * sigmas / (1 + (cd_shift - 1) * sigmas)
        ns = self.net.noise_scheduler
        t_schedule = t_schedule.clamp(ns.min_t, ns.max_t)
        self.register_buffer("cd_t_schedule", t_schedule.to(ns.t_precision))

    def _incremental_predict_chunks_no_grad(
        self,
        net,
        x_noisy: torch.Tensor,
        clean_x0: torch.Tensor,
        t_val: torch.Tensor,
        condition: Dict[str, Any],
        timer: _ForwardTimer | None = None,
        timer_prefix: str = "ema",
    ) -> List[torch.Tensor]:
        """Chunk-by-chunk AR with O(n) incremental clean prefill, all no_grad.

        For each chunk i:
        - If i > 0: prefill chunk i-1 with clean x0 (appends to existing cache)
        - Predict x0 from noisy chunk i at t_val (reads from cache)

        The cache accumulates incrementally: after processing chunk i,
        cache contains clean KV from chunks 0..i-1.

        Returns list of detached chunk tensors.
        """
        chunk_size = net.chunk_size
        B, C, T, H, W = x_noisy.shape
        num_chunks = T // chunk_size
        t_prec = self.net.noise_scheduler.t_precision
        t_zero = torch.zeros(B, device=self.device, dtype=t_prec)
        t_val_1d = t_val.expand(B)

        net.clear_caches()
        chunks = []
        for i in range(num_chunks):
            # Incrementally prefill chunk i-1 (cache already has 0..i-2)
            if i > 0:
                s_prev = (i - 1) * chunk_size
                tok = timer.record_start(f"{timer_prefix}_prefill") if timer else None
                net(
                    clean_x0[:, :, s_prev : s_prev + chunk_size],
                    t_zero,
                    condition=condition,
                    fwd_pred_type="x0",
                    is_ar=True,
                    cur_start_frame=s_prev,
                    store_kv=True,
                )
                if tok:
                    timer.record_end(tok)

            # Predict x0 from noisy chunk i (reads from cache containing 0..i-1)
            s = i * chunk_size
            tok = timer.record_start(f"{timer_prefix}_predict") if timer else None
            pred_i = net(
                x_noisy[:, :, s : s + chunk_size],
                t_val_1d,
                condition=condition,
                fwd_pred_type="x0",
                is_ar=True,
                cur_start_frame=s,
            )
            if tok:
                timer.record_end(tok)
            chunks.append(pred_i.detach())

        net.clear_caches()
        return chunks

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step for Causal Consistency Distillation.

        Three phases with O(n) incremental prefill + per-chunk backward:
        1. Teacher ODE step (no_grad, chunk-by-chunk AR with CFG) — 18 forwards
        2. EMA predictions (no_grad, chunk-by-chunk AR with clean prefill) — 9 forwards
        3. Student per-chunk backward: prefill + predict (with grad) + loss + backward — 9 fwd + 5 bwd

        Gradients are accumulated inside this method via per-chunk backward().
        The returned total_loss is a leaf tensor (for logging only); the trainer's
        backward() on it is harmless since it's not connected to model parameters.
        """
        torch.cuda.reset_peak_memory_stats()
        real_data, condition, neg_condition = self._prepare_training_data(data)
        clean_x0 = real_data  # [B, C, T, H, W]
        B = clean_x0.shape[0]

        # Sample from discrete schedule
        schedule = self.cd_t_schedule.to(self.device)
        idx = torch.randint(0, len(schedule) - 1, (1,), device=self.device)
        t_cur = schedule[idx]    # scalar-like
        t_next = schedule[idx + 1]  # t_next < t_cur (toward clean)

        # Add noise uniformly across all frames
        eps = torch.randn_like(clean_x0)
        x_t = self.net.noise_scheduler.forward_process(clean_x0, eps, t_cur.expand(B))

        t0_step = time.monotonic()
        timer = _ForwardTimer()

        # === Phase 1: Teacher ODE step (O(n) incremental, no_grad) ===
        t0 = time.monotonic()
        with torch.no_grad():
            x_t_next = self._teacher_ode_step(
                x_t, clean_x0, t_cur, t_next, condition, neg_condition,
                timer=timer,
            )
        torch.cuda.synchronize()
        t_teacher = time.monotonic() - t0

        # === Phase 2: EMA predictions (O(n) incremental, no_grad) ===
        t0 = time.monotonic()
        with torch.no_grad():
            ema_net = getattr(self, self.use_ema[0])
            ema_chunks = self._incremental_predict_chunks_no_grad(
                ema_net, x_t_next, clean_x0, t_next, condition,
                timer=timer, timer_prefix="ema",
            )
        torch.cuda.synchronize()
        t_ema = time.monotonic() - t0

        # === Phase 3: Student per-chunk backward (O(n) incremental) ===
        t0 = time.monotonic()
        chunk_size = self.net.chunk_size
        num_chunks = clean_x0.shape[2] // chunk_size
        t_prec = self.net.noise_scheduler.t_precision
        t_zero = torch.zeros(B, device=self.device, dtype=t_prec)
        t_cur_1d = t_cur.expand(B)

        self.net.clear_caches()
        student_chunks = []
        loss_accum = 0.0
        for i in range(num_chunks):
            # Incrementally prefill chunk i-1 (no_grad)
            if i > 0:
                with torch.no_grad():
                    s_prev = (i - 1) * chunk_size
                    tok = timer.record_start("stu_prefill")
                    self.net(
                        clean_x0[:, :, s_prev : s_prev + chunk_size],
                        t_zero,
                        condition=condition,
                        fwd_pred_type="x0",
                        is_ar=True,
                        cur_start_frame=s_prev,
                        store_kv=True,
                    )
                    timer.record_end(tok)

            # Predict x0 from noisy chunk i (WITH grad)
            s = i * chunk_size
            tok = timer.record_start("stu_predict")
            pred_i = self.net(
                x_t[:, :, s : s + chunk_size],
                t_cur_1d,
                condition=condition,
                fwd_pred_type="x0",
                is_ar=True,
                cur_start_frame=s,
            )
            timer.record_end(tok)

            # Per-chunk loss and backward — frees autograd graph immediately
            tok = timer.record_start("stu_backward")
            loss_i = F.mse_loss(pred_i, ema_chunks[i])
            (loss_i / num_chunks).backward()
            timer.record_end(tok)
            loss_accum += loss_i.item() / num_chunks

            student_chunks.append(pred_i.detach())

        self.net.clear_caches()
        torch.cuda.synchronize()
        t_student = time.monotonic() - t0

        t_total = time.monotonic() - t0_step

        # Profile logging
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(
            f"[CD iter {iteration}] teacher={t_teacher:.1f}s  ema={t_ema:.1f}s  "
            f"student={t_student:.1f}s  total={t_total:.1f}s  "
            f"peak_mem={peak_gb:.1f}GB  cur_mem={torch.cuda.memory_allocated()/1e9:.1f}GB"
        )
        logger.info(f"[CD iter {iteration} detail] {timer.summarize()}")

        # Reconstruct full x0_pred for _get_outputs (visualization)
        x0_pred = torch.cat(student_chunks, dim=2)

        # Leaf tensor for logging — trainer's backward() on this is harmless
        total_loss = torch.tensor(loss_accum, device=self.device, requires_grad=True)

        loss_map = {"total_loss": total_loss, "cd_loss": total_loss}
        outputs = self._get_outputs(x0_pred, condition=condition)
        return loss_map, outputs

    def _teacher_ode_step(
        self,
        x_t: torch.Tensor,
        clean_x0: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        condition: Any,
        neg_condition: Any,
        timer: _ForwardTimer | None = None,
    ) -> torch.Tensor:
        """Single Euler ODE step by frozen teacher, O(n) incremental prefill with CFG.

        For each CFG branch (pos/neg), processes chunks sequentially:
        - chunk 0: predict directly (no context needed)
        - chunk i>0: prefill chunk i-1 (incrementally extending cache), then predict chunk i

        Total: 2 branches × (4 prefills + 5 predicts) = 18 forwards.
        """
        chunk_size = self.teacher.chunk_size
        B, C, T, H, W = x_t.shape
        num_chunks = T // chunk_size
        t_prec = self.net.noise_scheduler.t_precision
        dt = (t_next - t_cur).to(x_t.dtype)

        t_zero = torch.zeros(B, device=self.device, dtype=t_prec)
        t_cur_1d = t_cur.expand(B)

        flows = {"pos": [], "neg": []}
        for branch, cond in [("pos", condition), ("neg", neg_condition)]:
            self.teacher.clear_caches()

            for i in range(num_chunks):
                # Incrementally prefill chunk i-1 (cache already has 0..i-2)
                if i > 0:
                    s_prev = (i - 1) * chunk_size
                    tok = timer.record_start(f"t_{branch}_prefill") if timer else None
                    self.teacher(
                        clean_x0[:, :, s_prev : s_prev + chunk_size],
                        t_zero,
                        condition=cond,
                        fwd_pred_type="flow",
                        is_ar=True,
                        cur_start_frame=s_prev,
                        store_kv=True,
                        cache_tag=branch,
                    )
                    if tok:
                        timer.record_end(tok)

                # Predict flow for noisy chunk i
                s = i * chunk_size
                tok = timer.record_start(f"t_{branch}_predict") if timer else None
                flow_i = self.teacher(
                    x_t[:, :, s : s + chunk_size],
                    t_cur_1d,
                    condition=cond,
                    fwd_pred_type="flow",
                    is_ar=True,
                    cur_start_frame=s,
                    cache_tag=branch,
                )
                if tok:
                    timer.record_end(tok)
                flows[branch].append(flow_i)

        self.teacher.clear_caches()

        # Apply CFG and Euler step per chunk
        cfg = self.config.guidance_scale
        parts = []
        for i in range(num_chunks):
            flow = flows["neg"][i] + cfg * (flows["pos"][i] - flows["neg"][i])
            s = i * chunk_size
            parts.append(x_t[:, :, s : s + chunk_size] + flow * dt)

        return torch.cat(parts, dim=2)
