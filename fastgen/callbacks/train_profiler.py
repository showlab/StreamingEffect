# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class TrainProfilerCallback(Callback):
    """Callback for profiling training speed and detailed timing breakdowns.

    Tracks:
    - iter_time: seconds per iteration (wall clock time)
    - data_load_time: time spent loading data
    - avg_forward_time: average forward pass time across accumulation steps
    - backward_time: time spent in backward pass
    - optim_step_time: time spent in optimizer step
    """

    def __init__(self, every_n: int = 100, detailed: bool = True):
        """Initialize the profiler callback.

        Args:
            every_n: Log metrics every N iterations
            detailed: If True, log detailed timing breakdown. If False, only log iter_time.
        """
        # For iter_time tracking
        self.last_log_time = None

        # For detailed profiling
        self.detailed = detailed
        self.train_step_begin_time = None
        self.accum_begin_times = None
        self.backward_begin_times = None
        self.optimizer_step_begin = None
        self.step_end_time = None
        self.every_n = every_n

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        if hasattr(self, "config"):
            # overwritten by logging_iter if self.config exists
            self.every_n = self.config.trainer.logging_iter
        logger.info(f"every_n to profile trainer: {self.every_n}")

    def on_training_step_begin(
        self,
        model: FastGenModel,
        iteration: int = 0,
    ):
        if self.detailed:
            self.train_step_begin_time = time.perf_counter()
            self.accum_begin_times = []
            self.backward_begin_times = []

    def on_training_accum_step_begin(
        self, model: FastGenModel, data_batch: dict[str, torch.Tensor], iteration: int = 0, accum_iter: int = 0
    ):
        if self.detailed:
            self.accum_begin_times.append(time.perf_counter())

    def on_backward_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ):
        if self.detailed:
            self.backward_begin_times.append(time.perf_counter())

    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0):
        if self.detailed:
            self.optimizer_step_begin = time.perf_counter()

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict

        if self.detailed:
            self.step_end_time = time.perf_counter()

        if hasattr(self, "config"):
            # only wandb log when config exists
            if iteration % self.every_n == 0 and is_rank0():
                metrics = {}

                # Calculate iter_time (wall clock time per iteration)
                cur_time = time.time()
                if self.last_log_time is not None:
                    iter_time = (cur_time - self.last_log_time) / self.every_n
                    logger.info(f"{iteration} : avg iteration time       {iter_time:.2f} seconds")
                    metrics["profiler/avg_iteration_time"] = iter_time
                self.last_log_time = cur_time

                # Calculate detailed timing breakdown
                if self.detailed and self.accum_begin_times and self.backward_begin_times:
                    data_load_time = self.accum_begin_times[0] - self.train_step_begin_time
                    forward_time = sum(
                        [b - a for (b, a) in zip(self.backward_begin_times, self.accum_begin_times)]
                    ) / len(self.accum_begin_times)
                    backward_time = self.optimizer_step_begin - self.backward_begin_times[-1]
                    optim_step_time = self.step_end_time - self.optimizer_step_begin

                    logger.info(f"{iteration} : data loading time        {data_load_time:.2f}")
                    logger.info(f"{iteration} : avg forward pass time    {forward_time:.2f}")
                    logger.info(f"{iteration} : backward pass time       {backward_time:.2f}")
                    logger.info(f"{iteration} : optimizer step time      {optim_step_time:.2f}")

                    metrics.update(
                        {
                            "profiler/data_loading_time": data_load_time,
                            "profiler/avg_forward_pass_time": forward_time,
                            "profiler/backward_pass_time": backward_time,
                            "profiler/optimizer_step_time": optim_step_time,
                        }
                    )

                if wandb.run and metrics:
                    wandb.log(metrics, step=iteration)
