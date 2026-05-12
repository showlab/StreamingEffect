# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Any, Dict, List

import pandas as pd
import psutil
import torch

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import world_size, is_rank0, synchronize
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def log_prof_data(data_list: List[Dict[str, Any]]):
    # Create a table to log data with rank information
    metrics = list(data_list[0].keys())

    # Initialize dictionaries to store min and max values for each metric
    min_values = {key: float("inf") for key in metrics}
    max_values = {key: float("-inf") for key in metrics}
    sum_values = {key: 0.0 for key in metrics}

    count = 0

    for _rank, prof_data in enumerate(data_list):
        count += 1

        # Update min, max, and sum values
        for key in metrics:
            min_values[key] = min(min_values[key], prof_data[key])
            max_values[key] = max(max_values[key], prof_data[key])
            sum_values[key] += prof_data[key]

    # Calculate average values
    avg_values = {key: sum_values[key] / count for key in metrics}
    summary_df = pd.DataFrame({"Avg": avg_values, "Max": max_values, "Min": min_values})

    logger.info(f"GPU stats:\n{summary_df.to_string()}")


class GPUStatsCallback(Callback):
    def __init__(self, every_n: int = 100):
        self.every_n = every_n

    def on_train_begin(self, model: FastGenModel, iteration: int = 0):
        torch.cuda.reset_peak_memory_stats()
        if hasattr(self, "config"):
            # overwritten by logging_iter if self.config exists
            self.every_n = self.config.trainer.logging_iter
        logger.info(f"every_n to measure gpus stats: {self.every_n}")

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict
        if iteration % self.every_n == 0:
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            cpu_mem_gb = cpu_memory_usage / (1024**3)

            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            util = torch.cuda.utilization()

            prof_data = {
                "cpu_mem_gb": float(cpu_mem_gb),
                "peak_gpu_mem_gb": float(peak_gpu_mem_gb),
                "peak_gpu_mem_reserved_gb": float(peak_gpu_mem_reserved_gb),
                "util": float(util),
            }

            synchronize()
            data_list = [prof_data] * world_size()
            # this is blocking by default
            if world_size() > 1:
                torch.distributed.all_gather_object(data_list, prof_data)

            if is_rank0():
                log_prof_data(data_list)
            synchronize()
