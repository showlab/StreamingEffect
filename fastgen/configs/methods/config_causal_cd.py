# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Default method config for Causal Consistency Distillation (Causal CD).

Uses ground-truth data only — no pre-computed ODE trajectories.
The teacher generates a single Euler step on-the-fly during training.
"""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from typing import Any
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.methods import CausalCDModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    context_noise: float = 0.0
    cd_num_steps: int = 48       # N for discrete CD schedule
    cd_shift: float = 5.0        # shift parameter for schedule
    use_ema: Any = True          # CD requires EMA network


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(CausalCDModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.add_teacher_to_fsdp_dict = True
    config.model.net_scheduler.warm_up_steps = [0]

    return config
