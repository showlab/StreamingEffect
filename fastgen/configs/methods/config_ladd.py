# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import LADDModel
from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)

""" Configs for the LADD model, on CIFAR-10 dataset. """


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    discriminator: DictConfig = attrs.field(factory=lambda: copy.deepcopy(Discriminator_EDM_CIFAR10_Config))
    # optimizer and scheduler for the discriminator
    discriminator_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    discriminator_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # student update frequency
    student_update_freq: int = 5

    # use the same t and noise to perturb the real data and fake data
    gan_use_same_t_noise: bool = False

    # R1 regularization weight (0 means no R1 reg, recommended value when using R1 reg: 100-1000)
    gan_r1_reg_weight: float = 0.0
    # R1 regularization noise scale (it only takes effect when gan_r1_reg_weight > 0)
    gan_r1_reg_alpha: float = 0.1


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(LADDModel)(
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

    config.dataloader_train.batch_size = 256
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
