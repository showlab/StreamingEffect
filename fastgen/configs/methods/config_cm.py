# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import attrs
from omegaconf import DictConfig
from typing import Any

from fastgen.configs.opt import RAdamOptimizerConfig
from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
    SampleTConfig as BaseSampleTConfig,
)
from fastgen.configs.callbacks import (
    CTSchedule_CALLBACK,
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from fastgen.methods import CMModel


@attrs.define(slots=False)
class SampleTConfig(BaseSampleTConfig):
    """Config for sampling t from a time distribution for CM."""

    time_dist_type: str = "lognormal"

    # lowest value for end point (min_r > 0 requires the network to satisfy net(x_t, t) ~ x_t for small t).
    min_r: float = 0.0
    # quantize t and r
    quantize: bool = False


@attrs.define(slots=False)
class LossConfig:
    """Config for the losses in CM"""

    # use consistency distillation
    use_cd: bool = False
    # the constant value in the pseudo-huber loss
    huber_const: float = 1e-8
    # use the squared l2 loss or the l2 loss
    use_squared_l2: bool = False
    # weighting of the CT loss, choices: ['default', 'c_out', 'c_out_sq', 'sqrt', 'one']
    weighting_ct_loss: str = "default"


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    # optimizer
    net_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(RAdamOptimizerConfig))

    # config for sampling t
    sample_t_cfg: SampleTConfig = attrs.field(factory=SampleTConfig)

    # EMA for the main net (requires EMACallback)
    use_ema: Any = True

    # config for losses
    loss_config: LossConfig = attrs.field(factory=LossConfig)


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(CMModel)(
        config=None,
    )


def create_config():
    config = Config()

    # trainer
    config.trainer.callbacks = DictConfig(
        {
            **CTSchedule_CALLBACK,
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )  # CM training must have the ct_schedule callback
    # recommended setting for CIFAR-10 is max_iter * batch_size // (8 * 1000)
    config.trainer.callbacks.ct_schedule.kimg_per_stage = 6400
    config.trainer.callbacks.grad_clip.grad_norm = 1000000
    config.trainer.tf32_enabled = False

    # model
    config.model.net.dropout = 0.2
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.net_optimizer.weight_decay = 0.0
    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003
    config.model.net.label_dim = 0

    # dataloader (unconditional CIFAR-10)
    config.dataloader_train.use_labels = False

    return config
