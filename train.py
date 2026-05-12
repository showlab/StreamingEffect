# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The training entry script for the FastGen project. Works for both DDP and FSDP training.
"""

import argparse
import warnings

import torch._dynamo
torch._dynamo.config.recompile_limit = 64  # default 8 is too low for multi-resolution bucketing

from fastgen.configs.config import BaseConfig
from fastgen.utils import instantiate
from fastgen.trainer import Trainer
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import synchronize, clean_up
from fastgen.utils.scripts import parse_args, setup

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 2.6.


def main(config: BaseConfig):
    # initiate the model
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    # initiate the trainer
    logger.info("Initializing trainer...")
    fastgen_trainer = Trainer(config)
    logger.success("Trainer initialized successfully")
    synchronize()

    # Start training
    fastgen_trainer.run(model)
    synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    args = parse_args(parser)
    config = setup(args)

    main(config)

    clean_up()
    logger.info("Training finished.")
