# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path
import io
import torch.nn
from typing import Optional, Dict, Union, Any

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from fastgen.configs.config import BaseCheckpointerConfig
from fastgen.utils.distributed.s3_filesystem import S3StorageWriter, S3StorageReader
from fastgen.utils.io_utils import s3_load, s3_save, latest_checkpoint
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import synchronize, is_rank0
from fastgen.callbacks.callback import CallbackDict


class Checkpointer:
    """Class to save and load model checkpoints"""

    def __init__(self, config: BaseCheckpointerConfig):
        self.config = config

    def _save_checkpoint(self, save_dict: Dict[str, Any], path: str):
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            logger.info(f"Saving the model to {path}")
            # convert the save_dict to bytes
            buffer = io.BytesIO()
            torch.save(save_dict, buffer)
            s3_save(path, buffer.getvalue())
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(save_dict, path)

    def _load_checkpoint(self, path: str, device: torch.device = "cpu") -> Dict[str, Any]:
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            state = torch.load(s3_load(path), map_location=device)
        else:
            assert os.path.exists(path), f"{path} does not exist"
            state = torch.load(path, map_location=device, weights_only=False)
        return state

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """
        synchronize()
        model_state = {k: v.state_dict() for k, v in model_dict.items()}
        optim_state = None if optimizer_dict is None else {k: v.state_dict() for k, v in optimizer_dict.items()}
        scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
        grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
        callbacks_state = None if callbacks is None else callbacks.state_dict()
        save_dict = {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": scheduler_state,
            "grad_scaler": grad_scaler_state,
            "callbacks": callbacks_state,
            "iteration": iteration,
        }

        if is_rank0():
            if path is None:
                if self.config.use_s3:
                    path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}.pth")
                else:
                    path = os.path.join(self.config.save_dir, f"{iteration:07d}.pth")

            if not self.config.use_s3:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving the model to {path}")
            self._save_checkpoint(save_dict, path)

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".

        Returns:
            int: The iteration number

        """
        # TODO: rank 0 load and broadcast to all other ranks
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path) + ".pth"
            if path == ".pth":
                # no checkpoint found, starting from iteration 0
                return 0
        if not os.path.exists(path):
            logger.critical(f"Checkpoint file not found at {path}")
            return 0
        logger.info(f"Loading model from {path}")
        state = self._load_checkpoint(path, device=device)

        logger.info("Loading the model_dict...")
        for k, v in model_dict.items():
            if k in state["model"] and v is not None:
                # strict is False to allow evaluating external checkpoints without, e.g., logvar parameters
                model_load_info = v.load_state_dict(state["model"][k], strict=False)
                logger.info(f"Model {k}, loading info: {model_load_info}")
            else:
                logger.warning(f"Model {k} not found in checkpoint.")

        if optimizer_dict is not None:
            logger.info("Loading the optimizer_dict...")
            for k, v in optimizer_dict.items():
                if k in state["optimizer"] and v is not None:
                    v.load_state_dict(state["optimizer"][k])
                else:
                    logger.warning(f"Optimizer {k} not found in checkpoint.")

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"] and v is not None:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        if "iteration" not in state:
            logger.warning("Iteration not found in checkpoint.")
            return 0
        return state["iteration"]


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(self, model: torch.nn.Module, options: StateDictOptions | None = None):
        self.model = model
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        return get_model_state_dict(self.model, options=self.options)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_model_state_dict(self.model, model_state_dict=state_dict, options=self.options)


class OptimizerWrapper(Stateful):
    """
    Wrapper for optimizer state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, options: StateDictOptions | None = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=self.options)
        return optimizer_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=state_dict, options=self.options)


class FSDPCheckpointer(Checkpointer):
    """Class to save and load model checkpoints"""

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.config.use_s3:
            return S3StorageWriter(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.config.use_s3:
            return S3StorageReader(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """

        if path is None:
            if self.config.use_s3:
                path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}")
            else:
                path = os.path.join(self.config.save_dir, f"{iteration:07d}")
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir, exist_ok=True)
        if path.endswith(".pth"):  # In the case of autoresume
            path = path[:-4]
        logger.info(f"Saving FSDP model to prefix {path}")

        synchronize()
        # fsdp should save on all ranks
        for k, v in model_dict.items():
            model_state_dict = ModelWrapper(model=v).state_dict()
            storage_writer = self.get_storage_writer(checkpoint_path=f"{path}.{k}_model")
            dcp.save(model_state_dict, storage_writer=storage_writer)

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                optim_state_dict = OptimizerWrapper(model=model_dict[k], optimizer=v).state_dict()
                storage_writer = self.get_storage_writer(checkpoint_path=f"{path}.{k}_optim")
                dcp.save(optim_state_dict, storage_writer=storage_writer)

        # other scalars only save on rank 0
        if is_rank0():
            scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
            grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
            callbacks_state = None if callbacks is None else callbacks.state_dict()
            save_dict = {
                "scheduler": scheduler_state,
                "grad_scaler": grad_scaler_state,
                "callbacks": callbacks_state,
                "iteration": iteration,
            }
            self._save_checkpoint(save_dict, f"{path}.pth")

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".

        Returns:
            int: The iteration number

        """
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path)
            if path == "":
                # no checkpoint found, starting from iteration 0
                return 0
        if path.endswith(".pth"):
            # switch to basic checkpointer
            logger.debug(f"Loading non-FSDP model from {path}")
            return super().load(
                model_dict,
                optimizer_dict=optimizer_dict,
                scheduler_dict=scheduler_dict,
                grad_scaler=grad_scaler,
                callbacks=callbacks,
                path=path,
                device=device,
            )
        if not os.path.exists(f"{path}.pth"):
            logger.critical(f"Checkpoint file not found at {path}")
            return 0
        logger.info(f"Loading FSDP model from prefix {path}")

        for k, v in model_dict.items():
            logger.info(f"Loading the FSDP model dict for key {k}...")
            model_wrapper = ModelWrapper(model=v)
            model_state_dict = model_wrapper.state_dict()
            assert os.path.exists(f"{path}.{k}_model"), f"Key {k} does not exist in FSDP model dict"
            storage_reader = self.get_storage_reader(checkpoint_path=f"{path}.{k}_model")
            dcp.load(
                state_dict=model_state_dict,
                storage_reader=storage_reader,
            )
            model_wrapper.load_state_dict(model_state_dict)

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                logger.info(f"Loading the FSDP optimizer dict for key {k}...")
                optim_wrapper = OptimizerWrapper(model=model_dict[k], optimizer=v)
                # For fresh optimizers with no state, we need to initialize with fake gradients
                # that are DTensors (not regular Tensors) to avoid the mixed Tensor/DTensor error
                if len(v.state) == 0:
                    # Set fake DTensor gradients to initialize optimizer state
                    for param in model_dict[k].parameters():
                        if param.requires_grad and param.grad is None:
                            param.grad = torch.zeros_like(param)
                optim_state_dict = optim_wrapper.state_dict()
                assert os.path.exists(f"{path}.{k}_model"), f"Key {k} does not exist in FSDP model dict"
                storage_reader = self.get_storage_reader(checkpoint_path=f"{path}.{k}_optim")

                try:
                    dcp.load(
                        state_dict=optim_state_dict,
                        storage_reader=storage_reader,
                    )
                    optim_wrapper.load_state_dict(optim_state_dict)
                    logger.success(f"Successfully loaded optimizer state for {k}")
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "Missing key" in error_msg
                        or "Unexpected key" in error_msg
                        or "CheckpointException" in error_msg
                    ):
                        logger.warning(f"Optimizer checkpoint compatibility issue for {k}: {type(e).__name__}")
                        logger.warning(f"Initializing fresh optimizer state for {k} - training will continue")
                        # Reset to fresh optimizer state
                        v.__setstate__({"state": {}, "param_groups": v.param_groups})
                        logger.info(f"Reset optimizer state for {k} due to parameter mismatch")
                    else:
                        logger.error(f"Unexpected optimizer loading error for {k}: {e}")
                        raise e

        state = self._load_checkpoint(f"{path}.pth", device=device)

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"]:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        return state["iteration"]
