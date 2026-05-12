# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Set, Union, Tuple, Any
from abc import ABC, abstractmethod

import torch

from fastgen.networks.noise_schedule import get_noise_schedule, NET_PRED_TYPES
import fastgen.utils.logging_utils as logger


class FastGenNetwork(ABC, torch.nn.Module):
    """
    Abstract base class for FastGen neural network architectures.

    This class provides a common interface for all neural network models used in the FastGen framework
    for fast generative modeling. It defines the core methods that must be implemented by specific
    network architectures like EDM, EDM2, SD15, CogVideoX, etc.

    The class handles noise scheduling, prediction types, and provides abstract methods for
    forward passes and sampling operations.

    Attributes:
        net_pred_type (str): The prediction type of the network ('x0', 'eps', 'v', 'flow').
        schedule_type (str): The noise schedule type used by the network.
        noise_scheduler: The noise scheduler instance for handling diffusion timesteps.

    Args:
        net_pred_type (str, optional): The prediction type of the network.
            Must be one of ['x0', 'eps', 'v', 'flow']. Defaults to "x0".
        schedule_type (str, optional): The noise schedule type to use.
            Common types include 'edm', 'rf', 'sd'. Defaults to "edm".
        **net_kwargs: Additional keyword arguments passed to the noise scheduler.
    """

    @staticmethod
    def _is_in_meta_context() -> bool:
        """Check if we're currently in a torch.device('meta') context.

        This allows networks to detect when they're being instantiated inside
        a meta device context manager (used for FSDP memory-efficient loading).
        When in meta context, networks should use from_config instead of from_pretrained
        to avoid loading weights that will be broadcast from rank 0.

        Returns:
            bool: True if currently in a meta device context, False otherwise.
        """
        test_param = torch.nn.Parameter(torch.empty(0))
        return test_param.device.type == "meta"

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        **net_kwargs,
    ):
        super().__init__()

        # Default prediction type - can be overridden by subclasses
        self.net_pred_type = net_pred_type
        self.schedule_type = schedule_type
        self._validate_net_pred_type(self.net_pred_type)
        self.set_noise_schedule(**net_kwargs)

    def _validate_net_pred_type(self, pred_type: str) -> None:
        """Validate that the network prediction type is supported.

        Args:
            pred_type (str): The prediction type to validate.

        Raises:
            ValueError: If the prediction type is not supported.
        """
        if pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported net_pred_type '{pred_type}'. " f"Supported types are: {NET_PRED_TYPES}")

    def set_noise_schedule(self, schedule_type: Optional[str] = None, **noise_schedule_kwargs) -> None:
        """Set up the noise scheduler for the network.

        Args:
            schedule_type (Optional[str]): Type of noise schedule to use.
                If None, uses the current self.schedule_type.
            **noise_schedule_kwargs: Additional arguments passed to the noise scheduler.
        """
        if schedule_type is not None:
            self.schedule_type = schedule_type
        self.noise_scheduler = get_noise_schedule(self.schedule_type, **noise_schedule_kwargs)

    def reset_parameters(self):
        """Reset the parameters of the network.

        Subclasses should override this method to reinitialize their specific parameters,
        and call super().reset_parameters() to handle common components like the noise scheduler.
        """
        # Reinitialize noise scheduler (its _sigmas tensor is not a registered buffer)
        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            self.set_noise_schedule()
            logger.debug("Reinitialized noise scheduler")

    def fully_shard(self, **kwargs):
        """Fully shard the network.

        Subclasses should override this method to shard their specific components.
        The default implementation raises NotImplementedError because the base class
        inherits from ABC, which causes issues with FSDP2's __class__ assignment.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the fully_shard method. "
            "Use external sharding methods or implement this method in the subclass."
        )

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the trained network.

        This method performs the sampling/inference process to generate new data from noise.
        It should implement the specific sampling algorithm for the network architecture.

        Note: This method is optional. Networks that don't support direct sampling
        can leave this unimplemented and use external sampling methods instead.

        Args:
            noise (torch.Tensor): Initial noise tensor to start sampling from.
                Shape should match the expected input format of the network.
            condition (Any): Conditioning information for guided generation.
                Can be text embeddings, class labels, or other structured conditioning data.
                Can be a single tensor, list of tensors, or dictionary mapping condition names to tensors.
                Defaults to None for unconditional generation.
            neg_condition (Any): Negative conditioning for classifier-free guidance. Defaults to None.
            guidance_scale (Optional[float], optional): Strength of guidance for conditional generation. None means no guidance.
                Higher values increase adherence to conditioning. Defaults to 5.0.
            num_steps (int, optional): Number of denoising steps to perform during sampling.
                More steps generally lead to higher quality but slower generation. Defaults to 50.
            **kwargs: Additional keyword arguments specific to the sampling implementation.

        Returns:
            torch.Tensor: Generated samples with the same spatial dimensions as latents
                but potentially different channel dimensions based on the network's output.

        Raises:
            NotImplementedError: If the network doesn't provide its own sampling implementation.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the sample method. "
            "Use external sampling methods or implement this method in the subclass."
        )

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of diffusion model.

        This method performs a single forward pass through the network, predicting the denoised
        output or noise based on the current noisy input and timestep.

        Args:
            x_t (torch.Tensor): The diffused/noisy data sample at timestep t.
                Shape should be [batch_size, channels, ...spatial_dims].
            t (torch.Tensor): The current timestep(s). Can be a scalar or tensor
                with shape [batch_size].
            condition (Any, optional): Optional conditioning information such as text embeddings, class labels,
                or attention masks. Can be a single tensor, list of tensors, or dictionary
                mapping condition names to tensors. Format depends on the specific network implementation.
                Defaults to None.
            r (Optional[torch.Tensor], optional): Additional timestep parameter,
                primarily used by mean flow methods. Defaults to None.
            return_features_early (bool, optional): If True, returns intermediate features
                as soon as they are computed, without completing the full forward pass.
                Defaults to False.
            feature_indices (Optional[Set[int]], optional): Set of layer indices to extract
                features from. If non-empty, features will be returned along with the output.
                Defaults to None (no feature extraction).
            return_logvar (bool, optional): If True, returns the log-variance estimate
                along with the main output. Defaults to False.
            fwd_pred_type (Optional[str], optional): Override the network's prediction type
                for this forward pass. Must be in ['x0', 'eps', 'v', 'flow'].
                None uses the network's default net_pred_type. Defaults to None.
            **fwd_kwargs: Additional keyword arguments specific to the network implementation.

        Returns:
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
                - If return_features_early=True: List of intermediate features
                - If feature_indices is non-empty: List containing [output, features]
                - If return_logvar=True: Tuple containing (output, logvar)
                - Otherwise: Single output tensor with predictions

        Raises:
            ValueError: If fwd_pred_type is provided but not supported.
            NotImplementedError: Must be implemented by concrete subclasses.
        """


class CausalFastGenNetwork(FastGenNetwork):
    """
    Abstract base class for causal (autoregressive) generative networks.

    This class extends FastGenNetwork to support causal generation patterns,
    where outputs are generated sequentially with dependency on previous outputs.
    It is designed for models that need to maintain temporal causality and can
    benefit from chunked processing for memory efficiency.

    Key Features:
    - Chunked processing for long sequences to manage memory usage
    - KV cache management for attention-based models
    - Support for causal attention patterns
    - Abstract interface for cache management operations

    Common use cases include:
    - Autoregressive video generation (like CausalWan)
    - Sequential text generation
    - Time-series modeling with causal dependencies
    - Any model requiring temporal causality constraints

    Args:
        chunk_size (int): Size of chunks for processing long sequences.
            Smaller values use less memory but may be slower due to increased
            overhead. Typical values range from 1-16 depending on available
            GPU memory and sequence length. Defaults to 3.
        **net_kwargs: Additional keyword arguments passed to FastGenNetwork,
            including net_pred_type, schedule_type, etc.

    Abstract Methods:
        clear_caches(): Must be implemented by subclasses to clear all internal
            caches (KV caches, attention caches, etc.). Should be called after
            generation sequences or when memory usage becomes too high.
    """

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        chunk_size: int = 3,
        total_num_frames: int = 21,
        **net_kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **net_kwargs)
        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames

    @abstractmethod
    def clear_caches(self):
        """
        Clear all internal caches used for causal generation.

        This method must be implemented by subclasses to clear model-specific
        caches such as:
        - Key-Value (KV) caches from attention layers
        - Intermediate activation caches
        - Positional embedding caches
        - Any other memory-intensive cached data

        Should be called:
        - After each complete generation sequence
        - When switching between different generation tasks
        - When memory usage becomes too high
        - Before starting a new causal generation session
        """
