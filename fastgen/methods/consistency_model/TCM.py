# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Dict, Any, Callable, TYPE_CHECKING, Optional

import torch
from torch import Tensor

from fastgen.methods import CMModel
from fastgen.utils.distributed import sync_all, sync_any, move_module_to_device
from fastgen.utils.basic_utils import convert_cfg_to_dict

if TYPE_CHECKING:
    from fastgen.configs.methods.config_tcm import ModelConfig
    from fastgen.networks.network import FastGenNetwork


class TCMPrecond(torch.nn.Module):
    """Two-stage Consistency Model Preconditioner.

    This module implements a two-stage consistency model that switches between
    a teacher network (stage-1) and a student network (stage-2) based on the
    transition time step. For timesteps below the transition threshold, it uses
    the teacher network; for timesteps above, it uses the student network.

    The implementation handles distributed training scenarios with proper
    synchronization across ranks to ensure consistent behavior in FSDP setups.

    Args:
        net_t: Teacher network for stage-1 (t < transition_t)
        net_s: Student network for stage-2 (t >= transition_t)
        transition_t: Transition time step threshold (default: 2.0)
    """

    def __init__(
        self,
        net_t: FastGenNetwork,
        net_s: FastGenNetwork,
        transition_t: float = 2.0,
    ) -> None:
        super().__init__()

        self.net_t = net_t
        self.net_s = net_s
        self.transition_t = transition_t

        assert hasattr(self.net_s, "noise_scheduler")
        assert (
            self.net_s.noise_scheduler.min_t <= transition_t <= self.net_s.noise_scheduler.max_t
        ), f"transition_t must be in [min_t, max_t], got {transition_t}"

        # every net should have pred_type and noise_scheduler
        self.net_pred_type = self.net_s.net_pred_type
        self.noise_scheduler = self.net_s.noise_scheduler

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Forward pass through the two-stage consistency model.

        Args:
            x_t: Noisy input tensor at time t
            t: Time step tensor
            condition: Optional conditioning inputs
            **model_kwargs: Additional keyword arguments passed to networks

        Returns:
            Model output (x0 prediction)

        Raises:
            RuntimeError: If input tensors have mismatched devices
        """
        device = x_t.device

        # Determine which samples are in second stage (t >= transition_t)
        second_stage_mask = t >= self.transition_t

        # Synchronize mask information across distributed ranks for FSDP consistency
        local_all_second_stage = second_stage_mask.all().item()
        local_any_second_stage = second_stage_mask.any().item()

        global_all_second_stage = sync_all(local_all_second_stage, device)
        global_any_second_stage = sync_any(local_any_second_stage, device)

        # Optimization: If all samples are in second stage, only use student network
        if global_all_second_stage:
            return self.net_s(x_t, t, condition=condition, **model_kwargs)

        # Get teacher network predictions (always needed as base)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            with torch.no_grad():
                out_t = self.net_t(x_t, t, condition=condition, **model_kwargs)

        # If some samples are in second stage, get student predictions and blend
        if global_any_second_stage:
            out_s = self.net_s(x_t, t, condition=condition, **model_kwargs)

            # Replace teacher outputs with student outputs for second stage samples
            out_t[second_stage_mask] = out_s[second_stage_mask]

        return out_t


class TCMModel(CMModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the TCM model
        """
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()

        # instantiate the cm teacher (stage-1 model) (note that we leave the cm teacher in train mode)
        self.cm_teacher = copy.deepcopy(self.net).requires_grad_(False)

        self.net_tcm = TCMPrecond(net_t=self.cm_teacher, net_s=self.net, transition_t=self.config.transition_t)

    def on_train_begin(self, is_fsdp=False):
        super().on_train_begin(is_fsdp=is_fsdp)

        # Handle cm_teacher separately if not in FSDP dict
        if "cm_teacher" not in self.fsdp_dict:
            move_module_to_device(self.cm_teacher, device=self.device, dtype=self.precision, name="cm_teacher")

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the TCM model.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        # Prepare training data and conditions
        real_data, condition, neg_condition = self._prepare_training_data(data)

        t = self.net.noise_scheduler.sample_t(
            real_data.shape[0], **convert_cfg_to_dict(self.sample_t_cfg), device=self.device
        )

        # sample the boundary time step with the `boundary_prob` probability
        num_elements_to_mask = int(t.shape[0] * self.config.boundary_prob)
        if num_elements_to_mask == 0:
            mask_t = torch.rand(t.shape[0], device=t.device) < self.config.boundary_prob
        else:
            indices = torch.randperm(t.shape[0])
            mask_indices = indices[:num_elements_to_mask]
            mask_t = torch.zeros(t.shape[0], device=t.device, dtype=torch.bool)
            mask_t[mask_indices] = True
            assert (
                mask_t.sum().item() == num_elements_to_mask
            ), f"Mask has {mask_t.sum().item()} elements, expected {num_elements_to_mask}"

        # set the boundary time steps to the transition time step
        t[mask_t] = self.config.transition_t + 1e-8
        if (t < self.config.transition_t).all():
            raise RuntimeError("Output will not require grad. Use a smaller transition_t.")

        # assert (
        #     self.net_tcm.net_s.training is self.net_tcm.net_t.training
        # ), f"Teacher has state {self.net_tcm.net_t.training} and student has state {self.net_tcm.net_s.training}"
        cm_loss, loss_unweighted, D_yt = self._compute_cm_loss(
            self.net_tcm, real_data=real_data, t=t, condition=condition, neg_condition=neg_condition
        )

        # get boundary loss
        assert (
            cm_loss.shape == t.shape == mask_t.shape
        ), f"cm_loss.shape: {cm_loss.shape}, t.shape: {t.shape}, mask_t.shape: {mask_t.shape}"
        loss_boundary = cm_loss[mask_t].mean()
        cm_loss = cm_loss[~mask_t].mean()
        loss_unweighted = loss_unweighted[~mask_t].mean()

        loss = cm_loss + self.config.w_boundary * loss_boundary
        loss_map = {
            "total_loss": loss,
            "cm_loss": cm_loss,
            "loss_boundary": loss_boundary,
            "unweighted_cm_loss": loss_unweighted,
        }
        outputs = self._get_outputs(D_yt, condition=condition)

        return loss_map, outputs

    @property
    def fsdp_dict(self):
        """Return dict containing all networks to be sharded."""
        model_dict = super().fsdp_dict
        if self.config.add_teacher_to_fsdp_dict and hasattr(self, "cm_teacher"):
            model_dict["cm_teacher"] = self.cm_teacher
        return model_dict
