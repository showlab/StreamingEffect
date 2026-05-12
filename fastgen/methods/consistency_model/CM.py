# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple, Optional

import torch

from fastgen.methods import FastGenModel
import fastgen.utils.logging_utils as logger
from fastgen.utils import expand_like
from fastgen.utils.basic_utils import convert_cfg_to_dict

if TYPE_CHECKING:
    from fastgen.configs.methods.config_cm import ModelConfig
    from fastgen.networks.network import FastGenNetwork
    from fastgen.networks.noise_schedule import BaseNoiseSchedule


def get_edm_c_out(noise_scheduler, t, sigma_data=0.5):
    # SNR matching to EDM schedule
    edm_sigma = noise_scheduler.sigma(t) / noise_scheduler.non_zero_clamp(noise_scheduler.alpha(t))
    return edm_sigma * sigma_data / (edm_sigma**2 + sigma_data**2).sqrt()


def t_to_r_sigmoid(t, ratio, min_r=1e-6):
    # Sigmoid delta t proposed by the ECT paper (https://github.com/locuslab/ect)
    # The effective ratio is an increasing function w.r.t t
    r = t - t * (1 - ratio) * (1 + 8 * torch.sigmoid(-t))
    r = torch.clamp(r, min=min_r)
    return r


@torch.no_grad()
def ode_solver(net, x_t, t, t_next, condition=None, neg_condition=None, guidance_scale=None, skip_layers=None):
    # TODO: may consider adding other prediction types,
    #  since it might be numerically more stable to directly use net_pred_type.
    flow_pred = net(x_t, t, condition=condition, fwd_pred_type="flow")
    if guidance_scale is not None:
        kwargs = {"condition": neg_condition, "fwd_pred_type": "flow"}
        if skip_layers is not None:
            kwargs["skip_layers"] = skip_layers
        # CFG guidance
        flow_pred_neg = net(x_t, t, **kwargs)
        flow_pred = flow_pred + (guidance_scale - 1) * (flow_pred - flow_pred_neg)

    delta_t = expand_like(t - t_next, x_t).to(x_t)
    x_t_next = x_t - delta_t * flow_pred
    return x_t_next


class CMModel(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the CM model
        """
        # Must set this BEFORE super().__init__() because build_model() is called there,
        if config.add_teacher_to_fsdp_dict and not config.loss_config.use_cd:
            logger.warning("`add_teacher_to_fsdp_dict` will be set to False, since no teacher network is instantiated.")
            config.add_teacher_to_fsdp_dict = False

        super().__init__(config)
        self.config = config
        self.sample_t_cfg = self.config.sample_t_cfg
        self.loss_config = self.config.loss_config
        self.ratio = 0.0
        if (
            getattr(self.sample_t_cfg, "min_r", None) is not None
            and self.sample_t_cfg.min_r < self.net.noise_scheduler.min_t
        ):
            logger.warning(
                f"sample_t_cfg.min_r {self.sample_t_cfg.min_r} is smaller than net.noise_scheduler.min_t {self.net.noise_scheduler.min_t} and will be set to {self.net.noise_scheduler.min_t}"
            )

    def build_model(self):
        super().build_model()
        if self.config.loss_config.use_cd:
            self.build_teacher()
        self.load_student_weights_and_ema()

    def _compute_cm_loss(
        self,
        net: FastGenNetwork,
        real_data: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the consistency model loss
        Args:
            net (torch.nn.Module): The consistency model network
            real_data (torch.Tensor): The real data
            t (torch.Tensor): The time step for student net
            condition (Any): The condition to generate
            neg_condition (Any): The negative condition to generate (default: None)
        Return:
            cm_loss (torch.Tensor): The consistency model loss
            loss_unweighted (torch.Tensor): The unweighted cm loss
            D_yt (torch.Tensor): The denoised output of the student net
        """
        # sample r from a pre-defined mapping function
        min_r = max(net.noise_scheduler.min_t, self.sample_t_cfg.min_r)
        r = t_to_r_sigmoid(t, self.ratio, min_r=min_r)

        # safety check: adjust t values where r > t - eps
        mask = r >= (t - net.noise_scheduler.clamp_min)
        if mask.any():
            logger.warning(
                f"Adjusting {mask.sum()} t values where r > t - eps. Adjust sample_t_cfg.min_t/min_r to avoid this."
            )
            t = torch.where(mask, r + net.noise_scheduler.clamp_min, t)

        if self.sample_t_cfg.quantize:
            # quantize t and r
            # get sigma_t and sigma_r
            sigma_t = net.noise_scheduler.sigma(t)
            sigma_r = net.noise_scheduler.sigma(r)

            # find indices of t and r
            t_idx = net.noise_scheduler.closest_sigma_idx(sigma_t)
            r_idx = net.noise_scheduler.closest_sigma_idx(sigma_r)

            # correct same indices
            same_idx = t_idx == r_idx
            large_idx = t_idx == net.noise_scheduler.num_steps - 1
            t_idx[same_idx & ~large_idx] += 1
            r_idx[same_idx & large_idx] -= 1

            # map back to t and r using sigma_idx_to_t
            t = net.noise_scheduler.sigma_idx_to_t(t_idx).to(t)
            r = net.noise_scheduler.sigma_idx_to_t(r_idx).to(r)

        # Shared noise direction
        eps = torch.randn_like(real_data)
        y_t = net.noise_scheduler.forward_process(real_data, eps, t)
        if self.loss_config.use_cd:
            assert self.teacher is not None, "There must exist a teacher network when use_cd=True"
            # consistency distillation (CD)
            y_r = ode_solver(
                self.teacher,
                x_t=y_t,
                t=t,
                t_next=r,
                condition=condition,
                neg_condition=neg_condition,
                guidance_scale=self.config.guidance_scale,
                skip_layers=self.config.skip_layers,
            )
        else:
            # consistency training (CT)
            y_r = net.noise_scheduler.forward_process(real_data, eps, r)

        # Shared Dropout Mask with FSDP-safe RNG handling
        assert (
            y_r.dtype == y_t.dtype == real_data.dtype
        ), f"y_r.dtype: {y_r.dtype}, y_t.dtype: {y_t.dtype}, real_data.dtype: {real_data.dtype}"
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            D_yt = net(y_t, t, condition=condition, fwd_pred_type="x0")
        with torch.no_grad():
            D_yr_candidate = net(y_r, r, condition=condition, fwd_pred_type="x0")

        mask = r > 0
        mask = expand_like(mask, real_data)  # Expand mask to match tensor dimensions
        if not D_yr_candidate.isfinite().all():
            logger.warning("Predicted output contains non-finite values. Adjusting them using torch.nan_to_num.")
        D_yr = mask * torch.nan_to_num(D_yr_candidate) + (~mask) * real_data
        assert (
            D_yt.shape == D_yr.shape == real_data.shape
        ), f"D_yt.shape: {D_yt.shape}, D_yr.shape: {D_yr.shape}, real_data.shape: {real_data.shape}"

        cm_loss, loss_unweighted = self._pred_to_loss(
            noise_scheduler=net.noise_scheduler, D_yt=D_yt, D_yr=D_yr, t=t, r=r
        )
        return cm_loss, loss_unweighted, D_yt

    def _pred_to_loss(
        self,
        noise_scheduler: BaseNoiseSchedule,
        D_yt: torch.Tensor,
        D_yr: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # perform loss computations in higher precision
        dtype = torch.float64
        t = t.to(dtype)
        r = r.to(dtype)

        # L2 Loss
        l2_distance = torch.linalg.vector_norm(D_yt - D_yr, dim=tuple(range(1, D_yt.ndim)), keepdim=False, dtype=dtype)

        # Huber Loss if needed
        c = self.loss_config.huber_const
        if c > 0:
            loss_unweighted = torch.sqrt(l2_distance**2 + c**2) - c
        elif self.loss_config.use_squared_l2:
            loss_unweighted = l2_distance**2
        else:
            loss_unweighted = l2_distance

        weighting = self.loss_config.weighting_ct_loss
        assert (
            loss_unweighted.shape == t.shape == r.shape
        ), f"loss_unweighted.shape: {loss_unweighted.shape}, t.shape: {t.shape}, r.shape: {r.shape}"
        if weighting == "default":
            cm_loss = loss_unweighted / (t - r)
        elif weighting == "c_out":
            cm_loss = loss_unweighted / get_edm_c_out(noise_scheduler, t)
        elif weighting == "c_out_sq":
            cm_loss = loss_unweighted / get_edm_c_out(noise_scheduler, t) ** 2
        elif weighting == "sigma_sq":
            cm_loss = loss_unweighted / noise_scheduler.sigma(t) ** 2
        elif weighting == "sqrt":
            cm_loss = loss_unweighted / (t - r) ** 0.5
        elif weighting == "one":
            cm_loss = loss_unweighted
        else:
            raise ValueError(
                f"Weighting function {weighting} not implemented. Will use the default weighting function."
            )
        return cm_loss, loss_unweighted

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        # callable to sample a batch of generated data
        noise = torch.randn_like(gen_data, dtype=self.precision)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=self.config.student_sample_type,
            t_list=self.config.sample_t_cfg.t_list,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for the CM model.

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
        cm_loss, loss_unweighted, D_yt = self._compute_cm_loss(
            net=self.net, real_data=real_data, t=t, condition=condition, neg_condition=neg_condition
        )

        loss = cm_loss.mean()
        loss_map = {
            "total_loss": loss,
            "cm_loss": loss,
            "unweighted_cm_loss": loss_unweighted.mean(),
        }
        outputs = self._get_outputs(D_yt, condition=condition)

        return loss_map, outputs
