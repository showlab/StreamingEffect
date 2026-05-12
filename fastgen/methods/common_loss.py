# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Optional

from fastgen.networks.noise_schedule import BaseNoiseSchedule
from fastgen.utils import expand_like


def denoising_score_matching_loss(
    pred_type: str,
    net_pred: torch.Tensor,
    x0: torch.Tensor = None,
    eps: torch.Tensor = None,
    noise_scheduler: Optional[BaseNoiseSchedule] = None,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the denoising diffusion objective.
        Forward process:
            x_t = alpha_t * x_0 + sigma_t * eps

        Note: We currently don't add any loss weighting for simplicity. In the future, we may include
            time-dependent weighting (e.g., by SNR, variance schedule).

    Args:
        pred_type (str): Prediction type: 'x0', 'eps', 'v', 'flow'.
        net_pred (torch.Tensor): The network output, and its meaning is determined by pred_type.
        noise_scheduler (BaseNoiseSchedule): Noise scheduler.
        x0 (torch.Tensor): The clean data x_0.
        eps (torch.Tensor): The epsilon used to compute noised data.
        t (torch.Tensor): The target data t.

    Raises:
        NotImplementedError: If an unknown pred_type is used.

    Returns:
        loss (torch.Tensor): The denoising diffusion loss.
    """
    if pred_type == "x0":
        assert x0 is not None, "x0 cannot be None"
        loss = F.mse_loss(x0, net_pred, reduction="mean")
    elif pred_type == "eps":
        assert eps is not None, "eps cannot be None"
        loss = F.mse_loss(eps, net_pred, reduction="mean")
    elif pred_type == "v":
        assert x0 is not None and eps is not None and t is not None, "x0, eps, t should not be None"
        assert noise_scheduler is not None, "noise_scheduler should not be None"
        alpha_t = expand_like(noise_scheduler.alpha(t), x0).to(device=x0.device, dtype=x0.dtype)
        sigma_t = expand_like(noise_scheduler.sigma(t), x0).to(device=x0.device, dtype=x0.dtype)
        v = alpha_t * eps - sigma_t * x0
        loss = F.mse_loss(v, net_pred, reduction="mean")
    elif pred_type == "flow":
        assert x0 is not None and eps is not None, "x0 and eps cannot be None"
        flow_velocity = eps - x0
        loss = F.mse_loss(flow_velocity, net_pred, reduction="mean")
    else:
        raise NotImplementedError(f"Unknown prediction type {pred_type}")
    return loss


def variational_score_distillation_loss(
    gen_data: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    additional_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the variational score distillation loss.
    Args:
        gen_data (torch.Tensor): generated data
        teacher_x0 (torch.Tensor): x0-prediction from the teacher
        fake_score_x0 (torch.Tensor): x0-prediction from the fake score
        additional_scale (Optional[torch.Tensor]): Additional scale parameter for the VSD loss.

    Returns:
        loss (torch.Tensor): The variational score distillation loss.
    """
    dims = tuple(range(1, teacher_x0.ndim))

    with torch.no_grad():
        # Perform weight calculation in fp32 for numerical stability
        original_dtype = gen_data.dtype
        gen_data_fp32 = gen_data.float()
        teacher_x0_fp32 = teacher_x0.float()

        # Compute weight in fp32 to avoid numerical instability
        diff_abs_mean = (gen_data_fp32 - teacher_x0_fp32).abs().mean(dim=dims, keepdim=True)
        w_fp32 = 1 / (diff_abs_mean + 1e-6)

        # Apply additional scale if provided
        if additional_scale is not None:
            w_fp32 *= expand_like(additional_scale.float(), w_fp32)

        # Convert weight back to original precision
        w = w_fp32.to(dtype=original_dtype)

        vsd_grad = (fake_score_x0 - teacher_x0) * w
        pseudo_target = gen_data - vsd_grad

    loss = 0.5 * F.mse_loss(gen_data, pseudo_target, reduction="mean")
    return loss


def gan_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the GAN loss for the generator
    Args:
        fake_logits (torch.Tensor): The logits for the fake data.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the generator.

    """

    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    gan_loss = F.softplus(-fake_logits).mean()
    return gan_loss


def gan_loss_discriminator(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the GAN loss for the discriminator
    Args:
        real_logits (torch.Tensor): The logits for the real data.
        fake_logits (torch.Tensor): The logits for the fake data.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the discriminator.
    """
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    gan_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()

    return gan_loss
