# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable, Optional
import torch

from fastgen.methods import FastGenModel
from fastgen.methods.common_loss import denoising_score_matching_loss
from fastgen.utils import expand_like
from fastgen.utils import basic_utils

if TYPE_CHECKING:
    from fastgen.configs.methods.config_sft import ModelConfig
    from fastgen.networks.network import FastGenNetwork


class SFTModel(FastGenModel):
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): The configuration for the DMD model
        """
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()
        self.load_student_weights_and_ema()

    def _mix_condition(self, condition, neg_condition):
        """
        Mix condition with neg_condition based on cond_dropout_prob.
        Works for torch.Tensor or dict[str, torch.Tensor].

        Args:
            condition: torch.Tensor or dict of tensors
            neg_condition: same type/structure as condition
        """
        if self.config.cond_dropout_prob is None:
            return condition

        if isinstance(condition, torch.Tensor):
            batch_size = condition.shape[0]
            sample_mask = (
                torch.rand(batch_size, device=condition.device, dtype=condition.dtype) >= self.config.cond_dropout_prob
            )
            mask = expand_like(sample_mask, condition)
            return torch.where(mask, condition, neg_condition)

        elif isinstance(condition, dict):
            # Pick one tensor to generate the mask
            keys_no_drop = self.config.cond_keys_no_dropout
            assert set(keys_no_drop).issubset(
                condition.keys()
            ), f"keys_no_drop: {keys_no_drop} not in {condition.keys()}"

            # Generate per-sample mask once, reuse for all keys
            first_key = next(iter(condition.keys() - keys_no_drop), None)
            if first_key is not None:
                tensor = condition[first_key]
                batch_size = tensor.shape[0]
                sample_mask = (
                    torch.rand(batch_size, device=tensor.device, dtype=tensor.dtype) >= self.config.cond_dropout_prob
                )

                condition = condition.copy()
                for k in condition.keys() - keys_no_drop:
                    mask = expand_like(sample_mask, condition[k])
                    condition[k] = torch.where(mask, condition[k], neg_condition[k])
            return condition

        else:
            raise TypeError(f"Unsupported type: {type(condition)}")

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            precision_amp=self.precision_amp_infer,
            guidance_scale=self.config.guidance_scale,
            num_steps=self.config.student_sample_steps,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise}

    @classmethod
    def generator_fn(
        cls,
        net: FastGenNetwork,
        noise: torch.Tensor,
        precision_amp: Optional[torch.dtype] = None,
        **kwargs,
    ):
        assert hasattr(net, "sample"), "net must have the sample() method"
        with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise.device.type):
            x = net.sample(noise, **kwargs)
        return x.to(dtype=noise.dtype)

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for supervised finetuning (sft)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        real_data, condition, neg_condition = self._prepare_training_data(data)
        batch_size = real_data.shape[0]

        # Perturb the data according to the underlying time sampling type
        t = self.net.noise_scheduler.sample_t(
            batch_size,
            **basic_utils.convert_cfg_to_dict(self.config.sample_t_cfg),
            device=self.device,
        )
        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)

        # replace condition with neg_condition with a probability for training
        # but keep original condition for _get_outputs (sampling uses original condition)
        condition_train = self._mix_condition(condition, neg_condition)

        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps, t)
        net_pred = self.net(noisy_real_data, t, condition=condition_train)

        loss = denoising_score_matching_loss(
            self.net.net_pred_type,
            net_pred=net_pred,
            noise_scheduler=self.net.noise_scheduler,
            x0=real_data,
            eps=eps,
            t=t,
        )

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "dsm_loss": loss,
        }

        outputs = self._get_outputs(net_pred, condition=condition, neg_condition=neg_condition)

        return loss_map, outputs


class CausalSFTModel(SFTModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
        neg_condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        context_noise = getattr(self.config, "context_noise", 0)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            precision_amp=self.precision_amp_infer,
            guidance_scale=self.config.guidance_scale,
            sample_steps=self.config.student_sample_steps,
            context_noise=context_noise,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for supervised finetuning (sft)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss (torch.Tensor): total loss
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """

        real_data, condition, neg_condition = self._prepare_training_data(data)
        batch_size = real_data.shape[0]

        # Support multi-resolution ("bucket") training: trust real_data.shape over
        # the static config input_shape. Invalidate the transformer block_mask
        # (cached by spatial size/frame_seqlen) whenever the shape changes.
        input_shape = list(real_data.shape[1:])
        last_shape = getattr(self, "_last_input_shape", None)
        if last_shape != input_shape:
            transformer = getattr(self.net, "transformer", None)
            if transformer is not None and hasattr(transformer, "block_mask"):
                transformer.block_mask = None
            self._last_input_shape = input_shape

        # Add noise to real image data (for multistep generation)
        eps_inhom = torch.randn(batch_size, *input_shape, device=self.device, dtype=real_data.dtype)
        assert hasattr(
            self.net.noise_scheduler, "sample_t_inhom_sft"
        ), "net.noise_scheduler does not have the sample_t_inhom_sft() method"
        t_inhom = self.net.noise_scheduler.sample_t_inhom_sft(
            batch_size,
            input_shape[1],
            self.net.chunk_size,
            **basic_utils.convert_cfg_to_dict(self.config.sample_t_cfg),
            device=self.device,
        )
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # shape [B, 1, T, 1, 1]
        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps_inhom, t_inhom_expanded)

        # replace condition with neg_condition with a probability for training
        # but keep original condition for _get_outputs (sampling uses original condition)
        condition_train = self._mix_condition(condition, neg_condition)

        net_pred = self.net(noisy_real_data, t_inhom, condition=condition_train)
        loss = denoising_score_matching_loss(
            self.net.net_pred_type,
            net_pred=net_pred,
            noise_scheduler=self.net.noise_scheduler,
            x0=real_data,
            eps=eps_inhom,
            t=t_inhom,
        )

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "dsm_loss": loss,
        }

        outputs = self._get_outputs(net_pred, condition=condition, neg_condition=neg_condition)

        return loss_map, outputs
