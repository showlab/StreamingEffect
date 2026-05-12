# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The implementation is based on the EDM2 repo."""

from typing import Set, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.distributed.fsdp import fully_shard

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES

from fastgen.networks.EDM.network import precond_input, precond_output
from fastgen.utils import expand_like

# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:

    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    dtype = torch.float64 if x.dtype is torch.float64 else torch.float32
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        dtype = torch.float64 if x.dtype is torch.float64 else torch.float32
        y = x.to(dtype)
        # Move buffers to input device and dtype
        freqs = self.freqs.to(device=x.device, dtype=dtype)
        phases = self.phases.to(device=x.device, dtype=dtype)
        y = y.ger(freqs)
        y = y + phases
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


@torch.no_grad()
def normalize_weights(m):
    if isinstance(m, MPConv):
        m.weight.copy_(normalize(m.weight))


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        dtype = torch.float64 if x.dtype is torch.float64 else torch.float32
        w = self.weight.to(dtype)
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=256,  # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([1]))
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).


class EMD2UNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Image channels.
        label_dim,  # Class label dimensionality. 0 = unconditional.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,
        # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,
        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[16, 8],  # List of resolutions with self-attention.
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        dropout=0,  # Dropout probability.
        dropout_resolutions=None,  # List of resolutions at which to apply dropout. None = all resolutions.
        embedding_type="mp_fourier",  # time embedding types: ["mp_fourier", "positional"]
        mp_fourier_bandwidth=1.0,  # bandwidth for the mp_fourier embedding
        r_timestep=False,  # Flag for taking target time r
        **block_kwargs,  # Arguments for Block.
    ):
        assert embedding_type in ["mp_fourier", "positional"]

        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.dropout = dropout
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([1]))

        # Embedding.
        self.emb_fourier = (
            PositionalEmbedding(cnoise, endpoint=True)
            if embedding_type == "positional"
            else MPFourier(cnoise, bandwidth=mp_fourier_bandwidth)
        )
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])

        if r_timestep:
            self.emb_fourier_r = (
                PositionalEmbedding(cnoise, endpoint=True)
                if embedding_type == "positional"
                else MPFourier(cnoise, bandwidth=mp_fourier_bandwidth)
            )
            self.emb_noise_r = MPConv(cnoise, cemb, kernel=[])
        else:
            self.emb_fourier_r = None
            self.emb_noise_r = None

        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            dout = self.dropout if (dropout_resolutions is None or res in dropout_resolutions) else 0

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down", dropout=dout, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), dropout=dout, **block_kwargs
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            dout = self.dropout if (dropout_resolutions is None or res in dropout_resolutions) else 0

            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, cemb, flavor="dec", attention=True, dropout=dout, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", dropout=dout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up", dropout=dout, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="dec", attention=(res in attn_resolutions), dropout=dout, **block_kwargs
                )
        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        r_noise_labels=None,
        return_features_early=False,
        feature_indices=None,
    ):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))

        if r_noise_labels is not None:
            if self.r_timestep is not None:
                emb_r = self.emb_noise_r(self.emb_fourier_r(r_noise_labels))
                emb = mp_sum(emb, emb_r, t=0.5)
            else:
                raise ValueError("r_noise_labels provided, but r_timestep is not set")

        if self.emb_label is not None:
            emb = mp_sum(
                emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])).to(emb.dtype), t=self.label_balance
            )
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        idx, features = 0, []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)
            if "block2" in name:
                if idx in feature_indices:
                    features.append(x)
                idx += 1

        # If we have all the features, we can exit early
        if return_features_early:
            assert len(features) == len(feature_indices)
            return features

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)

        if len(feature_indices) == 0:
            # no features requested, return only the model output
            out = x
        else:
            # score and features； score, features
            out = [x, features]
        return out


# ----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.


class EDM2Precond(FastGenNetwork):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Image channels.
        label_dim,  # Class label dimensionality. 0 = unconditional.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        sigma_shift=0.0,  # Shift sigma during inference as done in ECT.
        logvar_channels=128,  # Intermediate dimensionality for uncertainty estimation.
        drop_precond=None,  # Can be set to 'input'/'output'/'both' to drop the preconditioning of the input/output/both.
        net_pred_type="x0",  # Prediction type for FastGenNetwork
        schedule_type="edm",  # Schedule type for FastGenNetwork
        **model_kwargs,  # Keyword arguments for UNet and noise_scheduler.
    ):
        # Initialize FastGenNetwork with EDM2-specific defaults
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.sigma_shift = sigma_shift
        if drop_precond is not None and drop_precond not in ["input", "output", "both"]:
            raise ValueError(f"drop_precond must be one of 'input', 'output', 'both', or None, got {drop_precond}")
        self.drop_precond = drop_precond
        self.unet = EMD2UNet(
            img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **model_kwargs
        )
        embedding_type = model_kwargs.get("embedding_type", "mp_fourier")
        mp_fourier_bandwidth = model_kwargs.get("mp_fourier_bandwidth", 1.0)
        self.logvar_fourier = (
            PositionalEmbedding(logvar_channels, endpoint=True)
            if embedding_type == "positional"
            else MPFourier(logvar_channels, bandwidth=mp_fourier_bandwidth)
        )
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forced_weight_normalization(self):
        self.apply(normalize_weights)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear and convolutional layers, then applies weight normalization.
        """
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

    def fully_shard(self, **kwargs):
        """Fully shard the EDM2 network for FSDP.

        Note: EDM2 wraps the EMD2UNet with preconditioning.
        The underlying unet has encoder (enc) and decoder (dec) blocks stored as ModuleDicts.
        We shard each Block for optimal memory efficiency.
        """
        # Shard encoder blocks
        for name, block in self.unet.enc.items():
            if isinstance(block, Block):
                fully_shard(block, **kwargs)

        # Shard decoder blocks
        for name, block in self.unet.dec.items():
            if isinstance(block, Block):
                fully_shard(block, **kwargs)

        # Shard the entire unet
        fully_shard(self.unet, **kwargs)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x_t.device, dtype=x_t.dtype)
            if condition is None
            else condition.reshape(-1, self.label_dim)
        )

        # Preconditioning weights for input
        x_t_in, t_in = x_t, t
        if self.drop_precond not in ["input", "both"]:
            x_t, t, r = precond_input(x_t, t, r=r, sigma_data=self.sigma_data, eps=self.noise_scheduler.clamp_min)

        t = t.to(x_t.dtype)
        if r is not None:
            r = r.to(x_t.dtype)

        out = self.unet(
            x_t,
            t,
            class_labels=class_labels,
            r_noise_labels=r,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            **fwd_kwargs,
        )

        if return_features_early:
            return out

        sigma_shift = None if self.training else self.sigma_shift
        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            if self.drop_precond not in ["output", "both"]:
                out = precond_output(out, x_t_in, t_in, sigma_shift=sigma_shift, sigma_data=self.sigma_data)
            out = self.noise_scheduler.convert_model_output(
                x_t_in, out, t_in, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            assert isinstance(out, list)
            if self.drop_precond not in ["output", "both"]:
                out[0] = precond_output(out[0], x_t_in, t_in, sigma_shift=sigma_shift, sigma_data=self.sigma_data)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t_in, out[0], t_in, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(t)).reshape(-1, 1)
            return out, logvar  # u(sigma) in Equation 21
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using EDM2's deterministic Euler sampler.

        Args:
            noise: Initial noise tensor [B, C, H, W] (should be scaled by max sigma).
            condition: Class label conditioning (one-hot or class indices).
            neg_condition: Negative conditioning for CFG.
            guidance_scale: CFG guidance scale. None disables guidance.
            num_steps: Number of sampling steps.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples in latent space.
        """
        assert self.schedule_type == "edm", f"{self.schedule_type} is not supported"

        # Get sigma schedule from noise scheduler (in EDM, t == sigma)
        sigmas = self.noise_scheduler.get_t_list(num_steps, device=noise.device)

        x = self.noise_scheduler.latents(noise=noise, t_init=sigmas[0])
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            # Expand sigma for batch
            t = sigma.expand(x.shape[0])

            # Get x0 prediction with optional CFG
            if guidance_scale is not None and guidance_scale > 1.0 and neg_condition is not None:
                # CFG: predict with both conditions
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t, t], dim=0)
                cond_input = torch.cat([neg_condition, condition], dim=0)

                x0_pred = self(x_input, t_input, condition=cond_input, fwd_pred_type="x0")
                x0_uncond, x0_cond = x0_pred.chunk(2)
                x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0_pred = self(x, t, condition=condition, fwd_pred_type="x0")

            # EDM Euler step: x_next = x0 + sigma_next * (x - x0) / sigma
            # Equivalent to: x_next = x + (sigma_next - sigma) * (x - x0) / sigma
            d = (x - x0_pred) / expand_like(t, x)  # derivative estimate
            x = x + (sigma_next - sigma).to(x.dtype) * d

        return x


# ----------------------------------------------------------------------------
