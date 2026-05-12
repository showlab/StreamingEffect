# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 3 Self-Forcing — only_effect dataset.

Same architecture / loss / 4-step schedule as config_self_forcing_multibucket.py;
key deltas:
  - DATA_ROOT switched to /path/to/videoeffect-130k/wds
  - BUCKETS match the only_effect WDS layout (same as Stage 1 only_effect config)
  - use_tail_frame_as_ref_fallback = False
  - use_random_target_frame_as_ref = True
        Per step, sample one random latent frame index in [0, T-1] from
        target_latent (= GT / output-half latent) and use it as the ref.
        ref_dropout_prob stays 0.0 (Stage 3 baseline).
  - max_iter = 3000, save_ckpt_iter = 500, batch_size = 4

Each bucket must share total_num_frames=25 / chunk_size=5; only spatial differs.
"""

import os
import copy
from omegaconf import DictConfig
import fastgen.configs.methods.config_self_forcing as config_self_forcing_default
from fastgen.configs.net import CausalWan22_V2VBG_5B_Config, Wan22_V2VBG_5B_Config
from fastgen.configs.discriminator import Discriminator_Wan22_5B_Config
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from fastgen.callbacks.wandb import WandbCallback
from fastgen.datasets.wds_dataloaders import MultiBucketWDSLoader
from fastgen.utils import LazyCall as L


DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/videoeffect-130k/wds")

SHARED_KEY_MAP = {
    "real":              "target_latent.pth",
    "condition":         "txt_emb.pth",
    "foreground_latent": "foreground_latent.pth",
    "ref_latents":       "ref_latents.pth",
    "ref_mask":          "ref_mask.pth",
    "ref_frame_indices": "ref_frame_indices.json",
}


def _bucket(name: str, shape, weight):
    path = f"{DATA_ROOT}/{name}"
    return {
        "datatags":  [f"WDS:{path}"],
        "files_map": {"neg_condition": f"{path}/neg_txt_emb.npy"},
        "shape":     list(shape),
        "weight":    float(weight),
    }


# Bucket weights ≈ post-merge clip counts (effect/people_face_fashi + high_qual)
BUCKETS = [
    _bucket("736x384", [48, 25, 46, 24], 7720 + 507),
    _bucket("384x736", [48, 25, 24, 46],         673),
    _bucket("416x736", [48, 25, 26, 46],         438),
    _bucket("704x736", [48, 25, 44, 46], 9000 + 121),
    _bucket("736x736", [48, 25, 46, 46],          47),
    _bucket("640x736", [48, 25, 40, 46],          28),
]


def create_config():
    config = config_self_forcing_default.create_config()
    config.model.fsdp_meta_init = True

    config.trainer.max_iter = 3000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-6
    config.model.discriminator_optimizer.lr = 1e-6
    config.model.fake_score_optimizer.lr = 5e-6

    config.model.precision = "bfloat16"

    config.model.input_shape = [48, 25, 44, 46]

    # Student: causal AR V2VBG (loaded from Stage 1 only_effect SFT)
    config.model.net = copy.deepcopy(CausalWan22_V2VBG_5B_Config)
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.net.ref_dropout_prob = 0.0
    config.model.net.delete_cache_on_clear = True

    # Teacher: original bidirectional DiT (frozen)
    config.model.teacher = copy.deepcopy(Wan22_V2VBG_5B_Config)

    # ── Random GT frame as ref (replaces tail-frame fallback) ─────────────
    config.model.use_tail_frame_as_ref_fallback = False
    config.model.use_random_target_frame_as_ref = True

    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 7.0
    config.model.student_sample_type = "ode"
    config.model.enable_preprocessors = False

    # GAN
    config.model.gan_loss_weight_gen = 0.002
    config.model.student_update_freq = 2
    config.model.gan_r1_reg_weight = 10.0
    config.model.gan_r1_reg_alpha = 0.1
    config.model.discriminator = copy.deepcopy(Discriminator_Wan22_5B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_use_same_t_noise = True

    # Self-Forcing specific
    config.model.context_noise = 0.0

    # 4-step schedule
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.dataloader_train = L(MultiBucketWDSLoader)(
        buckets=BUCKETS,
        batch_size=4,
        shared_key_map=SHARED_KEY_MAP,
        num_workers=1,
        shuffle_size=200,
        seed=0,
    )

    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            "wandb": L(WandbCallback)(sample_logging_iter=100),
        }
    )

    config.log_config.group = "wan22_5b_v2v_bg_self_forcing_only_effect"
    return config
