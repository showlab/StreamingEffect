# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1 Causal SFT — only_effect dataset (effect_608x1080 + people_face_fashi
+ merged high_qual shards under /path/to/videoeffect-130k/wds).

Key delta vs. config_sft_causal_high_qual.py:
  - DATA_ROOT switched to /path/to/videoeffect-130k/wds
  - use_tail_frame_as_ref_fallback = False
  - use_random_target_frame_as_ref = True
        At each step, sample one random latent frame index in [0, T-1] from
        the GT (target_latent) and use it as the single ref. The stored
        ref_latents.pth (tail frame) on disk is ignored. ref_dropout_prob
        is unchanged (0.5), so the random ref is still gated by dropout.
  - max_iter / batch_size / etc. set in the launcher script.
"""

import os
import copy
from omegaconf import DictConfig
import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.net import CausalWan22_V2VBG_5B_Config
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)
from fastgen.callbacks.wandb import WandbCallback
from fastgen.datasets.wds_dataloaders import MultiBucketWDSLoader
from fastgen.utils import LazyCall as L
from fastgen.methods import CausalSFTModel


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


# Bucket weights ≈ expected clip counts (effect/people_face_fashi + merged
# high_qual). Re-tune if actual counts differ after preprocess_only_effect.py
# finishes — print the survey output to read the per-bucket totals.
#   effect_608x1080_243f_30fps   (3860 vids × 2 clips/half) → 736x384  ≈ 7720
#   people_face_fashi_30fps      (3000 vids × 3 clips/half) → 704x736  ≈ 9000
#   high_qual merged shards (counts from config_sft_causal_high_qual.py)
BUCKETS = [
    _bucket("736x384", [48, 25, 46, 24], 7720 + 507),
    _bucket("384x736", [48, 25, 24, 46],         673),
    _bucket("416x736", [48, 25, 26, 46],         438),
    _bucket("704x736", [48, 25, 44, 46], 9000 + 121),
    _bucket("736x736", [48, 25, 46, 46],          47),
    _bucket("640x736", [48, 25, 40, 46],          28),
]


def create_config():
    config = config_sft_default.create_config()
    config.model_class = L(CausalSFTModel)(config=None)
    config.model.fsdp_meta_init = True

    config.trainer.max_iter = 3000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500
    config.model.net_optimizer.lr = 5e-5
    config.model.net_optimizer.weight_decay = 1e-2
    config.model.guidance_scale = 5.0
    config.model.student_sample_steps = 50

    config.model.precision = "bfloat16"

    # Representative shape: 704x736 bucket (largest by clip count after merge)
    config.model.input_shape = [48, 25, 44, 46]

    config.model.net = copy.deepcopy(CausalWan22_V2VBG_5B_Config)
    config.model.net.total_num_frames = config.model.input_shape[1]   # 25
    config.model.net.ref_dropout_prob = 0.5
    config.model.net.delete_cache_on_clear = True

    # ── Random GT frame as ref (replaces tail-frame fallback) ─────────────
    config.model.use_tail_frame_as_ref_fallback = False
    config.model.use_random_target_frame_as_ref = True

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.enable_preprocessors = False

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

    config.log_config.group = "wan22_5b_v2v_bg_sft_causal_only_effect"
    return config
