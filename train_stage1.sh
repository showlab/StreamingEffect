#!/bin/bash
# Stage 1: Bidirectional-to-Causal SFT (Causal Autoregressive Student).
#
# Initializes the causal AR student from the bidirectional teacher checkpoint
# and trains with block-causal attention and heterogeneous per-chunk timesteps.
# Enables streaming (KV-cached) generation.
#
# Requirements:
#   - Merge the teacher LoRA weights into a single transformer .pt file
#     (set MERGED_MODEL below to that merged checkpoint path)
#   - Prepare VideoEffect-130K dataset (same as teacher training)
#   - See fastgen/configs/experiments/WanV2VBG/config_sft_causal_only_effect.py
#     for full configuration options
#
# Usage:
#   bash train_stage1.sh
#
#   # Custom iterations
#   MAX_ITER=3000 bash train_stage1.sh
#
#   # Override model path
#   MERGED_MODEL=/path/to/merged_transformer.pt bash train_stage1.sh

set -eo pipefail

MERGED_MODEL=${MERGED_MODEL:-/path/to/merged_transformer.pt}  # set to merged teacher checkpoint

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-./outputs/}

MAX_ITER=${MAX_ITER:-3000}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 \
    train.py \
    --config fastgen/configs/experiments/WanV2VBG/config_sft_causal_only_effect.py \
    - \
    model.pretrained_model_path="${MERGED_MODEL}" \
    trainer.resume=true \
    trainer.fsdp=true \
    trainer.max_iter=${MAX_ITER} \
    trainer.grad_accum_rounds=1 \
    dataloader_train.batch_size=4 \
    log_config.group=streamingeffect_stage1 \
    "$@"
