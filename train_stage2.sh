#!/bin/bash
# Stage 2: On-Policy Self-Forcing for Few-Step Streaming Generation.
#
# Takes the Stage 1 causal student and further distills it into a 4-step
# streaming model using on-policy rollouts with SNR-weighted regression.
# Closes both the few-step gap and the exposure-bias gap simultaneously.
#
# Requirements:
#   - Merge the Stage 1 FSDP checkpoint into a single transformer .pt file
#     (set STUDENT_CKPT below). See scripts/convert_fsdp_checkpoint.py.
#   - The original bidirectional teacher merged checkpoint (TEACHER_CKPT).
#   - VideoEffect-130K dataset (same as Stage 1)
#   - See fastgen/configs/experiments/WanV2VBG/config_self_forcing_only_effect.py
#
# Usage:
#   bash train_stage2.sh
#
#   # Custom iterations
#   MAX_ITER=3000 bash train_stage2.sh

set -eo pipefail

TEACHER_CKPT=${TEACHER_CKPT:-/path/to/merged_transformer.pt}  # bidirectional teacher
STUDENT_CKPT=${STUDENT_CKPT:-/path/to/stage1_net_iter3000.pt}  # merged Stage 1 student

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-./outputs/}

MAX_ITER=${MAX_ITER:-3000}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

echo "=== Stage 2: Self-Forcing ==="
echo "  GPUs:    ${NUM_GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  Teacher: ${TEACHER_CKPT}"
echo "  Student: ${STUDENT_CKPT}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NUM_GPUS} \
    train.py \
    --config fastgen/configs/experiments/WanV2VBG/config_self_forcing_only_effect.py \
    - \
    model.pretrained_model_path="${TEACHER_CKPT}" \
    model.pretrained_student_net_path="${STUDENT_CKPT}" \
    trainer.resume=true \
    trainer.fsdp=true \
    trainer.max_iter=${MAX_ITER} \
    trainer.grad_accum_rounds=1 \
    trainer.save_ckpt_iter=500 \
    dataloader_train.batch_size=4 \
    log_config.group=streamingeffect_stage2 \
    "$@"
