#!/usr/bin/env bash
# Train the bidirectional teacher model (Stage 0).
#
# The teacher uses full bidirectional attention over the entire sequence
# and is trained with a rectified-flow objective on reference-conditioned
# in-context video editing (source video + reference keyframe → target video).
#
# Requirements:
#   - Download Wan2.2-TI2V-5B-Diffusers from HuggingFace
#   - Prepare VideoEffect-130K dataset and set dataset_roots in configs/train_teacher.yaml
#   - Set model_id in configs/train_teacher.yaml to your local path or HF model ID
#
# Usage:
#   # 8-GPU training (recommended)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_teacher.sh
#
#   # 4-GPU training
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_teacher.sh
#
#   # Resume from checkpoint
#   CKPT_PATH=/path/to/checkpoint.ckpt bash train_teacher.sh

set -eo pipefail
export PYTHONPATH=$(pwd)

CONFIG=${CONFIG:-configs/train_teacher.yaml}
CKPT_PATH=${CKPT_PATH:-}  # leave empty to train from scratch, or set to resume

CMD="python src/wan2_trainer_plus.py --config=${CONFIG} --seed=1234"
if [ -n "${CKPT_PATH}" ]; then
    CMD="${CMD} --ckpt_path ${CKPT_PATH}"
fi

eval ${CMD} 2>&1 | tee outputs/train_teacher.log
