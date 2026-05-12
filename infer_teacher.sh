#!/usr/bin/env bash
# Batch inference with the bidirectional teacher model (Stage 0).
# Supports single-GPU and multi-GPU (torchrun). Skips already-generated outputs
# automatically, so rerunning is safe (resume-friendly).
#
# Usage:
#   # Single GPU
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash infer_teacher.sh
#
#   # Multi-GPU (default 4)
#   bash infer_teacher.sh
#
#   # Custom GPU count
#   NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash infer_teacher.sh
#
#   # Specify checkpoint
#   CKPT=/path/to/step=8000.ckpt bash infer_teacher.sh

set -e
export PYTHONPATH=$(pwd)

NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29504}
CKPT=${CKPT:-/path/to/stage0.ckpt}   # set to your checkpoint path or download from HuggingFace
CONFIG=${CONFIG:-configs/infer_teacher.yaml}
SEED=${SEED:-2542}

mkdir -p outputs
LOG_FILE=outputs/infer_teacher.log

echo "[run] NUM_GPUS=${NUM_GPUS}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] CKPT=${CKPT}"
echo "[run] CONFIG=${CONFIG}"

if [ "${NUM_GPUS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python src/wan2_inference_testset.py \
        --config=${CONFIG} \
        --seed=${SEED} \
        --ckpt_path ${CKPT} \
        2>&1 | tee -a ${LOG_FILE}
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    torchrun --master-port ${MASTER_PORT} --nproc_per_node=${NUM_GPUS} src/wan2_inference_testset.py \
        --config=${CONFIG} \
        --seed=${SEED} \
        --ckpt_path ${CKPT} \
        2>&1 | tee -a ${LOG_FILE}
fi
