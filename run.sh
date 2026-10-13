#!/usr/bin/env bash
# Запуск одного шага пайплайна.
#
#   bash run.sh 00                                   # CPU
#   CUDA_VISIBLE_DEVICES=0 bash run.sh 01            # GPU
#   CUDA_VISIBLE_DEVICES=0 bash run.sh 01 --domains O1 O2   # RDO только этих доменов
#
# Env: CONFIG (configs/qwen2.5-3b.yaml), FORK (../geometry-of-refusal), HF_HOME.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

STEP="$1"; shift
SCRIPT=$(ls scripts/${STEP}_*.py)
CONFIG="${CONFIG:-configs/qwen2.5-3b.yaml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$HF_HOME/hub}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[$SCRIPT] config=$CONFIG  GPU=${CUDA_VISIBLE_DEVICES:-—}"
uv run python "$SCRIPT" --config "$CONFIG" "$@"
