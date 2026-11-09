#!/usr/bin/env bash
# Этап 14 (read при одной теме) для моделей по очереди на одной карте; офлайн — модели и XSTest уже
# в HF-кэше сервера. Ошибка одной модели не останавливает очередь.
#
#   GPU=3 MODELS="qwen2.5-1.5b qwen2.5-3b" bash run/read.sh
#   GPU=1,4 HF_MAX_MEMORY="0:12GiB,1:70GiB" MODELS=qwen2.5-32b bash run/read.sh   # 32B на двух картах, первая занята
#
# Ход — в results/piles/read-<GPU>.status: started / ok <модель> / FAIL <модель> / finished.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?укажи GPU=<индекс>}"
MODELS="${MODELS:-qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen2.5-14b qwen2.5-32b llama3.2-1b llama3.2-3b llama3-8b llama3.1-8b gemma2-9b}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STATUS=results/piles/read-$GPU.status
mkdir -p results/piles
echo "started $(date '+%F %T') GPU=$GPU models=[$MODELS]" >> "$STATUS"
for name in $MODELS; do
  echo "################ $name — $(date '+%F %T') — GPU $GPU ################"
  if CONFIG=configs/${name}_salad-o1o2o5o6.yaml STEPS=14 GPU=$GPU bash scripts/pipeline.sh; then
    echo "ok $name $(date '+%F %T')" >> "$STATUS"
  else
    echo "FAIL $name $(date '+%F %T')" >> "$STATUS"
  fi
done
echo "finished $(date '+%F %T')" >> "$STATUS"
