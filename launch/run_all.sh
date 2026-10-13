#!/usr/bin/env bash
# Весь пайплайн 00 → 04 подряд на ОДНОЙ видимой GPU. Падает на первой ошибке.
# Шаги идемпотентны: 01 пропускает готовые DIM и обученные сиды RDO,
# поэтому после сбоя достаточно перезапустить (можно с середины: START=02).
#
#   GPU=1 bash launch/run_all.sh
#   GPU=1 START=03 bash launch/run_all.sh
#
# Env: GPU (обязателен), START (00), CONFIG.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
export CUDA_VISIBLE_DEVICES="$GPU"      # процессу видна только эта карта
START="${START:-00}"

for STEP in 00 01 02 03 04; do
  [[ "$STEP" < "$START" ]] && continue
  echo "================ шаг $STEP — $(date '+%F %T') — GPU $GPU ================"
  bash run.sh "$STEP"
done
echo "================ готово — $(date '+%F %T') ================"
