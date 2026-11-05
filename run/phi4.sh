#!/usr/bin/env bash
# Phi-4-mini-instruct × SALAD-Bench (O1, O2, O5, O6) — весь пайплайн с нуля на одной карте:
# скачивание модели (scripts/fetch_model.sh), этапы 00–06 (сплиты, 4 DIM + 20 RDO, общая
# компонента, аблация, анализ, сходимость, ответы на StrongREJECT), затем этап 12 — бенчмарки
# bench.tasks (MMLU, TruthfulQA, ARC-Challenge, GSM8K): датасеты с сетью, сами прогоны офлайн.
# Судья StrongREJECT (этап 07) — отдельно, где есть OpenRouter (с сервера — 403).
#
#   GPU=0 bash run/phi4.sh                                  # всё
#   GPU=0 STEPS="01" bash run/phi4.sh                       # только эти этапы 00–06
#   GPU=0 STEPS=none TASKS="mmlu arc_challenge" bash run/phi4.sh   # только бенчмарки, часть задач
#
# Env:
#   GPU    индекс карты, обязателен;
#   STEPS  этапы до бенчмарков (по умолчанию "00 01 02 03 04 05 06"; none — пропустить);
#   TASKS  подмножество bench.tasks (по умолчанию все; none — без бенчмарков).
# Готовое не пересчитывается — прерванный запуск можно просто повторить.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс свободной карты>}"
CONFIG=configs/phi4-mini_salad-o1o2o5o6.yaml
STEPS="${STEPS:-00 01 02 03 04 05 06}"
TASKS="${TASKS:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/l.boroday/.hf_token}"   # свой токен, не общий
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # номера GPU — как в nvidia-smi

echo "################ phi4-mini — $(date '+%F %T')"
bash scripts/fetch_model.sh "$CONFIG"
if [[ "$STEPS" != none ]]; then
  CONFIG=$CONFIG GPU=$GPU STEPS="$STEPS" bash scripts/pipeline.sh
fi
if [[ "$TASKS" != none ]]; then
  CONFIG=$CONFIG STEPS=12 bash scripts/pipeline.sh --prefetch-only ${TASKS:+--tasks $TASKS}
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CONFIG=$CONFIG STEPS=12 GPU=$GPU \
    bash scripts/pipeline.sh ${TASKS:+--tasks $TASKS}
fi
echo "################ готово — $(date '+%F %T')"
