#!/usr/bin/env bash
# Этап 08 (разложение DIM) на 8 моделях: Qwen2.5-3B — свой запуск; остальные 7 — DIM и RDO,
# обученные boroday (импорт scripts/import_run.py), затем 01 --assemble-only, 02 и 08.
# Готовое пропускается: импорт — если папка результатов есть, 08 — если есть decompose.json.
#
#   GPU=1 bash scripts/tmux.sh scripts/cross_model_decompose.sh
#   GPU=1 MODELS="llama3.2-1b" bash scripts/cross_model_decompose.sh    # только часть моделей
#
# Env: GPU (обязателен), MODELS (по умолчанию все, от малых к большим), SRC (папка results boroday).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?укажи GPU=<индекс>}"
SRC="${SRC:-$HOME/l.boroday/shared-refusal-component/results}"
export HF_HUB_OFFLINE=1          # всё из кэша: у Llama-3.1 у токена нет доступа к репозиторию
declare -A ALIAS=([qwen2.5-1.5b]=Qwen2.5-1.5B-Instruct [qwen2.5-7b]=Qwen2.5-7B-Instruct
                  [qwen2.5-14b]=Qwen2.5-14B-Instruct [llama3.2-1b]=Llama-3.2-1B-Instruct
                  [llama3.2-3b]=Llama-3.2-3B-Instruct [llama3-8b]=Meta-Llama-3-8B-Instruct
                  [llama3.1-8b]=Llama-3.1-8B-Instruct)
MODELS="${MODELS:-qwen2.5-3b llama3.2-1b qwen2.5-1.5b llama3.2-3b qwen2.5-7b llama3-8b llama3.1-8b qwen2.5-14b}"

for name in $MODELS; do
  run=${name}_salad-o1o2o5o6
  cfg=configs/$run.yaml
  echo "################ $run — $(date '+%F %T') ################"
  if [[ -f results/$run/decompose/decompose.json ]]; then echo "08 уже посчитан, пропуск"; continue; fi
  if [[ "$name" != qwen2.5-3b ]]; then
    [[ -d results/$run ]] || uv run python scripts/import_run.py --config "$cfg" --src "$SRC/${ALIAS[$name]}"
    CONFIG=$cfg STEPS=01 bash scripts/pipeline.sh --assemble-only
    CONFIG=$cfg STEPS=02 bash scripts/pipeline.sh
  fi
  CONFIG=$cfg STEPS=08 GPU=$GPU bash scripts/pipeline.sh
done
echo "################ готово — $(date '+%F %T') ################"
