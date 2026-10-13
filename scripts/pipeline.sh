#!/usr/bin/env bash
# Этапы пайплайна подряд для одного конфига. Общая логика — конкретные запуски в launch/.
#
#   CONFIG=configs/<имя>.yaml GPU=0 bash scripts/pipeline.sh                  # все этапы 00–04
#   CONFIG=configs/<имя>.yaml STEPS="02 04" bash scripts/pipeline.sh          # только CPU-этапы
#   CONFIG=configs/<имя>.yaml STEPS=01 bash scripts/pipeline.sh --assemble-only   # аргументы — одному этапу
#
# Env:
#   CONFIG         обязателен;
#   CONFIG_SHA256  если задан — конфиг должен иметь ровно этот хэш (так launch/ фиксирует запуск);
#   STEPS          этапы по порядку (по умолчанию "00 01 02 03 04");
#   GPU            индекс карты, обязателен, если среди STEPS есть GPU-этап (01, 03);
#   FORK, HF_HOME  пути к форку и HF-кэшу.
# Ни одна из переменных, кроме CONFIG, на числа не влияет.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

: "${CONFIG:?укажи CONFIG=configs/<имя>.yaml}"
[[ -f "$CONFIG" ]] || { echo "нет конфига $CONFIG" >&2; exit 1; }
if [[ -n "${CONFIG_SHA256:-}" ]]; then
  actual=$(sha256sum "$CONFIG" 2>/dev/null || shasum -a 256 "$CONFIG")
  actual=${actual%% *}
  if [[ "$actual" != "$CONFIG_SHA256" ]]; then
    echo "конфиг $CONFIG изменён после фиксации запуска: sha256 $actual, ожидался $CONFIG_SHA256" >&2
    echo "если правка осознанная — обнови CONFIG_SHA256 в launch/ или заведи новый конфиг и запуск" >&2
    exit 1
  fi
fi

STEPS="${STEPS:-00 01 02 03 04}"
read -ra STEP_LIST <<< "$STEPS"
if (( $# > 0 && ${#STEP_LIST[@]} != 1 )); then
  echo "аргументы ($*) можно передать только одному этапу: STEPS=<этап>" >&2; exit 1
fi
for STEP in "${STEP_LIST[@]}"; do
  if [[ "$STEP" == 01 || "$STEP" == 03 ]] && [[ " $* " != *" --assemble-only "* ]]; then
    : "${GPU:?этап $STEP требует GPU=<индекс>}"
  fi
done

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$HF_HOME/hub}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; else export CUDA_VISIBLE_DEVICES=""; fi

for STEP in "${STEP_LIST[@]}"; do
  SCRIPT=$(ls scripts/${STEP}_*.py)
  echo "================ $SCRIPT — $(date '+%F %T') — config=$CONFIG GPU=${GPU:-—} ================"
  uv run python "$SCRIPT" --config "$CONFIG" "$@"
done
echo "================ готово — $(date '+%F %T') ================"
