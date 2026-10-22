#!/usr/bin/env bash
# Очередь GPU-этапов для группы моделей на одной карте: по каждой модели 10 (кросс-нейтрализация,
# минуты), 06 (ответы на StrongREJECT с DIM union, w и остатком DIM⊥w) и 09 (ответы на безобидные
# запросы при вызове отказа). Судья (07 и для 09) — потом на Mac.
#
#   GPU=2 bash scripts/gpu_pile.sh qwen            # обычно запускает scripts/gpu_watcher.sh
#   GPU=2 MODELS="qwen2.5-7b" bash scripts/gpu_pile.sh qwen
#
# Группы: qwen — Qwen2.5 3B (первым: после него на Mac можно судить п. 3), 1.5B, 7B, 14B;
# llama — Llama-3.2-1B, 3.2-3B, 3-8B, 3.1-8B. Готовое пропускается: 06 и 09 — по готовым файлам
# ответов, 10 — если есть cross/cross.json. Ошибка одной модели не останавливает очередь.
# Ход — в results/piles/<группа>.status: started / ok <модель> / FAIL <модель> / finished.
set -uo pipefail
cd "$(dirname "$0")/.."
PILE="${1:?укажи группу: qwen или llama}"
: "${GPU:?укажи GPU=<индекс>}"
case "$PILE" in
  qwen)  DEFAULT="qwen2.5-3b qwen2.5-1.5b qwen2.5-7b qwen2.5-14b" ;;
  llama) DEFAULT="llama3.2-1b llama3.2-3b llama3-8b llama3.1-8b" ;;
  *) echo "нет группы $PILE" >&2; exit 1 ;;
esac
MODELS="${MODELS:-$DEFAULT}"
export HF_HUB_OFFLINE=1          # всё из кэша: прокси HF на сервере сбоит, у Llama-3.1 нет доступа к репо
STATUS=results/piles/$PILE.status
mkdir -p results/piles
echo "started $(date '+%F %T') GPU=$GPU models=[$MODELS]" >> "$STATUS"

for name in $MODELS; do
  cfg=configs/${name}_salad-o1o2o5o6.yaml
  echo "################ $name — $(date '+%F %T') — GPU $GPU ################"
  steps="06 09"
  [[ -f results/${name}_salad-o1o2o5o6/cross/cross.json ]] || steps="10 $steps"
  if CONFIG=$cfg STEPS="$steps" GPU=$GPU bash scripts/pipeline.sh; then
    echo "ok $name $(date '+%F %T')" >> "$STATUS"
  else
    echo "FAIL $name $(date '+%F %T')" >> "$STATUS"
  fi
done
echo "finished $(date '+%F %T')" >> "$STATUS"
echo "################ группа $PILE готова — $(date '+%F %T') ################"
