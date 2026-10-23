#!/usr/bin/env bash
# Судьи на Mac для моделей, у которых очередь на сервере закончилась (строка «ok <модель>» в
# results/piles/<группа>.status): копирует с сервера их результаты, гоняет 07 (StrongREJECT) и 11
# (XSTest для вызова отказа), собирает results/cross-model-judge.md и отправляет оценки обратно.
# С сервера OpenRouter недоступен (403), поэтому судьи — здесь. Готовые оценки повторно не
# отправляются, так что скрипт можно запускать сколько угодно раз по мере готовности моделей.
#
#   bash scripts/judge_mac.sh                     # все готовые модели
#   MODELS="qwen2.5-3b" bash scripts/judge_mac.sh
#
# Обратно копируются только оценки и отчёты, без manifest.json: на сервере в это время очередь
# может писать manifest той же модели. Env: HOST (peaceful-matsumoto), REMOTE (путь к репо там).
set -euo pipefail
export COPYFILE_DISABLE=1   # tar на macOS иначе кладёт в архив ._* (AppleDouble)
cd "$(dirname "$0")/.."
HOST="${HOST:-peaceful-matsumoto}"
REMOTE="${REMOTE:-~/f.zakharov/shared-refusal-component}"
mkdir -p results

if [[ -z "${MODELS:-}" ]]; then
  MODELS=$(ssh "$HOST" "cat $REMOTE/results/piles/*.status 2>/dev/null" | awk '$1 == "ok" {print $2}' | sort -u | tr '\n' ' ')
fi
[[ -n "${MODELS// }" ]] || { echo "готовых моделей пока нет"; exit 0; }
echo "модели: $MODELS"

for name in $MODELS; do
  run=${name}_salad-o1o2o5o6
  echo "################ $run — $(date '+%F %T') ################"
  ssh "$HOST" "cd $REMOTE/results && tar czf - --ignore-failed-read $run/manifest.json $run/strongreject $run/induce \
    $run/cross $run/decompose/decompose.json $run/ablation/ablation.json $run/splits 2>/dev/null" | tar xzf - -C results/
  CONFIG=configs/$run.yaml STEPS="07 11" bash scripts/pipeline.sh
  tar czf - -C results "$run/strongreject/judge" "$run/strongreject/strongreject.json" "$run/induce/judge" \
    "$run/induce/induce.json" "$run/reports" | ssh "$HOST" "tar xzf - -C $REMOTE/results"
done
uv run python scripts/summary_judge.py
tar czf - -C results cross-model-judge.md | ssh "$HOST" "tar xzf - -C $REMOTE/results"
