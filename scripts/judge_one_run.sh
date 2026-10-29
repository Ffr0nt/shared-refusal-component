#!/usr/bin/env bash
# Судья StrongREJECT (этап 07) для одного запуска, с Mac: забирает ответы с сервера, оценивает
# и отправляет оценки обратно. С сервера OpenRouter недоступен (403), поэтому судья здесь.
# Оценки кэшируются построчно, так что скрипт можно запускать повторно по мере готовности.
#
#   bash scripts/judge_one_run.sh configs/<имя>.yaml
#   STEPS="07 11" bash scripts/judge_one_run.sh configs/<имя>.yaml   # ещё и XSTest-судья
#
# Обратно копируются только оценки и отчёты, без manifest.json: на сервере в это время может
# идти другой этап того же запуска. Env: HOST, REMOTE, STEPS, OPENROUTER_API_KEY (или .env).
set -euo pipefail
export COPYFILE_DISABLE=1   # tar на macOS иначе кладёт в архив ._* (AppleDouble)
cd "$(dirname "$0")/.."
CONFIG="${1:?укажи configs/<имя>.yaml}"
[[ -f "$CONFIG" ]] || { echo "нет конфига $CONFIG" >&2; exit 1; }
RUN=$(basename "$CONFIG" .yaml)
HOST="${HOST:-peaceful-matsumoto}"
REMOTE="${REMOTE:-~/f.zakharov/shared-refusal-component}"
STEPS="${STEPS:-07}"
mkdir -p results

echo "################ $RUN — забираю ответы с $HOST — $(date '+%F %T') ################"
ssh "$HOST" "cd $REMOTE/results && tar czf - --ignore-failed-read \
  $RUN/manifest.json $RUN/strongreject $RUN/splits $RUN/ablation/ablation.json \
  $RUN/component 2>/dev/null" | tar xzf - -C results/

echo "################ судья ($STEPS) — $(date '+%F %T') ################"
CONFIG="$CONFIG" STEPS="$STEPS" bash scripts/pipeline.sh

echo "################ отправляю оценки обратно — $(date '+%F %T') ################"
# tar на macOS не знает --ignore-failed-read, поэтому отсутствующие пути отбираются заранее:
# strongreject.json и часть отчётов появляются не на каждом наборе STEPS.
SEND=()
for p in "$RUN/strongreject/judge" "$RUN/strongreject/strongreject.json" "$RUN/reports" \
         "$RUN/xstest/judge" "$RUN/xstest/xstest.json"; do
  [[ -e "results/$p" ]] && SEND+=("$p")
done
[[ ${#SEND[@]} -gt 0 ]] || { echo "нечего отправлять: судья не оставил файлов" >&2; exit 1; }
printf '  %s\n' "${SEND[@]}"
tar czf - -C results "${SEND[@]}" | ssh "$HOST" "tar xzf - -C $REMOTE/results"
echo "готово — $(date '+%F %T')"
