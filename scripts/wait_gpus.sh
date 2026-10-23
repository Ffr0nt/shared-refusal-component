#!/usr/bin/env bash
# Ждёт N свободных карт и печатает их индексы через запятую (для GPU=...). Карта свободна, если
# на ней занято меньше MAX_USED_MIB CONFIRM проверок подряд. Лог ожидания — в stderr.
#
#   GPU=$(bash scripts/wait_gpus.sh 2) && GPU=$GPU STEPS="03 06" bash launch/<имя>.sh
#
# Env: GPUS (из каких карт выбирать; по умолчанию все), INTERVAL (60), MAX_USED_MIB (1000), CONFIRM (2).
set -uo pipefail
N="${1:?укажи число карт}"
INTERVAL="${INTERVAL:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
CONFIRM="${CONFIRM:-2}"
declare -A FREE

while true; do
  ready=()
  state=""
  while IFS=', ' read -r idx used; do
    [[ -z "$idx" ]] && continue
    if [[ -n "${GPUS:-}" && " $GPUS " != *" $idx "* ]]; then continue; fi
    if (( used < MAX_USED_MIB )); then FREE[$idx]=$(( ${FREE[$idx]:-0} + 1 )); else FREE[$idx]=0; fi
    (( ${FREE[$idx]} >= CONFIRM )) && ready+=("$idx")
    state+=" $idx:${used}MiB"
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if (( ${#ready[@]} >= N )); then
    sel=$(IFS=,; echo "${ready[*]:0:N}")
    echo "[$(date '+%F %T')] свободны: $sel" >&2
    echo "$sel"
    exit 0
  fi
  echo "[$(date '+%F %T')] карты:$state; свободных ${#ready[@]} из нужных $N" >&2
  sleep "$INTERVAL"
done
