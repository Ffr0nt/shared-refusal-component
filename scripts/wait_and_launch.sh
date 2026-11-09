#!/usr/bin/env bash
# Ждёт N свободных карт и сразу запускает на них launch-файл. Нужен, когда карты заняты
# соседями: ставишь в tmux и уходишь, прогон начнётся сам.
#
#   tmux new-session -d -s fz-wait-<имя> "STEPS='01 02 03 04 05 06' bash scripts/wait_and_launch.sh launch/<имя>.sh 1"
#   tail -f results/<имя>/logs/<последний>.log      # ожидание и прогон в одном логе
#
# Карта считается свободной по правилам scripts/wait_gpus.sh (по умолчанию занято меньше
# 1000 MiB две проверки подряд) — чужую карту, освободившуюся на минуту, не хватаем.
# Сессия закрывается сама, когда прогон закончится.
#
# Env: STEPS (этапы, как в pipeline.sh), N из второго аргумента (по умолчанию 1);
#      GPUS, INTERVAL, MAX_USED_MIB, CONFIRM — для wait_gpus.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

LAUNCH="${1:?укажи launch/<имя>.sh}"
N="${2:-1}"
[[ -f "$LAUNCH" ]] || { echo "нет launch-файла $LAUNCH" >&2; exit 1; }
NAME=$(basename "$LAUNCH" .sh)
LOG="results/$NAME/logs/$(date +%Y%m%d-%H%M%S).log"
mkdir -p "results/$NAME/logs"
echo "лог: $LOG"

{
  echo "[$(date '+%F %T')] ждём $N свободных карт для $LAUNCH; STEPS=${STEPS:-по умолчанию}"
  GPU=$(bash scripts/wait_gpus.sh "$N") || exit 1
  echo "[$(date '+%F %T')] карты $GPU — старт"
  GPU="$GPU" bash "$LAUNCH"
  echo "[$(date '+%F %T')] прогон завершён, код $?"
} 2>&1 | tee -a "$LOG"
