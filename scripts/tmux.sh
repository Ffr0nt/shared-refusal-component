#!/usr/bin/env bash
# Запускает launch-файл в отдельной tmux-сессии; лог — в results/<имя>/logs/.
#
#   GPU=0 bash scripts/tmux.sh launch/<имя>.sh            # старт; env передаётся внутрь
#   tmux attach -t fz-<имя>                               # посмотреть (выйти: Ctrl-b d)
#
# Сессия закрывается сама по завершении (лог остаётся). Env: SESSION (fz-<имя>).
set -euo pipefail
cd "$(dirname "$0")/.."

LAUNCH="${1:?укажи launch/<имя>.sh}"; shift
NAME=$(basename "$LAUNCH" .sh)
SESSION="${SESSION:-fz-$NAME}"
LOG="results/$NAME/logs/$(date +%Y%m%d-%H%M%S).log"
mkdir -p "results/$NAME/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
ENV=$(printf '%q ' "GPU=${GPU:-}" "STEPS=${STEPS:-00 01 02 03 04 05}")
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "env $ENV bash $(printf '%q ' "$LAUNCH" "$@") 2>&1 | tee $LOG"
echo "сессия: $SESSION  лог: $LOG"
