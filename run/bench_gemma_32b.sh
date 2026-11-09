#!/usr/bin/env bash
# Этап 12 (GSM8K и ARC-Challenge при аблации) для двух моделей, которых нет в сводке бенчмарков:
# gemma-2-9b (третье семейство, одна карта) и Qwen2.5-32B (две карты, device_map="auto").
# Вотчер: ждёт свободные карты (scripts/wait_gpus.sh), запускает модель в своей tmux-сессии
# fz-bench-<модель> и ждёт её конца — модели по очереди, поэтому наших карт не больше двух. Готовая
# задача пропускается (bench/<задача>/summary.json). В конце — сводка и рисунок (summary_bench, plot_bench).
# Ход — в results/piles/bench-gemma-32b.status. Сессии — на своём сокете tmux (-L fz): общий сокет
# пользователя jovyan делят все, и 05.11 чей-то kill-server снял очередь посреди ARC.
#
#   tmux -L fz new-session -d -s fz-bench-watch "bash run/bench_gemma_32b.sh 2>&1 | tee -a results/piles/bench-gemma-32b.log"
#   tail -f results/piles/bench-gemma-32b.log         # что видит вотчер
#   tmux -L fz attach -t fz-bench-gemma2-9b                 # прогон модели (выйти: Ctrl-b d)
#
# Env: MODELS ("gemma2-9b qwen2.5-32b"), TASKS ("arc_challenge gsm8k"), INTERVAL (10 — освободившуюся
# карту соседи забирают за полминуты), MAX_USED_MIB, CONFIRM — как в scripts/wait_gpus.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
STATUS=results/piles/bench-gemma-32b.status

if [[ "${1:-}" == --job ]]; then        # внутри tmux-сессии: одна модель, задачи по одной
  M=$2 GPU=$3; shift 3
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  rc=0
  for T in "$@"; do
    GPU=$GPU STEPS=12 bash "launch/${M}_salad-o1o2o5o6.sh" --tasks "$T" || rc=1   # сбой задачи не отменяет следующую
  done
  echo "$rc" > "results/piles/.fz-bench-${M//./_}.rc"
  exit "$rc"
fi

MODELS="${MODELS:-gemma2-9b qwen2.5-32b}"
TASKS="${TASKS:-arc_challenge gsm8k}"
export INTERVAL="${INTERVAL:-10}"
mkdir -p results/piles
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
ncards() { case "$1" in qwen2.5-32b) echo 2 ;; *) echo 1 ;; esac; }

log "started models=[$MODELS] tasks=[$TASKS]"
for M in $MODELS; do
  todo=()
  for T in $TASKS; do [[ -f results/${M}_salad-o1o2o5o6/bench/$T/summary.json ]] || todo+=("$T"); done
  if (( ${#todo[@]} == 0 )); then log "skip $M — все задачи готовы"; continue; fi

  N=$(ncards "$M")
  log "жду $N карт(ы) для $M (задачи: ${todo[*]})"
  GPU=$(bash scripts/wait_gpus.sh "$N") || { log "FAIL $M — wait_gpus"; continue; }
  SESSION=fz-bench-${M//./_}          # tmux меняет точку в имени сессии на _
  LOGF=results/piles/bench-$M-$(date +%Y%m%d-%H%M%S).log
  rm -f "results/piles/.$SESSION.rc"
  tmux -L fz new-session -d -s "$SESSION" -c "$PWD" \
    "bash run/bench_gemma_32b.sh --job $M $GPU ${todo[*]} 2>&1 | tee $LOGF"
  log "$M на GPU $GPU — сессия $SESSION, лог $LOGF"
  while tmux -L fz has-session -t "$SESSION" 2>/dev/null; do sleep 60; done
  rc=$(cat "results/piles/.$SESSION.rc" 2>/dev/null || echo "?"); rm -f "results/piles/.$SESSION.rc"
  if [[ "$rc" == 0 ]]; then log "ok $M"; else log "FAIL $M (rc=$rc) — см. $LOGF"; fi
done
uv run python scripts/summary_bench.py && uv run python scripts/plot_bench.py
log "finished"
