#!/usr/bin/env bash
# Этапы 08–10 (decompose, induce, cross) для Phi-4-mini и Phi-4 — то, чего им не хватает до набора
# остальных моделей. Этапы 00–06 посчитал Л. Бородай этим же кодом (feat/phi4-mini, 58d0e3d); его
# результаты скопированы в results/ без bench. Судья induce (этап 11) — потом с Mac.
# Вотчер: ждёт свободную карту (scripts/wait_gpus.sh), запускает модель в своей tmux-сессии
# fz-phi4-<модель> и ждёт её конца — модели по очереди, наша карта одна. Ход — в
# results/piles/phi4-stages.status. Сессии — на своём сокете tmux (-L fz).
#
#   tmux -L fz new-session -d -s fz-phi4-watch "bash run/phi4_stages.sh 2>&1 | tee -a results/piles/phi4-stages.log"
#   tmux -L fz attach -t fz-phi4-phi4-mini                  # прогон модели (выйти: Ctrl-b d)
#
# Env: MODELS ("phi4-mini phi4"), STEPS ("08 09 10"), INTERVAL (10), MAX_USED_MIB, CONFIRM — как в
# scripts/wait_gpus.sh.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
STATUS=results/piles/phi4-stages.status

if [[ "${1:-}" == --job ]]; then        # внутри tmux-сессии: одна модель, этапы подряд
  M=$2 GPU=$3; shift 3
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CONFIG=configs/${M}_salad-o1o2o5o6.yaml GPU=$GPU STEPS="$*" bash scripts/pipeline.sh
  rc=$?
  echo "$rc" > "results/piles/.fz-phi4-${M//./_}.rc"
  exit "$rc"
fi

MODELS="${MODELS:-phi4-mini phi4}"
STEPS="${STEPS:-08 09 10}"
export INTERVAL="${INTERVAL:-10}"
mkdir -p results/piles
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "started models=[$MODELS] steps=[$STEPS]"
for M in $MODELS; do
  log "жду карту для $M"
  GPU=$(bash scripts/wait_gpus.sh 1) || { log "FAIL $M — wait_gpus"; continue; }
  SESSION=fz-phi4-${M//./_}             # tmux меняет точку в имени сессии на _
  LOGF=results/piles/phi4-stages-$M-$(date +%Y%m%d-%H%M%S).log
  rm -f "results/piles/.$SESSION.rc"
  tmux -L fz new-session -d -s "$SESSION" -c "$PWD" \
    "bash run/phi4_stages.sh --job $M $GPU $STEPS 2>&1 | tee $LOGF"
  log "$M на GPU $GPU — сессия $SESSION, лог $LOGF"
  while tmux -L fz has-session -t "$SESSION" 2>/dev/null; do sleep 60; done
  rc=$(cat "results/piles/.$SESSION.rc" 2>/dev/null || echo "?"); rm -f "results/piles/.$SESSION.rc"
  if [[ "$rc" == 0 ]]; then log "ok $M"; else log "FAIL $M (rc=$rc) — см. $LOGF"; fi
done
log "finished"
