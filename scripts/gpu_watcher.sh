#!/usr/bin/env bash
# Ждёт свободные карты и запускает на них очереди scripts/gpu_pile.sh — каждую в своей tmux-сессии.
# Раз в INTERVAL секунд смотрит nvidia-smi; карта свободна, если на ней занято меньше MAX_USED_MIB
# CONFIRM проверок подряд (не хватаем карту, которую сосед освободил на минуту). Свободная карта
# получает первую ещё не запущенную группу из PILES. Группа считается запущенной, если есть её
# сессия fz-pile-<группа> или её статус results/piles/<группа>.status. Когда запущены все — выходит.
#
#   tmux new-session -d -s fz-gpu-watcher "bash scripts/gpu_watcher.sh 2>&1 | tee -a results/piles/watcher.log"
#   tail -f results/piles/watcher.log            # что видит вотчер
#   tmux attach -t fz-pile-qwen                  # очередь (выйти: Ctrl-b d)
#
# Env: PILES ("qwen llama"), GPUS (какие карты можно брать; по умолчанию все), INTERVAL (60),
# MAX_USED_MIB (1000), CONFIRM (2).
set -uo pipefail
cd "$(dirname "$0")/.."
PILES="${PILES:-qwen llama}"
INTERVAL="${INTERVAL:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
CONFIRM="${CONFIRM:-2}"
mkdir -p results/piles
declare -A FREE CLAIMED   # карта -> проверок подряд свободна; карта -> группа, запущенная на ней

log() { echo "[$(date '+%F %T')] $*"; }
launched() { tmux has-session -t "fz-pile-$1" 2>/dev/null || [[ -f "results/piles/$1.status" ]]; }

log "вотчер: группы [$PILES], карты [${GPUS:-все}], раз в ${INTERVAL} с, свободна при < ${MAX_USED_MIB} MiB ${CONFIRM} раза подряд"
while true; do
  pending=()
  for p in $PILES; do launched "$p" || pending+=("$p"); done
  if (( ${#pending[@]} == 0 )); then log "все группы запущены — выхожу"; exit 0; fi

  # карта, на которой мы уже запустили группу, занята, пока жива её сессия
  for g in "${!CLAIMED[@]}"; do
    tmux has-session -t "fz-pile-${CLAIMED[$g]}" 2>/dev/null || unset "CLAIMED[$g]"
  done

  state=""
  while IFS=', ' read -r idx used; do
    [[ -z "$idx" ]] && continue
    if [[ -n "${GPUS:-}" && " $GPUS " != *" $idx "* ]]; then continue; fi
    if [[ -z "${CLAIMED[$idx]:-}" ]] && (( used < MAX_USED_MIB )); then
      FREE[$idx]=$(( ${FREE[$idx]:-0} + 1 ))
    else
      FREE[$idx]=0
    fi
    state+=" $idx:${used}MiB"
    if (( ${FREE[$idx]} >= CONFIRM && ${#pending[@]} > 0 )); then
      p="${pending[0]}"; pending=("${pending[@]:1}")
      logf="results/piles/$p-$(date +%Y%m%d-%H%M%S).log"
      tmux new-session -d -s "fz-pile-$p" -c "$PWD" "GPU=$idx bash scripts/gpu_pile.sh $p 2>&1 | tee $logf"
      CLAIMED[$idx]=$p; FREE[$idx]=0
      log "карта $idx свободна — запустил группу $p (сессия fz-pile-$p, лог $logf)"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  log "карты:$state; ждут: [${pending[*]:-}]"
  sleep "$INTERVAL"
done
