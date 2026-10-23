#!/usr/bin/env bash
# Ответы на StrongREJECT для 20 RDO на 7 моделях (на Qwen2.5-3B они уже есть) — чтобы судья
# сравнил RDO с DIM и w на всех 8 моделях (23.10.2026). В конфигах generate.select: {} — этап 06
# добирает только направления без готовых ответов, остальное пропускает.
#
#   GPU_QWEN=0 GPU_LLAMA=1 bash launch/rdo_strongreject.sh
#   tmux attach -t fz-rdo-qwen                      # посмотреть (выйти: Ctrl-b d)
#
# Две очереди scripts/gpu_pile.sh в tmux, по карте на семейство (не больше двух карт). Сессии
# закрываются сами, лог — results/piles/rdo-<группа>.log; конец очереди — строка «группа … готова».
# Судья потом на Mac, когда обе очереди закончатся:
#   MODELS="qwen2.5-1.5b qwen2.5-7b qwen2.5-14b llama3.2-1b llama3.2-3b llama3-8b llama3.1-8b" bash scripts/judge_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU_QWEN:?укажи GPU_QWEN=<индекс>}"
: "${GPU_LLAMA:?укажи GPU_LLAMA=<индекс>}"
mkdir -p results/piles

start() {  # группа, карта, модели
  local session=fz-rdo-$1
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "сессия $session уже есть — tmux attach -t $session" >&2; return 1
  fi
  tmux new-session -d -s "$session" -c "$PWD" \
    "GPU=$2 MODELS='$3' bash scripts/gpu_pile.sh $1 2>&1 | tee -a results/piles/rdo-$1.log"
  echo "сессия: $session  GPU $2  модели: $3  лог: results/piles/rdo-$1.log"
}
start qwen  "$GPU_QWEN"  "qwen2.5-1.5b qwen2.5-7b qwen2.5-14b"
start llama "$GPU_LLAMA" "llama3.2-1b llama3.2-3b llama3-8b llama3.1-8b"
