#!/usr/bin/env bash
# Этап 12 (бенчмарки lm-eval при аблации) для прогонов boroday — одна карта, модели по очереди.
# Для каждой модели: импорт DIM и RDO из старой раскладки results/<модель>/ (scripts/import_run.py,
# если папки запуска ещё нет), сборка набора (01 --assemble-only) и компоненты (02), скачивание
# датасетов задач (12 --prefetch-only, с сетью) и сам этап 12 офлайн — модель и датасеты уже в кэше.
#
#   GPU=4 bash run/bench.sh                                          # 7 моделей, все задачи bench.tasks
#   GPU=4 TASKS=mmlu bash run/bench.sh                               # только MMLU
#   GPU=4 SMOKE=1 MODELS=llama3.2-1b TASKS=mmlu bash run/bench.sh    # limit 20 + тот же замер lm_eval CLI
#
# Env:
#   GPU     индекс карты, обязателен;
#   MODELS  имена конфигов без _salad-o1o2o5o6.yaml (по умолчанию все 7, от малых к большим);
#   TASKS   подмножество bench.tasks (по умолчанию все);
#   HF_MAX_MEMORY  потолок весов на видимых картах при делении модели, напр. "0:9GiB,1:22GiB"
#           (shared_refusal.models.max_memory); на числа не влияет;
#   SMOKE=1 копия конфига с bench.limit = 20 в отдельный запуск <имя>_smoke и для сверки «none» —
#           обычный lm_eval с теми же флагами в results/<имя>_smoke/bench_cli/.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс свободной карты>}"
MODELS="${MODELS:-llama3.2-1b qwen2.5-1.5b llama3.2-3b qwen2.5-7b llama3-8b llama3.1-8b qwen2.5-14b}"
TASKS="${TASKS:-}"
SMOKE="${SMOKE:-0}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/l.boroday/.hf_token}"   # свой; модели уже в HF-кэше
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cfg() { uv run python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print($2)" "$1"; }

for M in $MODELS; do
  CONFIG=configs/${M}_salad-o1o2o5o6.yaml
  [[ -f "$CONFIG" ]] || { echo "нет конфига $CONFIG" >&2; exit 1; }
  ALIAS=$(cfg "$CONFIG" "c['model'].split('/')[-1]")
  if [[ "$SMOKE" == 1 ]]; then
    mkdir -p results/_configs
    S=results/_configs/${M}_salad-o1o2o5o6_smoke.yaml
    uv run python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); c['bench']['limit'] = 20; \
yaml.safe_dump(c, open(sys.argv[2], 'w'), allow_unicode=True, sort_keys=False)" "$CONFIG" "$S"
    CONFIG=$S
  fi
  RUN=$(basename "$CONFIG" .yaml)
  echo "################ $M ($RUN) — $(date '+%F %T')"

  if [[ ! -d results/$RUN ]]; then
    uv run python scripts/import_run.py --config "$CONFIG" --src "results/$ALIAS"
  fi
  CONFIG=$CONFIG STEPS=01 bash scripts/pipeline.sh --assemble-only
  CONFIG=$CONFIG STEPS=02 bash scripts/pipeline.sh
  CONFIG=$CONFIG STEPS=12 bash scripts/pipeline.sh --prefetch-only ${TASKS:+--tasks $TASKS}
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CONFIG=$CONFIG STEPS=12 GPU=$GPU \
    bash scripts/pipeline.sh ${TASKS:+--tasks $TASKS}

  if [[ "$SMOKE" == 1 ]]; then
    MODEL=$(cfg "$CONFIG" "c['model']")
    REV=$(cfg "$CONFIG" "c['model_revision']")
    for T in ${TASKS:-$(cfg "$CONFIG" "' '.join(c['bench']['tasks'])")}; do
      NF=$(cfg "$CONFIG" "c['bench']['tasks']['$T']['num_fewshot']")
      echo "================ lm_eval CLI: $T — $(date '+%F %T')"
      HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU uv run lm-eval run --model hf \
        --model_args "pretrained=$MODEL,revision=$REV,dtype=bfloat16" --tasks "$T" --num_fewshot "$NF" \
        --limit 20 --apply_chat_template --fewshot_as_multiturn true --batch_size 16 --log_samples \
        --seed 0,0,0,0 --output_path "results/$RUN/bench_cli/$T"
    done
  fi
done
echo "################ готово — $(date '+%F %T')"
