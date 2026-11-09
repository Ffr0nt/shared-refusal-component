#!/usr/bin/env bash
# Gemma-2-9B-it × SALAD-Bench (O1, O2, O5, O6): 4 DIM + 20 одномерных RDO (сиды 1–20) — третье
# семейство после Llama и Qwen. Помещается на одну H100.
#
#   bash scripts/fetch_model.sh configs/gemma2-9b_salad-o1o2o5o6.yaml        # один раз: веса (gated, нужен HF-токен)
#   GPU=0 STEPS="00 01" bash scripts/tmux.sh launch/gemma2-9b_salad-o1o2o5o6.sh   # сплиты, DIM, RDO
#   STEPS="02 04 05" bash launch/gemma2-9b_salad-o1o2o5o6.sh                  # CPU-этапы
#   GPU=0 STEPS="03 06" bash launch/gemma2-9b_salad-o1o2o5o6.sh               # аблация и StrongREJECT
#
# Конфиг зафиксирован хэшем: правка конфига без обновления CONFIG_SHA256 не запустится.
export CONFIG=configs/gemma2-9b_salad-o1o2o5o6.yaml
export CONFIG_SHA256=19a7e3512e9f03b83c7b2500ef632da3b94ba84d7e92777737f196073a7feb40
exec bash "$(dirname "$0")/../scripts/pipeline.sh" "$@"
