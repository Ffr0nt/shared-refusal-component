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
export CONFIG_SHA256=4ba4fb788804bc6a345cee9b6c0d5add643f708d6153d52df59327dc6ee0dbc4
exec bash "$(dirname "$0")/../scripts/pipeline.sh" "$@"
