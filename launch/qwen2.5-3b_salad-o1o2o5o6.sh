#!/usr/bin/env bash
# Qwen2.5-3B-Instruct × SALAD-Bench (O1, O2, O5, O6): 4 DIM + 20 одномерных RDO (сиды 1–20),
# общая компонента, аблация, анализ, сходимость по числу векторов, проверка на StrongREJECT.
# Этапы 0–1 и 3 посчитаны 12.10.2026 на GPU 1.
#
#   GPU=0 bash launch/qwen2.5-3b_salad-o1o2o5o6.sh                    # этапы 00–06
#   STEPS=07 bash launch/qwen2.5-3b_salad-o1o2o5o6.sh                 # судья, где есть OpenRouter
#   STEPS="02 04 05" bash launch/qwen2.5-3b_salad-o1o2o5o6.sh         # только CPU-этапы
#   GPU=0 bash scripts/tmux.sh launch/qwen2.5-3b_salad-o1o2o5o6.sh    # то же в tmux с логом
#
# Конфиг зафиксирован хэшем: правка конфига без обновления CONFIG_SHA256 не запустится.
export CONFIG=configs/qwen2.5-3b_salad-o1o2o5o6.yaml
export CONFIG_SHA256=8daaf4cd0833e1cc15620b8e13e9bc50dc413ef250606839ececdc829c9c9b09
exec bash "$(dirname "$0")/../scripts/pipeline.sh" "$@"
