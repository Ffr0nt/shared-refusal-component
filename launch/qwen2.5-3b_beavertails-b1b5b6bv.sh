#!/usr/bin/env bash
# Qwen2.5-3B-Instruct × BeaverTails (B1, B5, B6, BV): контроль целей RDO.
# Цели обучения RDO берутся готовыми из датасета, DIM в них не участвует (лосс L_abl + L_ret);
# всё остальное — обычный протокол: 4 DIM + 20 одномерных RDO, общая компонента по 24 векторам.
# Постановка и разбор — concept_dim/experiments/natural_targets/.
#
#   GPU=0 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh                 # этапы 00–06
#   STEPS=00 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh              # только сплиты (CPU)
#   STEPS=01 GPU=0 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh        # цели, DIM и 20 RDO
#   STEPS=07 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh              # судья, где есть OpenRouter
#   STEPS="02 04 05" bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh      # только CPU-этапы
#   GPU=0 bash scripts/tmux.sh launch/qwen2.5-3b_beavertails-b1b5b6bv.sh # то же в tmux с логом
#
# Конфиг зафиксирован хэшем: правка конфига без обновления CONFIG_SHA256 не запустится.
export CONFIG=configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml
export CONFIG_SHA256=9e4d6562abbbcf0acb33828037c6551c202dea5ea44346553bfd549f494fafc0
exec bash "$(dirname "$0")/../scripts/pipeline.sh" "$@"
