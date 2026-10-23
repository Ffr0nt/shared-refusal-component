#!/usr/bin/env bash
# Qwen2.5-32B-Instruct × SALAD-Bench (O1, O2, O5, O6): 4 DIM + 20 одномерных RDO (сиды 1–20).
# Модель (~65 ГБ в bf16) не помещается на одну H100 вместе с обучением RDO — GPU всегда две
# карты через запятую, device_map="auto" делит слои между ними.
#
#   bash scripts/fetch_model.sh configs/qwen2.5-32b_salad-o1o2o5o6.yaml   # один раз: веса в HF-кэш
#   GPU=0,1 STEPS="00 01" bash scripts/tmux.sh launch/qwen2.5-32b_salad-o1o2o5o6.sh   # сплиты, DIM, RDO
#   STEPS="02 04 05" bash launch/qwen2.5-32b_salad-o1o2o5o6.sh         # CPU-этапы по готовым направлениям
#   GPU=0,1 STEPS="03 06" bash launch/qwen2.5-32b_salad-o1o2o5o6.sh    # аблация и StrongREJECT
#
# Конфиг зафиксирован хэшем: правка конфига без обновления CONFIG_SHA256 не запустится.
export CONFIG=configs/qwen2.5-32b_salad-o1o2o5o6.yaml
export CONFIG_SHA256=a62a0283882ec4aa9e0c98fae82566c8f6efef3a50ed3f2513160fe2f2b11a55
exec bash "$(dirname "$0")/../scripts/pipeline.sh" "$@"
