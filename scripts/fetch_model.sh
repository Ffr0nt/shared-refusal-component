#!/usr/bin/env bash
# Скачивает модель конфига в HF-кэш (ветку main — так её грузит форк) и проверяет, что main
# указывает ровно на model_revision из конфига. Битые ссылки в snapshots/ (кэш без blobs/)
# перекачиваются. GPU не нужен.
#
#   bash scripts/fetch_model.sh configs/<имя>.yaml
#
# Env: HF_HOME (по умолчанию ~/.cache/huggingface).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

CONFIG="${1:?укажи configs/<имя>.yaml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$HF_HOME/hub}"
df -h "$HUGGINGFACE_CACHE_DIR" | tail -1

uv run python - "$CONFIG" <<'EOF'
import sys

from huggingface_hub import snapshot_download

from shared_refusal.config import load_config
from shared_refusal.provenance import check_revision

cfg = load_config(sys.argv[1])
path = snapshot_download(cfg["model"], revision="main")
print(f"{cfg['model']} -> {path}")
check_revision(cfg["model"], cfg["model_revision"])
print(f"main = {cfg['model_revision']}, как в конфиге")
EOF
