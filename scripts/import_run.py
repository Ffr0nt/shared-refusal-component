"""Импорт готового этапа 1 из чужой папки результатов того же формата (CPU, без модели).

Копирует сплиты, DIM (домены и union, только direction.pt и метаданные) и сиды RDO из конфига,
переносит в manifest секции splits и dim источника и помечает, откуда взято. Дальше — обычные
этапы: 01 --assemble-only, 02, 08. Так считаются модели, для которых DIM и RDO уже обучены
(прогоны boroday: BorodAILA/shared-refusal-component, тот же шаблон и refusal-токены, сплиты
побайтно совпадают).

    uv run python scripts/import_run.py --config configs/<имя>.yaml --src <папка результатов>

Уже существующую папку результатов не трогает.
"""
import argparse
import os
import shutil
import subprocess

from shared_refusal.artifacts import Run, read_json
from shared_refusal.config import load_config
from shared_refusal.directions.rdo import seed_path

SPLIT_FILES = [f"{h}_{s}.json" for h in ("harmful", "harmless") for s in ("train", "val", "test")]


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True, help="папка результатов источника (results/<модель>)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    src = os.path.abspath(args.src)
    assert not os.path.exists(run.root), f"{run.root} уже есть — импорт только в новую папку"
    sm = read_json(os.path.join(src, "manifest.json"))
    assert sm["model"] == cfg["model"], f"модель источника {sm['model']} ≠ {cfg['model']}"
    domains = list(cfg["domains"])

    for d in domains:
        for f in SPLIT_FILES:
            copy(os.path.join(src, "splits", d, f), os.path.join(run.splits(d), f))
    for d in domains + ["union"]:
        for f in ("direction.pt", "direction_metadata.json"):
            copy(os.path.join(src, "dim", d, f), os.path.join(run.dim(d), f))
    for d in domains:
        for s in cfg["rdo"]["seeds"][d]:
            copy(seed_path(os.path.join(src, "rdo", d), s), seed_path(run.rdo(d), s))

    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run.update_manifest("splits", sm["splits"])
    run.update_manifest("dim", sm["dim"])
    run.update_manifest("import", {"src": src, "src_commit": commit or None,
                                   "what": "splits, dim (direction.pt), rdo seeds"})
    print(f"импортировано из {src} -> {run.root}")


if __name__ == "__main__":
    main()
