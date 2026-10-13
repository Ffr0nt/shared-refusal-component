"""Общие пути, конфиг и manifest для шагов 00–05."""
import json
import os

import yaml

from .geometry import unit

# src/shared_refusal/common.py -> корень репозитория (пакет ставится editable, файл остаётся на месте)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK = os.path.abspath(os.getenv("FORK", os.path.join(REPO, "..", "geometry-of-refusal")))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["model_alias"] = cfg["model"].split("/")[-1]
    return cfg


def results_dir(cfg, *parts):
    return os.path.join(REPO, "results", cfg["model_alias"], *parts)


def splits_dir(cfg, domain):
    return results_dir(cfg, "splits", domain)


def dim_dir(cfg, domain):
    return results_dir(cfg, "dim", domain)


def rdo_dir(cfg, domain):
    return results_dir(cfg, "rdo", domain)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def update_manifest(cfg, key, value):
    """manifest.json — что получено каждым шагом (точка извлечения, сиды, хэши сплитов)."""
    path = results_dir(cfg, "manifest.json")
    m = read_json(path) if os.path.exists(path) else {"model": cfg["model"]}
    m[key] = value
    write_json(m, path)


def read_manifest(cfg):
    return read_json(results_dir(cfg, "manifest.json"))


def write_report(cfg, name, lines):
    path = results_dir(cfg, "reports", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[отчёт: {path}]")


def load_vectors(cfg):
    """Все 24 вектора в фиксированном порядке: по доменам — DIM, затем RDO по сидам.
    Возвращает (names, V [N, hidden] float64, единичные строки)."""
    import torch

    names, vecs = [], []
    for d in cfg["domains"]:
        vecs.append(torch.load(os.path.join(dim_dir(cfg, d), "direction.pt"), map_location="cpu", weights_only=True))
        names.append(f"dim_{d}")
        for s in cfg["rdo"]["seeds"][d]:
            p = os.path.join(rdo_dir(cfg, d), f"seed_{s:02d}.pt")
            vecs.append(torch.load(p, map_location="cpu", weights_only=True)["vector"])
            names.append(f"rdo_{d}_s{s:02d}")
    V = torch.stack([unit(v.double()) for v in vecs])
    return names, V
