"""Конфиг запуска: чтение и хэш. Значений по умолчанию здесь нет — всё, что влияет на
числа, должно быть записано в yaml.

Имя запуска — имя файла конфига без расширения: configs/<имя>.yaml, launch/<имя>.sh и
results/<имя>/ называются одинаково.
"""
import hashlib
import os

import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["model_alias"] = cfg["model"].split("/")[-1]
    cfg["run_name"] = os.path.splitext(os.path.basename(path))[0]
    cfg["config_path"] = os.path.relpath(os.path.abspath(path), _repo())
    return cfg


def _repo():
    from .artifacts import REPO
    return REPO


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
