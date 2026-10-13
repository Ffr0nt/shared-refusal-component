"""Конфиг запуска: чтение и хэш. Значений по умолчанию здесь нет — всё, что влияет на
числа, должно быть записано в yaml."""
import hashlib

import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["model_alias"] = cfg["model"].split("/")[-1]
    return cfg


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
