"""Происхождение результатов: из какого конфига, на каких коммитах и ревизиях посчитан этап.

Каждый этап зависит от своих секций конфига и от секций этапов выше по течению. Хэш этих
секций пишется в manifest при завершении этапа; перед стартом этап сверяет хэши уже готовых
этапов выше по течению с текущим конфигом. Поэтому в одной папке результатов не смешиваются
числа от разных конфигов: поменял rdo.hparams — этапы 2–4 откажутся работать поверх старых
направлений, пока не пересчитан этап 1. А добавить вариант компоненты можно поверх готового
этапа 1: секция component на него не влияет.
"""
import hashlib
import json
import os
import subprocess

from .artifacts import FORK, REPO

# Секции конфига, от которых зависит этап сам по себе, и этапы выше по течению.
OWN = {"data": ("dataset", "domains", "splits"),
       "directions": ("model", "model_revision", "rdo"),
       "component": ("component",),
       "ablation": ("ablation",),
       "analysis": (),
       "convergence": ("convergence",)}
UPSTREAM = {"data": (),
            "directions": ("data",),
            "component": ("directions",),
            "ablation": ("data", "directions", "component"),
            "analysis": ("directions", "component"),
            "convergence": ("directions",)}


def sections(stage):
    """Все секции, от которых зависит этап (с учётом этапов выше по течению)."""
    out = set(OWN[stage])
    for up in UPSTREAM[stage]:
        out |= sections(up)
    return out


def stage_hash(cfg, stage):
    picked = {k: cfg.get(k) for k in sorted(sections(stage))}
    return hashlib.sha256(json.dumps(picked, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def git_state(path):
    def git(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return {"commit": commit or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def check_upstream(run, cfg, stage):
    """Этапы выше по течению, уже записанные в manifest, посчитаны с тем же конфигом."""
    done = run.manifest().get("provenance", {})
    for up in UPSTREAM[stage]:
        if up in done and done[up]["config_hash"] != stage_hash(cfg, up):
            raise RuntimeError(
                f"этап {up} в {run.root} посчитан с другим конфигом (секции {sorted(sections(up))}); "
                f"пересчитай {up} или заведи новый запуск")


def record(run, cfg, stage):
    prov = run.manifest().get("provenance", {})
    prov[stage] = {"config_hash": stage_hash(cfg, stage), "config": cfg.get("config_path"),
                   "repo": git_state(REPO), "fork": git_state(FORK)}
    for k in ("repo", "fork"):
        if prov[stage][k]["dirty"]:
            print(f"[предупреждение: рабочее дерево {k} не закоммичено — результат не воспроизводится по коммиту]")
    run.update_manifest("provenance", prov)


def hub_ref(repo_id, repo_type="model"):
    """Коммит, на который в локальном HF-кэше указывает refs/main (None, если репозитория нет)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    folder = f"{repo_type}s--{repo_id.replace('/', '--')}"
    p = os.path.join(os.getenv("HUGGINGFACE_CACHE_DIR") or HF_HUB_CACHE, folder, "refs", "main")
    return open(p).read().strip() if os.path.exists(p) else None


def check_revision(repo_id, revision, repo_type="model"):
    """Форк грузит модель по имени (ветка main), без ревизии. Поэтому перед этапом проверяем,
    что main в кэше указывает ровно на зафиксированную в конфиге ревизию."""
    ref = hub_ref(repo_id, repo_type)
    if ref != revision:
        raise RuntimeError(f"{repo_type} {repo_id}: в кэше main = {ref}, в конфиге {revision}")
