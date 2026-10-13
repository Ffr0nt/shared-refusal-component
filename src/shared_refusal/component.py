"""Этап 2 — общая компонента набора направлений (CPU, без модели).

Вариант = правило отбора из DirectionSet (методы, домены, исключения по именам). Популяция
варианта взвешена поровну: P = (1/N) sum v v^T, w — первый собственный вектор, lam_1 = mean cos^2.
Знак w — по среднему DIM варианта (добавление w наводит отказ); если DIM в варианте нет —
по среднему всех направлений.
"""
import os

import torch

from .artifacts import read_json, write_json
from .geometry import null_lam1, shared_component

SELECTION_KEYS = ("methods", "domains", "exclude")


def select(ds, spec):
    unknown = set(spec) - set(SELECTION_KEYS)
    assert not unknown, f"неизвестные ключи отбора {sorted(unknown)}; допустимы {SELECTION_KEYS}"
    sub = ds.select(methods=spec.get("methods"), domains=spec.get("domains"), exclude=spec.get("exclude", ()))
    assert len(sub) >= 2, f"в варианте {spec} меньше двух направлений"
    return sub


def compute(ds, n_null, null_seed):
    dims = ds.where("dim")
    ref = ds.V[dims].mean(0) if dims else ds.V.mean(0)
    w, lam, cos = shared_component(ds.V, ref)
    null = null_lam1(len(ds), ds.hidden, n_null, null_seed)
    return {"w": w, "lam": lam, "cos": cos, "names": list(ds.names),
            "n_by_method": {m: len(ds.where(m)) for m in dict.fromkeys(x.get("method") for x in ds.meta)},
            "null": {"mean": float(null.mean()), "p95": float(null.quantile(0.95)),
                     "max": float(null.max()), "n": n_null}}


def save(res, out_dir, spec):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"w": res["w"].float(), "lam": res["lam"][:8].float(), "names": res["names"],
                "cos": res["cos"].float()}, os.path.join(out_dir, "component.pt"))
    write_json({"selection": spec, "N": len(res["names"]), "hidden": int(res["w"].shape[0]),
                "n_by_method": res["n_by_method"], "lam": [float(x) for x in res["lam"][:8]],
                "null": res["null"], "cos": dict(zip(res["names"], map(float, res["cos"])))},
               os.path.join(out_dir, "component.json"))


def load_w(out_dir):
    return torch.load(os.path.join(out_dir, "component.pt"), map_location="cpu", weights_only=True)["w"].double()


def load_summary(out_dir):
    return read_json(os.path.join(out_dir, "component.json"))
