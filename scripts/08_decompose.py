"""Этап 8 — разложение DIM (GPU): bypass / induce / read для DIM, остатка DIM⊥w, w, w⊥DIM и
случайных направлений (см. src/shared_refusal/decompose.py).

DIM — объединённый (union) и по доменам, из dim/<…>/direction.pt этапа 1; w — компоненты
вариантов decompose.components (этап 2). Точка read и слой induce — точка DIM из manifest;
единица силы induce α0 = ‖DIM union‖. Промпты — test-сплиты без фильтра (базовые уровни
печатаются отдельно).

Пишет results/<запуск>/decompose/decompose.json и reports/decompose.md.
"""
import argparse
import functools
import os

import torch

from shared_refusal import decompose, models, provenance, report
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    dc = cfg["decompose"]
    domains = list(cfg["domains"])
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "decompose")

    point = run.manifest()["dim"]
    dims = {f"dim_{d}": torch.load(os.path.join(run.dim(d), "direction.pt"), map_location="cpu", weights_only=True)
            for d in dc["dims"]}
    alpha0 = float(dims["dim_union"].norm())
    ws = {v: load_w(run.component(v)) for v in dc["components"]}
    ds, energy = decompose.directions(dims, ws, dc["n_rand"], dc["rand_seed"])

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")

    model = models.load(cfg["model"], cfg["model_revision"])
    bs = dc["batch_size"]
    res = decompose.evaluate(
        functools.partial(models.last_token_logits, model, batch_size=bs),
        lambda p, v, c: models.steer_logits(model, p, v, c, point["layer"], bs),
        lambda p: models.resid_at(model, p, point["layer"], point["pos"], bs),
        fam["refusal_toks"], harmful, harmless, ds, dc["scales"], alpha0, dc["n_boot"], dc["boot_seed"],
        log=lambda s: print(s, flush=True))
    res.update({"energy": energy, "alpha0": alpha0, "point": {"pos": point["pos"], "layer": point["layer"]},
                "scales": dc["scales"]})

    write_json(res, run.path("decompose", "decompose.json"))
    run.write_report("decompose.md", report.decompose(cfg["model_alias"], res))
    run.update_manifest("decompose", {"energy": energy, "alpha0": alpha0, "directions": ds.names})
    provenance.record(run, cfg, "decompose")


if __name__ == "__main__":
    main()
