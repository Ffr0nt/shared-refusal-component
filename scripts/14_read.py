"""Этап 14 — read при одной теме (GPU, только прямой проход): AUROC проекции активаций в точке DIM
на DIM, остаток DIM⊥w и n_rand случайных направлений — на XSTest (unsafe против safe) и на
test-сплитах SALAD (harmful против harmless, как этап 8). См. src/shared_refusal/read.py.

Активации сохраняются (read/acts.pt), так что новые направления считаются потом на CPU.
Пишет results/<запуск>/read/read.json и reports/read.md.
"""
import argparse
import os

import torch

from shared_refusal import models, provenance, read, report
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    rc = cfg["read"]
    domains = list(cfg["domains"])
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "read")

    from datasets import load_dataset
    xs = load_dataset(rc["dataset"], revision=rc["revision"], split=rc["split"])
    unsafe = [r for r in xs if r["label"] == "unsafe"]
    safe = [r for r in xs if r["label"] == "safe"]

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    point = run.manifest()["dim"]
    acts_path = run.path("read", "acts.pt")
    if os.path.exists(acts_path):
        acts = torch.load(acts_path, weights_only=True)
    else:
        model = models.load(cfg["model"], cfg["model_revision"])
        resid = lambda p: models.resid_at(model, p, point["layer"], point["pos"], rc["batch_size"])
        acts = {"xstest_unsafe": resid(models.prompts(fam, [r["prompt"] for r in unsafe])),
                "xstest_safe": resid(models.prompts(fam, [r["prompt"] for r in safe])),
                "salad_harmful": torch.cat([resid(test_prompts(d, "harmful")) for d in domains]),
                "salad_harmless": resid(test_prompts(domains[0], "harmless"))}
        os.makedirs(os.path.dirname(acts_path), exist_ok=True)
        torch.save(acts, acts_path)

    dims = {f"dim_{d}": torch.load(os.path.join(run.dim(d), "direction.pt"), map_location="cpu", weights_only=True)
            for d in rc["dims"]}
    ws = {v: load_w(run.component(v)) for v in rc["components"]}
    dirs, energy = read.directions(dims, ws, rc["n_rand"], rc["rand_seed"])

    groups = ([read.group(r["type"]) for r in unsafe], [read.group(r["type"]) for r in safe])
    res = {"xstest": read.evaluate_pair(acts["xstest_unsafe"], acts["xstest_safe"], dirs, rc["n_boot"],
                                        rc["boot_seed"], groups),
           "salad": read.evaluate_pair(acts["salad_harmful"], acts["salad_harmless"], dirs, rc["n_boot"],
                                       rc["boot_seed"]),
           "energy": energy, "point": {"pos": point["pos"], "layer": point["layer"]}}
    for pair in ("xstest", "salad"):
        r = res[pair]
        print(f"{pair}: rand mean {r['null']['mean']:.3f} q95 {r['null']['q95']:.3f} max {r['null']['max']:.3f}; " +
              "; ".join(f"{n} {x['auroc']:.3f} (p {x['p_rand']:.3f})" for n, x in r["rows"].items()), flush=True)

    write_json(res, run.path("read", "read.json"))
    run.write_report("read.md", report.read(cfg["model_alias"], res))
    run.update_manifest("read", {"n_rand": rc["n_rand"], "directions": [n for n in dirs if not n.startswith("rand_")]})
    provenance.record(run, cfg, "read")


if __name__ == "__main__":
    main()
