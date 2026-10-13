"""Этап 3 — аблация: снимает ли направление отказ (GPU).

Проверяемый набор (ablation в конфиге): общие компоненты из component.variants (имя
`w:<вариант>`), при base: true — все направления этапа 1 (DIM, RDO), и n_rand случайных.
Интервенция — частичная аблация во всех слоях (вход блока, выход attn, выход mlp):
h <- h - a (h·v) v, a ∈ alphas. Сравнивать направления честно только при близком KL.

Test-сплиты не фильтруются: базовый ASR без аблации печатается отдельно.

Пишет results/<запуск>/ablation/ablation.json и reports/ablation.md.
"""
import argparse
import functools
import os

import torch

from shared_refusal import ablation, component, models, report
from shared_refusal.artifacts import DirectionSet, Run, read_json, write_json
from shared_refusal.config import load_config
from shared_refusal.geometry import random_directions


def directions_to_test(cfg, run):
    ac = cfg["ablation"]
    base = DirectionSet.load(run.path("directions", "directions.pt"))
    ws = [component.load_w(run.component(v)) for v in ac["components"]]
    ds = DirectionSet([f"w:{v}" for v in ac["components"]], torch.stack(ws) if ws else torch.empty(0, base.hidden),
                      [{"method": "component", "domain": None, "variant": v} for v in ac["components"]])
    if ac["base"]:
        ds = ds + base
    rand = random_directions(base.hidden, ac["n_rand"], seed=ac["rand_seed"])
    if rand:
        ds = ds + DirectionSet.from_raw([f"rand_{i}" for i in range(len(rand))], rand,
                                        [{"method": "random", "domain": None} for _ in rand])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    ac = cfg["ablation"]
    domains = list(cfg["domains"])
    fam = models.family(cfg["model"])

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    ds = directions_to_test(cfg, run)

    model = models.load(cfg["model"])
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds,
                            ac["alphas"], log=lambda s: print(s, flush=True))

    write_json({**res, "n_test": {d: len(p) for d, p in harmful.items()}, "n_harmless": len(harmless)},
               run.path("ablation", "ablation.json"))
    run.write_report("ablation.md", report.ablation(cfg["model_alias"], res["base_asr"],
                                                    ablation.summary_table(res, domains), domains))
    run.update_manifest("ablation", {"directions": ds.names, "alphas": ac["alphas"]})


if __name__ == "__main__":
    main()
