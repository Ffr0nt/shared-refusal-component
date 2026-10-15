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

from shared_refusal import ablation, models, provenance, report
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    ac = cfg["ablation"]
    domains = list(cfg["domains"])
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "ablation")

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    ds = ablation.directions_to_test(ac, run)

    model = models.load(cfg["model"], cfg["model_revision"])
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds,
                            ac["alphas"], log=lambda s: print(s, flush=True))

    write_json({**res, "n_test": {d: len(p) for d, p in harmful.items()}, "n_harmless": len(harmless)},
               run.path("ablation", "ablation.json"))
    run.write_report("ablation.md", report.ablation(cfg["model_alias"], res["base_asr"],
                                                    ablation.summary_table(res, domains), domains))
    run.update_manifest("ablation", {"directions": ds.names, "alphas": ac["alphas"]})
    provenance.record(run, cfg, "ablation")


if __name__ == "__main__":
    main()
