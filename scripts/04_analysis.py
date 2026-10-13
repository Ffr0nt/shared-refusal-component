"""Этап 4 — свойства общей компоненты, по вариантам (CPU).

Для w варианта и ПОЛНОГО набора этапа 1 (варианты различаются только тем, из чего
построена w):
  - доля энергии DIM каждого домена вдоль w (cos^2; случайный уровень 1/hidden);
  - средний cos^2(w, RDO) по сидам домена;
  - cos между DIM разных доменов;
  - контроль общего старта: cos^2(w, стартовый вектор) и cos(итог, старт) для каждого сида.

Пишет results/<запуск>/analysis/<вариант>/analysis.json и reports/analysis-<вариант>.md.
"""
import argparse
import os

from shared_refusal import analysis, component, provenance, report
from shared_refusal.artifacts import DirectionSet, Run, write_json
from shared_refusal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=None, help="по умолчанию — все из конфига")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    provenance.check_upstream(run, cfg, "analysis")

    ds = DirectionSet.load(run.path("directions", "directions.pt"))
    init = DirectionSet.load(run.path("directions", "init.pt"))
    done = run.manifest().get("analysis", {})
    for variant in args.variants or list(cfg["component"]["variants"]):
        w = component.load_w(run.component(variant))
        energy = analysis.dim_energy(w, ds)
        res = {"random_level": 1 / ds.hidden, "dim_energy": energy,
               "rdo_cos2_mean": analysis.mean_cos2_by_domain(w, ds, "rdo"),
               "dim_cos": analysis.dim_cosines(ds), "init_control": analysis.init_control(w, ds, init)}
        write_json(res, os.path.join(run.analysis(variant), "analysis.json"))
        run.write_report(f"analysis-{variant}.md",
                         report.analysis(cfg["model_alias"], variant, ds.hidden, energy, res["rdo_cos2_mean"],
                                         res["dim_cos"], res["init_control"]))
        done[variant] = {"dim_energy": energy}
        for d, e in energy.items():
            print(f"{variant} {d}: доля DIM вдоль w = {e:.3f}")
    run.update_manifest("analysis", done)
    provenance.record(run, cfg, "analysis")


if __name__ == "__main__":
    main()
