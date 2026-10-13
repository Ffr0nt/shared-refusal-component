"""Этап 2 — общая компонента набора направлений, по вариантам отбора (CPU).

Вариант — правило отбора из directions/directions.pt (component.variants в конфиге):
methods / domains / exclude. Для каждого: w — первый собственный вектор (1/N) sum v v^T,
спектр, нулевой уровень (N случайных единичных векторов той же размерности, n_null раз).

Пишет results/<запуск>/component/<вариант>/{component.pt, component.json} и
reports/component-<вариант>.md.
"""
import argparse

from shared_refusal import component, report
from shared_refusal.artifacts import DirectionSet, Run
from shared_refusal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=None, help="по умолчанию — все из конфига")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    cc = cfg["component"]

    ds = DirectionSet.load(run.path("directions", "directions.pt"))
    done = run.manifest().get("component", {})
    for variant in args.variants or list(cc["variants"]):
        spec = cc["variants"][variant] or {}
        res = component.compute(component.select(ds, spec), cc["n_null"], cc["null_seed"])
        out = run.component(variant)
        component.save(res, out, spec)
        done[variant] = {"lam1": float(res["lam"][0]), "null_p95": res["null"]["p95"], "N": len(res["names"])}
        run.write_report(f"component-{variant}.md",
                         report.component(cfg["model_alias"], variant, spec, component.load_summary(out)))
        print(f"{variant}: N={len(res['names'])}  lam1={res['lam'][0]:.3f}  lam2={res['lam'][1]:.3f}  "
              f"нуль p95={res['null']['p95']:.3f}")
    run.update_manifest("component", done)


if __name__ == "__main__":
    main()
