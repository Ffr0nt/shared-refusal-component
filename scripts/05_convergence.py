"""Этап 5 — сходимость общей компоненты по числу векторов (CPU).

Пул — направления этапа 1 после отбора convergence.selection (по умолчанию все DIM и RDO,
без различия между ними). Для N = 1 … n_max:
  - split-half (N ≤ пул/2): |cos(w_A, w_B)| для непересекающихся A, B по N — оценка ρ_N²
    (сходство с компонентой бесконечного числа векторов);
  - vs_full: cos²(w_S, w_всех) и λ1(S) для случайных S размера N (завышена при больших N);
и по n_orders случайным порядкам — сдвиг компоненты от ещё одного вектора.
По split-half подгоняется 1/ρ² − 1 = k/N (+ b) и считается, сколько векторов нужно для
целевых ρ². Не больше n_samples комбинаций на N; где их меньше — берутся все.

Пишет results/<запуск>/convergence/convergence.json и reports/convergence.md.
"""
import argparse
import math
import statistics as st

import torch

from shared_refusal import component, convergence as C, provenance, report
from shared_refusal.artifacts import DirectionSet, Run, write_json
from shared_refusal.config import load_config


def summary(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"mean": st.fmean(xs), "sd": st.pstdev(xs), "p05": q(0.05), "p50": q(0.5), "p95": q(0.95), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    cc = cfg["convergence"]
    provenance.check_upstream(run, cfg, "convergence")

    ds = component.select(DirectionSet.load(run.path("directions", "directions.pt")), cc["selection"] or {})
    V, n_pool = ds.V, len(ds)
    is_dim = [m.get("method") == "dim" for m in ds.meta]
    gen = torch.Generator().manual_seed(cc["seed"])
    n_max = min(cc["n_max"], n_pool)

    half, full = {}, {}
    for n in range(1, n_max + 1):
        if 2 * n <= n_pool:
            rows = C.split_half(V, n, cc["n_samples"], gen)
            for r in rows:
                r["n_dim"] = [sum(is_dim[i] for i in r["A"]), sum(is_dim[i] for i in r["B"])]
            half[n] = rows
        rows = C.vs_full(V, n, cc["n_samples"], gen)
        for r in rows:
            r["n_dim"] = sum(is_dim[i] for i in r["S"])
        full[n] = rows
        print(f"N={n:2d}  split-half {len(half.get(n, [])):4d} пар, ρ²≈"
              f"{st.fmean(r['r'] for r in half[n]) if n in half else float('nan'):.3f}   "
              f"vs_full {len(rows):4d}, cos²={st.fmean(r['cos2_full'] for r in rows):.3f}", flush=True)
    step_rows = C.steps(V, cc["n_orders"], gen)

    ns = sorted(half)
    rho2 = [st.fmean(r["r"] for r in half[n]) for n in ns]
    fits = {"no_plateau": C.fit(ns, rho2, plateau=False), "plateau": C.fit(ns, rho2, plateau=True)}
    need = {name: {str(t): C.n_for(t, **f) for t in cc["targets"]} for name, f in fits.items()}
    gains = {name: {str(n): C.gain(n, **f) for n in (5, 10, 20, 50)} for name, f in fits.items()}

    res = {"pool": ds.names, "n_pool": n_pool, "n_dim": sum(is_dim), "seed": cc["seed"],
           "split_half": {str(n): {"rho2": summary([r["r"] for r in half[n]]), "rows": half[n]} for n in ns},
           "vs_full": {str(n): {"cos2_full": summary([r["cos2_full"] for r in full[n]]),
                                "lam1": summary([r["lam1"] for r in full[n]]), "rows": full[n]} for n in full},
           "steps": {str(n + 1): summary([s[n] for s in step_rows]) for n in range(n_pool - 1)},
           "fit": fits, "n_for_target": need, "gain": gains}
    write_json(res, run.path("convergence", "convergence.json"))
    run.write_report("convergence.md", report.convergence(cfg["model_alias"], res, cc["targets"]))
    run.update_manifest("convergence", {"fit": fits, "n_for_target": {k: {t: (None if math.isinf(v) else v)
                                                                          for t, v in d.items()}
                                                                      for k, d in need.items()}})
    provenance.record(run, cfg, "convergence")
    for name, f in fits.items():
        print(f"{name}: k={f['k']:.3f} b={f['b']:.4f}  N для ρ²≥" +
              ", ".join(f"{t}: {v:.1f}" for t, v in need[name].items()))


if __name__ == "__main__":
    main()
