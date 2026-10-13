"""Шаг 03 — общая компонента всех векторов (CPU).

Популяция: DIM каждого домена + одномерные конусы RDO всех сидов (сейчас 4 + 20 = 24),
все единичные, веса равные:
    P = (1/N) sum_i v_i v_i^T.
Общая компонента w — первый собственный вектор P; lam_1 = mean_i cos^2(w, v_i).
Знак w выбран так, чтобы cos(w, средний DIM) > 0 (добавление w наводит отказ).

Спектр, знак и нулевой уровень — shared_refusal.geometry.

Нулевой уровень: N независимых случайных единичных векторов той же размерности. Нижняя
граница lam_1 = 1/N (все векторы взаимно ортогональны), потолок 1 (все совпадают).

Пишет results/<model>/component/{component.pt, component.json} и reports/component.md.
"""
import argparse
import os

import torch

from shared_refusal.common import load_config, load_vectors, results_dir, update_manifest, write_json, write_report
from shared_refusal.geometry import null_lam1, shared_component


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["component"]

    names, V = load_vectors(cfg)
    N, hidden = V.shape
    is_dim = torch.tensor([n.startswith("dim_") for n in names])
    n_dim = int(is_dim.sum())
    mean_dim = V[is_dim].mean(0)
    w, lam, cos = shared_component(V, mean_dim)            # cos — со знаком, [N]

    null = null_lam1(N, hidden, cc["n_null"], cc["null_seed"])
    null_mean, null_p95, null_max = float(null.mean()), float(null.quantile(0.95)), float(null.max())

    out = results_dir(cfg, "component")
    os.makedirs(out, exist_ok=True)
    torch.save({"w": w.float(), "lam": lam[:8].float(), "names": names, "cos": cos.float()},
               f"{out}/component.pt")
    summary = {"N": N, "hidden": hidden, "lam": [float(x) for x in lam[:8]],
               "null": {"mean": null_mean, "p95": null_p95, "max": null_max, "n": cc["n_null"]},
               "cos": dict(zip(names, map(float, cos)))}
    write_json(summary, f"{out}/component.json")
    update_manifest(cfg, "component", {"lam1": float(lam[0]), "null_p95": null_p95})

    lines = [f"# Общая компонента — {cfg['model_alias']}", "",
             f"N = {N} векторов ({n_dim} DIM + {N - n_dim} RDO), равные веса, hidden = {hidden}.", "",
             "| | λ1 | λ2 | λ3 | нуль: среднее | нуль: p95 | нижняя граница 1/N |",
             "|---|---|---|---|---|---|---|",
             f"| | **{lam[0]:.3f}** | {lam[1]:.3f} | {lam[2]:.3f} | {null_mean:.3f} | {null_p95:.3f} | {1 / N:.3f} |",
             "", "cos(w, v) по векторам (знак w — по среднему DIM):", "",
             "| вектор | cos | cos² |", "|---|---|---|"]
    lines += [f"| {n} | {c:+.3f} | {c * c:.3f} |" for n, c in zip(names, cos.tolist())]
    write_report(cfg, "component.md", lines)
    print(f"lam1={lam[0]:.3f}  lam2={lam[1]:.3f}  нуль p95={null_p95:.3f}")


if __name__ == "__main__":
    main()
