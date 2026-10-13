"""Шаг 04 — какую часть энергии каждого DIM составляет общая компонента (CPU).

Для единичных векторов доля энергии DIM_d вдоль w — это cos^2(w, DIM_d); остаток
1 - cos^2 лежит в ортогональном дополнении к w. Случайный уровень — E cos^2 = 1/hidden.

Рядом, для чтения этих чисел:
  - cos между DIM разных доменов (все в одной точке извлечения);
  - средний cos^2(w, RDO) по сидам домена;
  - контроль общего старта: cos^2(w, стартовый вектор) и cos(итог, старт) для каждого сида.

Пишет results/<model>/energy/energy.json и reports/energy.md.
"""
import argparse
import os

import torch

from shared_refusal.common import dim_dir, load_config, rdo_dir, results_dir, update_manifest, write_json, write_report
from shared_refusal.geometry import unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    domains = list(cfg["domains"])

    w = torch.load(results_dir(cfg, "component", "component.pt"), map_location="cpu", weights_only=True)["w"].double()
    hidden = w.shape[0]
    dims = {d: unit(torch.load(os.path.join(dim_dir(cfg, d), "direction.pt"), map_location="cpu", weights_only=True).double())
            for d in domains}

    rows, init_rows = {}, []
    for d in domains:
        rdo_cos2 = []
        for s in cfg["rdo"]["seeds"][d]:
            r = torch.load(os.path.join(rdo_dir(cfg, d), f"seed_{s:02d}.pt"), map_location="cpu", weights_only=True)
            v, v0 = unit(r["vector"].double()), unit(r["init_vector"].double())
            rdo_cos2.append(float((w @ v) ** 2))
            init_rows.append({"domain": d, "seed": s, "cos2_w_init": float((w @ v0) ** 2),
                              "cos_final_init": float(v @ v0)})
        rows[d] = {"dim_energy": float((w @ dims[d]) ** 2), "rdo_cos2_mean": sum(rdo_cos2) / len(rdo_cos2)}

    dim_cos = {a: {b: float(dims[a] @ dims[b]) for b in domains} for a in domains}
    summary = {"random_level": 1 / hidden, "domains": rows, "dim_cos": dim_cos, "init_control": init_rows}
    write_json(summary, results_dir(cfg, "energy", "energy.json"))
    update_manifest(cfg, "energy", {d: r["dim_energy"] for d, r in rows.items()})

    lines = [f"# Доля общей компоненты в DIM — {cfg['model_alias']}", "",
             f"Случайный уровень cos² = 1/hidden = {1 / hidden:.5f}.", "",
             "| домен | доля энергии DIM вдоль w | остаток | средний cos²(w, RDO) |", "|---|---|---|---|"]
    lines += [f"| {d} | **{r['dim_energy']:.3f}** | {1 - r['dim_energy']:.3f} | {r['rdo_cos2_mean']:.3f} |"
              for d, r in rows.items()]
    lines += ["", "cos между DIM доменов (одна точка извлечения):", "",
              "| | " + " | ".join(domains) + " |", "|---" * (len(domains) + 1) + "|"]
    lines += [f"| {a} | " + " | ".join(f"{dim_cos[a][b]:.3f}" for b in domains) + " |" for a in domains]
    c2 = [r["cos2_w_init"] for r in init_rows]
    cf = [r["cos_final_init"] for r in init_rows]
    lines += ["", "Контроль общего старта (20 сидов):", "",
              f"- cos²(w, старт): среднее {sum(c2) / len(c2):.5f}, максимум {max(c2):.5f}",
              f"- cos(итог, старт): среднее {sum(cf) / len(cf):.3f}, от {min(cf):.3f} до {max(cf):.3f}"]
    write_report(cfg, "energy.md", lines)
    for d, r in rows.items():
        print(f"{d}: доля DIM вдоль w = {r['dim_energy']:.3f}")


if __name__ == "__main__":
    main()
