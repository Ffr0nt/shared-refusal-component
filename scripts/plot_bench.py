"""Инфографика к сводке способностей при аблации: GSM8K и ARC-Challenge по моделям (CPU, без сети).

Те же данные, что в results/cross-model-bench.md (scripts/summary_bench.py): парная разница к модели
без вмешательства с 95% ДИ для w, DIM, остатка, RDO и случайных направлений, плюс прямое сравнение
w − DIM. Закрашенный маркер — ДИ не накрывает 0. Пишет results/figures/cross-model-bench.{png,svg}.

  uv run python scripts/plot_bench.py
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared_refusal.artifacts import REPO, Run, read_json
from shared_refusal.config import load_config

DIM, W, RES = "dim_union", "w:all", "dim_union⊥w:all"
TASKS = [("gsm8k", "GSM8K", "exact match, 5-shot"), ("arc_challenge", "ARC-Challenge", "acc_norm, 0-shot")]
# направление, подпись, цвет, маркер; rand — серый контроль
SERIES = [(W, "w", "#2a78d6", "o"), (DIM, "DIM", "#eb6834", "s"), (RES, "остаток DIM⊥w", "#1baf7a", "D"),
          ("rdo:split", "RDO", "#eda100", "^"), ("rand:mean", "rand", "#8a8984", "v")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(run, *parts):
    p = run.path(*parts)
    return read_json(p) if os.path.exists(p) else None


def sig(c):
    return c["lo"] > 0 or c["hi"] < 0


def point(ax, y, c, color, marker, size=6.5):
    pp = 100                                            # доли → п.п.
    ax.plot([c["lo"] * pp, c["hi"] * pp], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(c["mean"] * pp, y, marker=marker, ms=size, mec=color, mew=1.6,
            mfc=color if sig(c) else SURFACE, zorder=3, ls="none")


def style(ax, title, subtitle, n_rows):
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for i in range(0, n_rows, 2):                       # полосы по моделям
        ax.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0, lw=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel("Δ к модели без вмешательства, п.п.", color=MUTED, fontsize=9)
    ax.set_title(f"{title}\n", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "configs", "*_salad-*.yaml"))))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "figures", "cross-model-bench"))
    args = ap.parse_args()
    runs = []
    for p in args.configs:
        cfg = load_config(p)
        runs.append((cfg["model_alias"], Run.from_config(cfg)))
    size = lambda a: float(re.search(r"(\d+(?:\.\d+)?)[Bb]", a).group(1))
    runs.sort(key=lambda a: (a[0].split("-")[0].replace("Meta", "Llama"), size(a[0]), a[0]))
    data = {(a, t): load(r, "bench", t, "summary.json") for a, r in runs for t, _, _ in TASKS}
    missing = [a for a, _ in runs if not any(data[a, t] for t, _, _ in TASKS)]
    runs = [(a, r) for a, r in runs if a not in missing]
    names = [a.replace("-Instruct", "").replace("Meta-", "") for a, _ in runs]
    n = len(runs)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.62 * n + 2.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.8], "wspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    offs = [(-0.3 + 0.15 * k) for k in range(len(SERIES))]
    for ax, (task, title, metric) in zip(axes, TASKS):
        style(ax, title, f"{metric}; справа — точность без вмешательства", n)
        for i, (a, _) in enumerate(runs):
            sm = data[a, task]
            if sm is None:
                continue
            by = {r["direction"]: r for r in sm["rows"]}
            for (d, _, color, marker), o in zip(SERIES, offs):
                if by.get(d) and by[d]["vs_base"]:
                    point(ax, i + o, by[d]["vs_base"], color, marker)
            ax.text(1.0, i, f"{by['none']['mean']:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8, color=MUTED)

    ax = axes[2]                                        # прямое сравнение w − DIM
    style(ax, "w − DIM", "парная разница; < 0 — w портит сильнее", n)
    for i, (a, _) in enumerate(runs):
        for (task, _, _), o, marker in zip(TASKS, (-0.13, 0.13), ("o", "s")):
            sm = data[a, task]
            if sm is None:
                continue
            pr = {(p["a"], p["b"]): p for p in sm["pairs"]}.get((W, DIM))
            if pr:
                point(ax, i + o, pr, INK, marker, size=6)
    ax.set_xlabel("Δ (w − DIM), п.п.", color=MUTED, fontsize=9)

    lim = max(abs(v) for v in axes[0].get_xlim() + axes[1].get_xlim())
    for a in axes[:2]:
        a.set_xlim(-lim, lim)
    l3 = max(abs(v) for v in axes[2].get_xlim())
    axes[2].set_xlim(-l3, l3)
    axes[0].set_yticks(range(n), names, fontsize=10, color=INK)

    handles = [Line2D([], [], color=c, marker=m, ms=7, mfc=c, lw=1.6, label=l) for _, l, c, m in SERIES]
    handles += [Line2D([], [], ls="none", label="   "),
                Line2D([], [], color=MUTED, marker="o", ms=7, mfc=MUTED, lw=0, label="ДИ не накрывает 0"),
                Line2D([], [], color=MUTED, marker="o", ms=7, mfc=SURFACE, lw=0, label="ДИ накрывает 0"),
                Line2D([], [], ls="none", label="   "),
                Line2D([], [], color=INK, marker="o", ms=6, mfc=SURFACE, lw=1.6, label="w − DIM: GSM8K"),
                Line2D([], [], color=INK, marker="s", ms=6, mfc=SURFACE, lw=1.6, label="w − DIM: ARC")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=MUTED, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.1)
    fig.suptitle("Аблация направлений отказа почти не задевает способности модели", x=0.125, y=0.995,
                 ha="left", fontsize=15, fontweight="bold", color=INK)
    note = "Аблация α = 1.0 во всех слоях, salad-o1o2o5o6. Линии — 95% ДИ, бутстреп по вопросам (GSM8K n = 1319, ARC n = 1172)."
    if missing:
        note += f" Не посчитаны: {', '.join(missing)}."
    fig.text(0.125, 0.955, note, ha="left", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.86, bottom=0.12)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"[рисунок: {args.out}.png, .svg]")


if __name__ == "__main__":
    main()
