"""Сводка по моделям: способности при аблации — GSM8K и ARC-Challenge (CPU, без сети).

Собирает из results/<запуск>/bench/<задача>/summary.json (scripts/12_bench.py) парные разницы
к модели без вмешательства для DIM union, w, остатка DIM⊥w, отдельных RDO и случайных направлений.
Чего ещё нет — прочерк. Пишет results/cross-model-bench.md.

  uv run python scripts/summary_bench.py                 # все salad-конфиги из configs/
  uv run python scripts/summary_bench.py --configs configs/qwen2.5-3b_salad-o1o2o5o6.yaml
"""
import argparse
import glob
import os
from datetime import date

from shared_refusal.artifacts import REPO, Run, read_json
from shared_refusal.config import load_config
from shared_refusal.report import table

DIM, W, RES = "dim_union", "w:all", "dim_union⊥w:all"
TASKS = [("gsm8k", "GSM8K", "exact_match (flexible-extract), 5-shot"),
         ("arc_challenge", "ARC-Challenge", "acc_norm, 0-shot")]
COLS = [(W, "w"), (DIM, "DIM"), (RES, "остаток"), ("rdo:split", "RDO"), ("rand:mean", "rand")]


def load(run, *parts):
    p = run.path(*parts)
    return read_json(p) if os.path.exists(p) else None


def delta(c):
    if not c:
        return "—"
    s = f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
    return f"**{s}**" if c["lo"] > 0 or c["hi"] < 0 else s


def task_row(alias, sm):
    if sm is None:
        return [alias] + ["—"] * (2 + len(COLS) + 1)
    by = {r["direction"]: r for r in sm["rows"]}
    pairs = {(p["a"], p["b"]): p for p in sm["pairs"]}
    base = by["none"]
    return [alias, base["n"], f"{base['mean']:.3f} [{base['lo']:.3f}, {base['hi']:.3f}]"] + \
           [delta((by.get(d) or {}).get("vs_base")) for d, _ in COLS] + [delta(pairs.get((RES, DIM)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "configs", "*_salad-*.yaml"))))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "cross-model-bench.md"))
    args = ap.parse_args()
    runs = []
    for p in args.configs:
        cfg = load_config(p)
        runs.append((cfg["model_alias"], Run.from_config(cfg)))
    order = lambda a: (a[0].startswith("Llama") or a[0].startswith("Meta"), a[0])
    runs.sort(key=order)

    lines = ["# Способности при аблации по моделям — сводка", "",
             f"Собрано {date.today():%d.%m.%Y} скриптом `scripts/summary_bench.py` из `bench/*/summary.json` "
             "(scripts/12_bench.py). Аблация α = 1.0 во всех слоях, chat template. DIM — объединённый (union), "
             "w — компонента `all`, остаток — DIM⊥w, RDO — отдельные RDO-векторы с разбивкой вопросов между ними "
             "(rdo:split), rand — среднее по случайным направлениям. Прочерк — для модели не посчитано.", "",
             "Без вмешательства — абсолютная точность; остальные столбцы — парная разница к нему на тех же "
             "вопросах, 95% ДИ бутстрепом по вопросам. Жирным — ДИ не накрывает 0.", ""]
    for i, (task, name, metric) in enumerate(TASKS, 1):
        lines += [f"## {i}. {name}", "", f"Метрика: {metric}.", ""]
        lines += table(["модель", "n", "без вмешательства"] + [f"Δ {c}" for _, c in COLS] + ["остаток − DIM"],
                       [task_row(a, load(r, "bench", task, "summary.json")) for a, r in runs])
        lines += [""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[сводка: {args.out}]")


if __name__ == "__main__":
    main()
