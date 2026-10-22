"""Сводка по моделям: судья для DIM union, w и остатка DIM⊥w (CPU, без сети).

Собирает из results/<запуск>/ трёх этапов:
  strongreject/strongreject.json (этап 7) — снятие отказа на StrongREJECT, оценка судьи и KL;
  induce/induce.json (этап 11) — вызов отказа на harmless_test, классификатор XSTest;
  cross/cross.json (этап 10) — кросс-нейтрализация, прокси.
Чего ещё нет — прочерк. Пишет results/cross-model-judge.md.

  uv run python scripts/summary_judge.py                 # все конфиги из configs/
  uv run python scripts/summary_judge.py --configs configs/qwen2.5-3b_salad-o1o2o5o6.yaml
"""
import argparse
import glob
import os
from datetime import date

from shared_refusal.artifacts import REPO, Run, read_json
from shared_refusal.config import load_config
from shared_refusal.report import table

DIM, W, RES = "dim_union", "w:all", "dim_union⊥w:all"


def load(run, *parts):
    p = run.path(*parts)
    return read_json(p) if os.path.exists(p) else None


def ci(c):
    return "—" if not c or c["mean"] is None else f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def val(c, bold=False):
    if not c or c["mean"] is None:
        return "—"
    return f"**{c['mean']:.3f}**" if bold else f"{c['mean']:.3f}"


def bypass_row(alias, sr):
    if sr is None:
        return [alias] + ["—"] * 9
    by = {r["direction"]: r for r in sr["rows"]}
    d = {(x["a"], x["b"]): x for x in sr.get("diffs", [])}
    rand = next((r for r in sr["rows"] if r["direction"].startswith("rand")), None)

    def cell(name, bold=False):
        r = by.get(name)
        if r is None:
            return "—"
        kl = "" if r.get("kl") is None else f" (KL {r['kl']:.3f})"
        return val(r["score"], bold) + kl

    return [alias, sr["n_prompts"], cell("none"), cell(DIM), cell(W), cell(RES, True),
            "—" if rand is None else val(rand["score"]),
            ci(d.get((RES, "none"), {}).get("score")), ci(d.get((W, DIM), {}).get("score")),
            sum(r["n_unjudged"] for r in sr["rows"])]


def induce_rows(alias, ind):
    if ind is None:
        return [[alias, "—"] + ["—"] * 6]
    out = []
    meta, rows = ind["meta"], ind["rows"]
    d = {(x["a"], x["b"]): x for x in ind["diffs"]}
    for s in sorted({m["scale"] for m in meta.values() if m["scale"]}):
        job = lambda n: f"{n}_s{s:g}"
        cell = lambda n, bold=False: val(rows[job(n)]["any_refusal"], bold) if job(n) in rows else "—"
        rand = next((j for j, m in meta.items() if m["method"] == "random" and m["scale"] == s), None)
        out.append([alias, f"{s:g}", val(rows["none"]["any_refusal"]), cell(DIM), cell(W), cell(RES, True),
                    "—" if rand is None else val(rows[rand]["any_refusal"]),
                    ci(d.get((job(RES), "none"), {}).get("any_refusal"))])
    return out


def cross_rows(alias, cr):
    if cr is None:
        return [[alias, "—", "—"] + ["—"] * 5]
    by = {(r["induce"], r["ablate"]): r for r in cr["rows"]}
    out = []
    for ind, abls in ((DIM, ["—", W, DIM, RES, "rand_0"]), (W, ["—", DIM, W, RES, "rand_0"])):
        s = next((r["scale"] for r in cr["rows"] if r["induce"] == ind), None)
        out.append([alias, ind, "—" if s is None else f"{s:g}"] +
                   [val(by[(ind, a)]) if (ind, a) in by else "—" for a in abls])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "configs", "*.yaml"))))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "cross-model-judge.md"))
    args = ap.parse_args()
    runs = []
    for p in args.configs:
        cfg = load_config(p)
        runs.append((cfg["model_alias"], Run.from_config(cfg)))
    order = lambda a: (a[0].startswith("Llama") or a[0].startswith("Meta"), a[0])
    runs.sort(key=order)

    lines = [f"# Судья для DIM, w и остатка DIM⊥w по моделям — сводка", "",
             f"Собрано {date.today():%d.%m.%Y} скриптом `scripts/summary_judge.py` из отчётов этапов 7, 10 и 11. "
             "DIM — объединённый (union), w — компонента `all`, остаток — DIM⊥w (как в этапе 8). Прочерк — этап "
             "для модели ещё не посчитан.", "",
             "## 1. Снятие отказа: StrongREJECT, судья", "",
             "Средняя оценка рубрики StrongREJECT (0…1) при аблации a = 1 во всех слоях; в скобках KL на harmless_test "
             "(этап 3, иначе этап 8). Δ — парная разница на тех же запросах с 95% ДИ; ДИ не накрывает 0 — "
             "разница значима. «б/о» — ответов без оценки судьи во всех строках.", ""]
    lines += table(["модель", "запросов", "без вмешательства", "DIM", "w", "остаток", "rand", "Δ остаток − без",
                    "Δ w − DIM", "б/о"],
                   [bypass_row(a, load(r, "strongreject", "strongreject.json")) for a, r in runs])
    lines += ["", "## 2. Вызов отказа: harmless_test, классификатор XSTest", "",
              "Доля любого отказа (полный + частичный) при h + s·α0·v̂ на входе слоя DIM, полные ответы.", ""]
    lines += table(["модель", "s", "без вмешательства", "DIM", "w", "остаток", "rand", "Δ остаток − без"],
                   [row for a, r in runs for row in induce_rows(a, load(r, "induce", "induce.json"))])
    lines += ["", "## 3. Кросс-нейтрализация (прокси первого токена)", "",
              "Доля отказа на harmless_test: вызываем отказ направлением в рабочей точке этапа 8 (s) и одновременно "
              "вырезаем другое во всех слоях.", ""]
    lines += table(["модель", "вызываем", "s", "одно", "− w / − DIM", "− само", "− остаток", "− rand"],
                   [row for a, r in runs for row in cross_rows(a, load(r, "cross", "cross.json"))])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[сводка: {args.out}]")


if __name__ == "__main__":
    main()
