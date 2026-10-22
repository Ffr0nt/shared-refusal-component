"""Этап 10 — кросс-нейтрализация (GPU): вызываем отказ одним направлением и одновременно вырезаем
другое во всех слоях.

Повтор блока «кросс-нейтрализация» из concept_dim/common_axis (там — на конусной оси w_cone) на
SRC-компоненте w и остатке DIM⊥w. Вызов — h + s*·α0·v̂ на входе слоя DIM (α0 = ‖DIM union‖),
вырезание — аблация a = 1 во всех слоях (вход блока, выход attn, выход mlp); на входе слоя DIM
сначала вырезание, потом прибавление. s* — рабочая точка вызова из этапа 8: s с наибольшей долей
отказа (отклик немонотонен — при перестреле отказ падает вместе с осмысленным выходом, и тест
мерил бы поломку). Пары:
  вызываем DIM: — / DIM / w / DIM⊥w / случайное;
  вызываем w:   — / w / DIM / DIM⊥w / случайное.
Если вырезание w гасит отказ, вызванный DIM, — они делят канал; если вырезание DIM⊥w не гасит — остаток
в вызове не участвует. Метрика — прокси первого токена на harmless_test (доля refusal_score > 0) с 95%
ДИ бутстрэпом по запросам.

Пишет results/<запуск>/cross/cross.json и reports/cross.md.
"""
import argparse
import os

import numpy as np

from shared_refusal import ablation, models, provenance, report
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config
from shared_refusal.strongreject import bootstrap_ci


def working_scale(dec, name):
    """s с наибольшей долей отказа при вызове в этапе 8 (при равенстве — меньший s)."""
    row = next(r for r in dec["rows"] if r["direction"] == name)
    return max(sorted(row["induce"].items(), key=lambda kv: float(kv[0])), key=lambda kv: kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    cc = cfg["cross"]
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "cross")

    dim, w = f"dim_{cc['dim']}", f"w:{cc['component']}"
    res = f"{dim}⊥{w}"
    ds = ablation.directions_to_test({"components": [cc["component"]], "dims": [cc["dim"]],
                                      "residuals": [{"dim": cc["dim"], "component": cc["component"]}],
                                      "base": False, "n_rand": cc["n_rand"], "rand_seed": cc["rand_seed"]}, run)
    rands = [n for n in ds.names if n.startswith("rand_")]
    dec = read_json(run.path("decompose", "decompose.json"))
    point = run.manifest()["dim"]
    alpha0 = float(ablation.load_dim(run, cc["dim"]).norm())
    assert abs(alpha0 - dec["alpha0"]) < 1e-3 * alpha0, "α0 этапа 8 не совпадает с ‖DIM‖ — другой DIM?"

    domain = next(iter(cfg["domains"]))
    rows = read_json(os.path.join(run.splits(domain), "harmless_test.json"))
    prompts = models.prompts(fam, [r["instruction"] for r in rows])
    model, tok = models.load_hf(cfg["model"], cfg["model_revision"])

    def rate(add, abl):
        out = models.generate(model, tok, prompts, None if abl is None else ds[abl], 1.0, 1, cc["batch_size"],
                              fam["refusal_toks"], add=add)
        s = np.array([sc for _, sc in out])
        return {**bootstrap_ci((s > 0).astype(float), cc["n_boot"], cc["boot_seed"]), "score_mean": float(s.mean())}

    base = rate(None, None)
    print(f"без вмешательства: отказ {base['mean']:.3f}", flush=True)
    out = []
    for ind in (dim, w):
        s, s_rate8 = working_scale(dec, ind)
        add = (ds[ind], float(s) * alpha0, point["layer"])
        for abl in [None, ind] + [x for x in (dim, w, res) if x != ind] + rands:
            r = {"induce": ind, "scale": float(s), "induce_rate_stage8": s_rate8, "ablate": abl or "—",
                 **rate(add, abl)}
            out.append(r)
            print(f"вызываем {ind:10s} s={s:<4} + вырезаем {r['ablate']:22s} -> отказ {r['mean']:.3f} "
                  f"[{r['lo']:.3f}, {r['hi']:.3f}]", flush=True)

    result = {"base": base, "rows": out, "alpha0": alpha0, "layer": point["layer"], "n": len(prompts)}
    write_json(result, run.path("cross", "cross.json"))
    lines = [f"# Кросс-нейтрализация — {cfg['model_alias']}", "",
             f"harmless_test {len(prompts)}; вызов h + s·α0·v̂ на входе слоя {point['layer']}, α0 = {alpha0:.2f}; "
             "s — рабочая точка вызова из этапа 8; вырезание a = 1 во всех слоях. Доля отказа — прокси первого "
             "токена (refusal_score > 0), 95% ДИ бутстрэпом по запросам.", "",
             f"Без вмешательства: {base['mean']:.3f}.", ""]
    lines += report.table(["вызываем", "s", "+ вырезаем", "доля отказа [95% ДИ]"],
                          [[r["induce"], f"{r['scale']:g}", r["ablate"],
                            f"**{r['mean']:.3f}** [{r['lo']:.3f}, {r['hi']:.3f}]"] for r in out])
    run.write_report("cross.md", lines)
    provenance.record(run, cfg, "cross")


if __name__ == "__main__":
    main()
