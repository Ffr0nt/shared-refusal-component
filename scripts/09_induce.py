"""Этап 9 — полные ответы на безобидные запросы при вызове отказа (GPU).

Вызов отказа (induce) этапа 8, но полными ответами, а не прокси первого токена: h + s·α0·v̂ на
входе слоя DIM на всех позициях и каждом шаге декодирования, α0 = ‖DIM union‖, s из
induce.scales. Набор направлений — как в ablation/generate (компоненты, DIM из dims, остатки
DIM⊥w из residuals, случайные); ответ без вмешательства ("none") есть всегда. Запросы —
harmless_test (общий для доменов, как в этапе 8). Ответы потом оценивает судья: отказала ли
модель на безобидном запросе.

Пишет results/<запуск>/induce/prompts.json ({id, prompt}) и
induce/generations/<направление>_s<s>.jsonl ({id, response, proxy_score}); готовые файлы
пропускаются.
"""
import argparse
import hashlib
import json
import os

from shared_refusal import ablation, models, provenance
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config

NONE = "none"


def gen_path(run, name):
    return run.path("induce", "generations", name.replace(":", "_") + ".jsonl")


def done(path, ids):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return [json.loads(line)["id"] for line in f] == ids


def answer_hash(cfg, point, alpha0):
    """То, что влияет на текст ответов: запросы, точка, α0 и параметры генерации (не набор направлений)."""
    ic = cfg["induce"]
    payload = {"splits": cfg.get("splits"), "point": point, "alpha0": round(alpha0, 6),
               "generate": {k: ic[k] for k in ("max_new_tokens", "batch_size")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    ic = cfg["induce"]
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "induce")

    point = run.manifest()["dim"]
    point = {"pos": point["pos"], "layer": point["layer"]}
    alpha0 = float(ablation.load_dim(run, "union").norm())
    h = answer_hash(cfg, point, alpha0)
    prev = run.manifest().get("induce", {}).get("answer_hash")
    if prev and prev != h and os.path.isdir(run.path("induce", "generations")):
        raise RuntimeError("ответы в induce/generations получены с другими запросами или параметрами; "
                           "удали эту папку или заведи новый запуск")

    domain = next(iter(cfg["domains"]))
    rows = read_json(os.path.join(run.splits(domain), "harmless_test.json"))
    kept = [{"id": i, "prompt": r["instruction"]} for i, r in enumerate(rows)]
    write_json(kept, run.path("induce", "prompts.json"))
    ids = [r["id"] for r in kept]

    ds = ablation.directions_to_test(ic, run)
    jobs = [(NONE, None, 0.0)] + [(f"{n}_s{s:g}", v, s) for n, v in zip(ds.names, ds.V) for s in ic["scales"]]
    todo = [j for j in jobs if not done(gen_path(run, j[0]), ids)]
    print(f"harmless_test {len(kept)}, α0 = {alpha0:.2f}, слой {point['layer']}; "
          f"заданий {len(jobs)}, уже готово {len(jobs) - len(todo)}")
    if todo:
        model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
        prompts = models.prompts(fam, [r["prompt"] for r in kept])
        for name, v, s in todo:
            add = None if v is None else (v, s * alpha0, point["layer"])
            out = models.generate(model, tok, prompts, None, 0, ic["max_new_tokens"], ic["batch_size"],
                                  fam["refusal_toks"], add=add)
            path = gen_path(run, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                for r, (text, score) in zip(kept, out):
                    f.write(json.dumps({"id": r["id"], "response": text, "proxy_score": score}, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
            refused = sum(sc > 0 for _, sc in out) / len(out)
            print(f"{name:28s} готово: доля отказа по прокси {refused:.3f}", flush=True)

    run.update_manifest("induce", {
        "answer_hash": h, "n": len(kept), "alpha0": alpha0, "point": point, "scales": ic["scales"],
        "jobs": [j[0] for j in jobs],
        "meta": {NONE: {"method": NONE, "domain": None, "scale": 0},
                 **{f"{n}_s{s:g}": {**mm, "direction": n, "scale": s}
                    for n, mm in zip(ds.names, ds.meta) for s in ic["scales"]}},
        "max_new_tokens": ic["max_new_tokens"]})
    provenance.record(run, cfg, "induce")


if __name__ == "__main__":
    main()
