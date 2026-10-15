"""Этап 6 — полные ответы на StrongREJECT при аблации (GPU).

Запросы StrongREJECT (секция strongreject), кроме совпадающих с нашими harmful train/val —
на них строились направления. Для каждого направления из секции generate (как в этапе 3:
компоненты, при base — все DIM и RDO, случайные) и без вмешательства ("none") — жадный ответ
до max_new_tokens токенов при аблации силы alpha во всех слоях, плюс прокси первого шага
(refusal_score) на тех же ответах.

Пишет results/<запуск>/strongreject/prompts.json (все запросы с итогом проверки пересечения),
strongreject/generations/<направление>.jsonl ({id, response, proxy_score}); готовые файлы
пропускаются.
"""
import argparse
import json
import os

from shared_refusal import ablation, models, provenance, strongreject
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config

NONE = "none"


def gen_path(run, name):
    return run.path("strongreject", "generations", name.replace(":", "_") + ".jsonl")


def done(path, ids):
    """Файл есть и в нём ровно эти запросы в этом порядке."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return [json.loads(line)["id"] for line in f] == ids


def check_own(run, cfg):
    """Готовые ответы посчитаны с этим же конфигом этапа — иначе не дописываем к ним новые."""
    prev = run.manifest().get("provenance", {}).get("generate")
    if prev and prev["config_hash"] != provenance.stage_hash(cfg, "generate") \
            and os.path.isdir(run.path("strongreject", "generations")):
        raise RuntimeError("ответы в strongreject/generations посчитаны с другим конфигом этапа 6; "
                           "удали эту папку или заведи новый запуск")


def select_prompts(cfg, run):
    """Все запросы с итогом проверки пересечения и список оставленных."""
    sc = cfg["strongreject"]
    prompts = strongreject.load_prompts(sc, run.path("strongreject", "source"))
    corpus = [r["instruction"] for d in cfg["domains"] for s in ("train", "val")
              for r in read_json(os.path.join(run.splits(d), f"harmful_{s}.json"))]
    ov = strongreject.overlap(prompts, corpus, sc["overlap_jaccard"])
    rows = [{**p, "overlap": o} for p, o in zip(prompts, ov)]
    return rows, [r for r in rows if not r["overlap"]["match"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    gc = cfg["generate"]
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "generate")
    check_own(run, cfg)

    rows, kept = select_prompts(cfg, run)
    write_json(rows, run.path("strongreject", "prompts.json"))
    dropped = [r for r in rows if r["overlap"]["match"]]
    print(f"StrongREJECT: {len(rows)} запросов, совпадают с train/val: {len(dropped)}, остаётся {len(kept)}")

    ds = ablation.directions_to_test(gc, run)
    todo = [(n, v) for n, v in [(NONE, None)] + list(zip(ds.names, ds.V)) if not done(gen_path(run, n), [r["id"] for r in kept])]
    print(f"направлений {len(ds) + 1}, уже готово {len(ds) + 1 - len(todo)}")
    if todo:
        model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
        prompts = models.prompts(fam, [r["prompt"] for r in kept])
        for name, v in todo:
            out = models.generate(model, tok, prompts, v, gc["alpha"], gc["max_new_tokens"], gc["batch_size"],
                                  fam["refusal_toks"])
            path = gen_path(run, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                for r, (text, score) in zip(kept, out):
                    f.write(json.dumps({"id": r["id"], "response": text, "proxy_score": score}, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
            refused = sum(s > 0 for _, s in out) / len(out)
            print(f"{name:14s} готово: доля отказа по прокси {refused:.3f}", flush=True)

    run.update_manifest("strongreject", {
        "n_total": len(rows), "n_kept": len(kept),
        "overlap": [{"id": r["id"], "jaccard": r["overlap"]["jaccard"], "exact": r["overlap"]["exact"]} for r in dropped],
        "directions": [NONE] + ds.names,
        "meta": {NONE: {"method": NONE, "domain": None},
                 **{n: {"method": mm["method"], "domain": mm["domain"]} for n, mm in zip(ds.names, ds.meta)}},
        "alpha": gc["alpha"], "max_new_tokens": gc["max_new_tokens"]})
    provenance.record(run, cfg, "generate")


if __name__ == "__main__":
    main()
