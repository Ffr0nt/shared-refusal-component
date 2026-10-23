"""Этап 6 — полные ответы на StrongREJECT при аблации (GPU).

Запросы StrongREJECT (секция strongreject), кроме совпадающих с нашими harmful train/val —
на них строились направления. Для каждого направления из секции generate (как в этапе 3:
компоненты, при base — все DIM и RDO, случайные) и без вмешательства ("none") — жадный ответ
до max_new_tokens токенов при аблации силы alpha во всех слоях, плюс прокси первого шага
(refusal_score) на тех же ответах.

С generate.rdo_split каждый RDO отвечает только на свою 1/20 запросов (strongreject.rdo_split,
раздача — в manifest): вместе сиды покрывают набор один раз, и этап 7 оценивает среднее по RDO
за цену одного направления. Готовый файл с полным набором тоже засчитывается.

Пишет results/<запуск>/strongreject/prompts.json (все запросы с итогом проверки пересечения),
strongreject/generations/<направление>.jsonl ({id, response, proxy_score}); готовые файлы
пропускаются.
"""
import argparse
import hashlib
import json
import os

from shared_refusal import ablation, models, provenance, strongreject
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config

NONE = "none"


def gen_path(run, name):
    return run.path("strongreject", "generations", name.replace(":", "_") + ".jsonl")


def done(path, ids, superset=False):
    """Файл есть и в нём ровно эти запросы в этом порядке (superset — хотя бы эти)."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        have = [json.loads(line)["id"] for line in f]
    return set(ids) <= set(have) if superset else have == ids


def answer_hash(cfg):
    """Хэш того, что влияет на сам текст ответа: секция strongreject (набор запросов) и
    параметры генерации. batch_size включён: в bf16 разбиение на батчи может чуть сдвинуть
    числа. Набор направлений не включён — он определяет, какие файлы появятся, а не
    содержимое уже посчитанных, поэтому добавить направления можно к готовым ответам."""
    keys = ("alpha", "max_new_tokens", "batch_size")
    payload = {"strongreject": cfg["strongreject"], "generate": {k: cfg["generate"][k] for k in keys}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_own(run, cfg):
    """Готовые ответы сгенерированы с теми же запросами и параметрами — иначе не дописываем."""
    prev = run.manifest().get("strongreject", {}).get("answer_hash")
    if prev and prev != answer_hash(cfg) and os.path.isdir(run.path("strongreject", "generations")):
        raise RuntimeError("ответы в strongreject/generations сгенерированы с другими запросами или "
                           "параметрами генерации; удали эту папку или заведи новый запуск")


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
    all_ids = [r["id"] for r in kept]
    rdo = [n for n, mm in zip(ds.names, ds.meta) if mm["method"] == "rdo"]
    split = strongreject.rdo_split(all_ids, rdo, gc["rdo_split"]["seed"]) if gc.get("rdo_split") and rdo else {}
    rows_for = lambda n: [r for r in kept if r["id"] in set(split[n])] if n in split else kept
    todo = [(n, v) for n, v in [(NONE, None)] + list(zip(ds.names, ds.V))
            if not done(gen_path(run, n), [r["id"] for r in rows_for(n)], superset=n in split)]
    print(f"направлений {len(ds) + 1}, уже готово {len(ds) + 1 - len(todo)}"
          + (f"; RDO по 1/{len(split)} запросов ({len(all_ids)} на всех)" if split else ""))
    if todo:
        model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
        for name, v in todo:
            sub = rows_for(name)
            out = models.generate(model, tok, models.prompts(fam, [r["prompt"] for r in sub]), v, gc["alpha"],
                                  gc["max_new_tokens"], gc["batch_size"], fam["refusal_toks"])
            path = gen_path(run, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                for r, (text, score) in zip(sub, out):
                    f.write(json.dumps({"id": r["id"], "response": text, "proxy_score": score}, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
            refused = sum(s > 0 for _, s in out) / len(out)
            print(f"{name:14s} готово: доля отказа по прокси {refused:.3f}", flush=True)

    run.update_manifest("strongreject", {
        "n_total": len(rows), "n_kept": len(kept), "answer_hash": answer_hash(cfg),
        "overlap": [{"id": r["id"], "jaccard": r["overlap"]["jaccard"], "exact": r["overlap"]["exact"]} for r in dropped],
        "directions": [NONE] + ds.names,
        "meta": {NONE: {"method": NONE, "domain": None},
                 **{n: {"method": mm["method"], "domain": mm["domain"]} for n, mm in zip(ds.names, ds.meta)}},
        "rdo_split": split or None,
        "alpha": gc["alpha"], "max_new_tokens": gc["max_new_tokens"]})
    provenance.record(run, cfg, "generate")


if __name__ == "__main__":
    main()
