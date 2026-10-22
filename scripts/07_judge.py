"""Этап 7 — оценка ответов этапа 6 рубрикой StrongREJECT (сеть, без GPU).

Судья — секция judge (gpt-4o-mini через OpenRouter, ключ OPENROUTER_API_KEY из окружения или из .env).
С сервера OpenRouter недоступен: этап запускается там, где есть выход к нему (например, на
Mac после rsync папки results/<запуск>/), и пишет результаты в ту же папку.

Оценки кэшируются построчно в strongreject/judge/<направление>.jsonl: повторный запуск
досылает только то, чего в кэше нет (ответы, на которых судья отказался, повторно не
отправляются — для этого есть --retry-unjudged). Когда оценено всё, пишет strongreject/strongreject.json и
reports/strongreject.md. Без ключа, если всё уже оценено, — только пересчёт сводки.

Сводка: средняя оценка StrongREJECT (0…1) с 95% ДИ — на всех оставленных запросах и только на
тех, где модель без аблации отказывает (по судье); доля ответов без отказа по судье, по
поиску фраз и по прокси первого токена; RDO — среднее по сидам домена, случайные — среднее.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared_refusal import provenance, report, strongreject
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def judge_path(run, name):
    return run.path("strongreject", "judge", name.replace(":", "_") + ".jsonl")


def judge_missing(run, names, prompts, cfg, workers, retry_unjudged=False, log=print):
    """Досылает судье ответы, которых ещё нет в кэше; возвращает число отправленных.
    Ответ, на котором судья отказался (score None), второй раз не отправляется: он почти всегда
    отказывается снова, а перезапуск ради сводки не должен стоить денег. retry_unjudged — всё же
    отправить такие ещё раз (например, после смены модели судьи)."""
    jc = cfg["judge"]
    jobs = []
    for name in names:
        gens = read_jsonl(run.path("strongreject", "generations", name.replace(":", "_") + ".jsonl"))
        have = {r["id"] for r in read_jsonl(judge_path(run, name))
                if r.get("score") is not None or not retry_unjudged}
        jobs += [(name, g) for g in gens if g["id"] not in have]
    if not jobs:
        return 0
    templates = strongreject.load_templates(cfg["strongreject"], run.path("strongreject", "source"))
    client = strongreject.openrouter_client(jc["timeout"])
    locks = {n: threading.Lock() for n in names}
    log(f"судье: {len(jobs)} ответов, {workers} потоков")

    def one(name, g):
        res = strongreject.judge_one(client, jc, templates, prompts[g["id"]]["prompt"], g["response"])
        row = {"id": g["id"], **res}
        with locks[name]:
            path = judge_path(run, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(one, n, g) for n, g in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            failed += fut.result()["score"] is None
            if k % 200 == 0 or k == len(futures):
                log(f"  {k}/{len(futures)}, без оценки {failed}")
    return len(jobs)


def collect(run, names, prompts):
    """{направление: строки} — последняя оценка на каждый id плюс поиск фраз и прокси."""
    out = {}
    for name in names:
        gens = {g["id"]: g for g in read_jsonl(run.path("strongreject", "generations",
                                                        name.replace(":", "_") + ".jsonl"))}
        judged = {}
        for r in read_jsonl(judge_path(run, name)):
            if r.get("score") is not None or r["id"] not in judged:
                judged[r["id"]] = r
        out[name] = [{**judged.get(i, {"score": None, "refusal": None}), "id": i,
                      "substring_refusal": float(strongreject.substring_refusal(g["response"])),
                      "proxy_refusal": float(g["proxy_score"] > 0)} for i, g in gens.items()]
    return out


def groups_of(names, meta_by_name, domains):
    """Строки сводки: без вмешательства, компоненты, DIM и остатки DIM⊥w поштучно, RDO — по доменам,
    случайные — вместе."""
    g = [("none", ["none"])]
    g += [(n, [n]) for n in names if meta_by_name[n]["method"] in ("component", "dim", "residual")]
    for d in domains:
        rdo = [n for n in names if meta_by_name[n]["method"] == "rdo" and meta_by_name[n]["domain"] == d]
        if rdo:
            g.append((f"rdo_{d} ({len(rdo)} сидов)", rdo))
    rand = [n for n in names if meta_by_name[n]["method"] == "random"]
    if rand:
        g.append(("rand (среднее)", rand))
    return g


def kl_at(run, alpha):
    """KL на harmless_test при той же силе аблации: {направление: KL}. Из этапа 3, если он есть;
    чего там нет (DIM union, остатки; у моделей без этапа 3 — всё) — из этапа 8, где аблация
    a = 1 на том же harmless_test. В ablation.json прогона 12.10 единственная компонента названа
    `w` — для неё заводим и нынешнее имя `w:all` (пересчёт этапа 3 новым кодом записал бы его сам)."""
    kl = {}
    p = run.path("ablation", "ablation.json")
    if os.path.exists(p):
        kl = {r["direction"]: r["kl"] for r in read_json(p)["rows"] if r["alpha"] == alpha}
        if "w" in kl:
            kl.setdefault("w:all", kl["w"])
    p = run.path("decompose", "decompose.json")
    if alpha == 1.0 and os.path.exists(p):
        for r in read_json(p)["rows"]:
            kl.setdefault(r["direction"], r["kl"])
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=16, help="параллельных запросов к судье (на числа не влияет)")
    ap.add_argument("--retry-unjudged", action="store_true",
                    help="отправить судье и те ответы, на которых он уже отказался")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    jc = cfg["judge"]
    provenance.check_upstream(run, cfg, "judge")

    m = run.manifest()["strongreject"]
    names = m["directions"]
    prompts = {r["id"]: r for r in read_json(run.path("strongreject", "prompts.json"))}
    judge_missing(run, names, prompts, cfg, args.workers, args.retry_unjudged)

    judged = collect(run, names, prompts)
    missing = {n: sum(r["score"] is None for r in rows) for n, rows in judged.items()}
    if any(missing.values()):
        print("без оценки после всех попыток: " + ", ".join(f"{n}: {k}" for n, k in missing.items() if k))

    groups = groups_of(names[1:], m["meta"], list(cfg["domains"]))
    ids = sorted(r["id"] for r in judged["none"])
    base_refused = [r["id"] for r in judged["none"] if r.get("refusal") == 1.0]
    categories = {i: prompts[i]["category"] for i in ids}
    rows = strongreject.summarize(judged, groups, ids, base_refused, categories, jc["n_boot"], jc["boot_seed"])
    kl = kl_at(run, m["alpha"])
    for r, (_, members) in zip(rows, groups):
        vals = [kl.get(n, 0.0 if n == "none" else None) for n in members]
        r["kl"] = sum(vals) / len(vals) if None not in vals else None
    pairs = [p for p in jc.get("pairs", []) if p[0] in judged and p[1] in judged]
    diffs = strongreject.paired_diffs(judged, pairs, ids, jc["n_boot"], jc["boot_seed"])
    agree = strongreject.agreement(judged, names)
    served = sorted({r.get("judge_model") for rs in judged.values() for r in rs if r.get("judge_model")})

    res = {"n_prompts": len(ids), "n_base_refused": len(base_refused), "rows": rows, "diffs": diffs, "agreement": agree,
           "judge_model": jc["model"], "served_models": served, "missing": missing}
    write_json(res, run.path("strongreject", "strongreject.json"))
    run.write_report("strongreject.md", report.strongreject(cfg["model_alias"], res, m))
    run.update_manifest("judge", {"model": jc["model"], "served_models": served, "missing": sum(missing.values()),
                                  "n_prompts": len(ids), "n_base_refused": len(base_refused)})
    provenance.record(run, cfg, "judge")


if __name__ == "__main__":
    main()
