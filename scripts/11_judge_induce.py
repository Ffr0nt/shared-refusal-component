"""Этап 11 — судья для ответов этапа 9: отказала ли модель на безобидном запросе (сеть, без GPU).

Классификатор XSTest (src/shared_refusal/xstest.py): полное выполнение / полный отказ / частичный
отказ; модель и параметры — секция judge (gpt-4o-mini через OpenRouter, как этап 7). Запускается
там, где есть выход к OpenRouter (Mac), по копии results/<запуск>/induce/.

Оценки кэшируются построчно в induce/judge/<задание>.jsonl, повторный запуск досылает только
недостающее. Сводка — доля полного отказа и любого отказа (полный + частичный) с 95% ДИ, доля
отказа по прокси первого токена на тех же ответах, и парные разницы на одних и тех же запросах:
остаток − без вмешательства, w − DIM, случайное − без вмешательства (при каждом s). Неоценённые
ответы дают диапазон: нижняя граница — по оценённым, верхняя — считая такой ответ отказом
(верхняя граница для вызова отказа). Пишет induce/induce.json и reports/induce.md.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from shared_refusal import provenance, report, strongreject, xstest
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config

KEYS = ("full_refusal", "any_refusal")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def gen_path(run, job):
    return run.path("induce", "generations", job.replace(":", "_") + ".jsonl")


def judge_path(run, job):
    return run.path("induce", "judge", job.replace(":", "_") + ".jsonl")


def judge_missing(run, jobs, prompts, jc, workers, log=print):
    todo = []
    for job in jobs:
        have = {r["id"] for r in read_jsonl(judge_path(run, job)) if r.get("cls") is not None}
        todo += [(job, g) for g in read_jsonl(gen_path(run, job)) if g["id"] not in have]
    if not todo:
        return 0
    client = strongreject.openrouter_client(jc["timeout"])
    locks = {j: threading.Lock() for j in jobs}
    log(f"судье: {len(todo)} ответов, {workers} потоков")

    def one(job, g):
        row = {"id": g["id"], **xstest.judge_one(client, jc, prompts[g["id"]], g["response"])}
        with locks[job]:
            path = judge_path(run, job)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(one, j, g) for j, g in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            failed += fut.result()["cls"] is None
            if k % 200 == 0 or k == len(futures):
                log(f"  {k}/{len(futures)}, без оценки {failed}")
    return len(todo)


def collect(run, jobs):
    """{задание: {id: строка}} — последняя оценка на каждый id (оценённая важнее неоценённой) и прокси."""
    out = {}
    for job in jobs:
        judged = {}
        for r in read_jsonl(judge_path(run, job)):
            if strongreject.api_error(r):
                continue
            if r.get("cls") is not None or r["id"] not in judged:
                judged[r["id"]] = r
        gens = read_jsonl(gen_path(run, job))
        lost = [g["id"] for g in gens if g["id"] not in judged]
        if lost:
            raise RuntimeError(f"{job}: {len(lost)} ответов без оценки из-за сбоя API — запусти этап ещё раз")
        out[job] = {g["id"]: {**judged[g["id"]],
                              "proxy_refusal": float(g["proxy_score"] > 0)} for g in gens}
    return out


def values(rows, ids, key, fill):
    return [fill if rows[i].get(key) is None else float(rows[i][key]) for i in ids]


def summarize(judged, ids, n_boot, seed):
    out = {}
    for job, rows in judged.items():
        r = {"n_unjudged": sum(rows[i]["cls"] is None for i in ids),
             "proxy_refusal": float(np.mean([rows[i]["proxy_refusal"] for i in ids]))}
        for k in KEYS:
            r[k] = strongreject.bootstrap_ci(values(rows, ids, k, np.nan), n_boot, seed)
            r[k + "_upper"] = strongreject.bootstrap_ci(values(rows, ids, k, 1.0), n_boot, seed)
        out[job] = r
    return out


def default_pairs(meta):
    """(a, b) на каждом s: остаток − none, w − DIM того же имени, что в остатке, случайное − none."""
    pairs = []
    for job, m in meta.items():
        if m["method"] == "residual":
            s = m["scale"]
            pairs.append((job, "none"))
            dim = f"dim_{m['domain']}_s{s:g}"
            w = f"w:{m['variant']}_s{s:g}"
            if dim in meta and w in meta:
                pairs.append((w, dim))
        elif m["method"] == "random":
            pairs.append((job, "none"))
    return pairs


def paired(judged, pairs, ids, n_boot, seed):
    out = []
    for a, b in pairs:
        row = {"a": a, "b": b}
        for k in KEYS:
            for suffix, fill in (("", np.nan), ("_upper", 1.0)):
                va, vb = values(judged[a], ids, k, fill), values(judged[b], ids, k, fill)
                row[k + suffix] = strongreject.bootstrap_ci([x - y for x, y in zip(va, vb)], n_boot, seed)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=16, help="параллельных запросов к судье (на числа не влияет)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    jc = cfg["judge"]
    provenance.check_upstream(run, cfg, "judge_induce")

    m = run.manifest()["induce"]
    jobs = m["jobs"]
    prompts = {r["id"]: r["prompt"] for r in read_json(run.path("induce", "prompts.json"))}
    judge_missing(run, jobs, prompts, jc, args.workers)

    judged = collect(run, jobs)
    ids = sorted(judged["none"])
    summary = summarize(judged, ids, jc["n_boot"], jc["boot_seed"])
    diffs = paired(judged, default_pairs(m["meta"]), ids, jc["n_boot"], jc["boot_seed"])
    served = sorted({r.get("judge_model") for rows in judged.values() for r in rows.values() if r.get("judge_model")})
    res = {"n": len(ids), "rows": summary, "diffs": diffs, "meta": m["meta"], "alpha0": m["alpha0"],
           "point": m["point"], "judge_model": jc["model"], "served_models": served}
    write_json(res, run.path("induce", "induce.json"))
    run.write_report("induce.md", report.induce(cfg["model_alias"], res))
    run.update_manifest("judge_induce", {"model": jc["model"], "served_models": served,
                                         "missing": sum(r["n_unjudged"] for r in summary.values())})
    provenance.record(run, cfg, "judge_induce")


if __name__ == "__main__":
    main()
