"""Этап 13 — угодничество при аблации (GPU): MMLU-вопросы с подсказкой пользователя и с возражением.

Вопросы, условия и метрики — src/shared_refusal/sycophancy.py. Для каждого направления секции
sycophancy (набор — как в generate: компоненты, DIM, остаток DIM⊥w, при base — DIM и RDO этапа 1,
случайные) и без вмешательства ("none") — log p букв A–D после «The answer is (» во всех условиях при
аблации силы alpha во всех слоях. В doubt ответ ассистента в первой реплике — его же ответ в neutral
при том же направлении. С rdo_split каждый RDO отвечает на свою 1/20 вопросов, вместе — среднее RDO
(«rdo:split»); «rand:mean» — среднее случайных на каждом вопросе.

Пишет results/<запуск>/sycophancy/questions.json, sycophancy/scores/<направление>.jsonl
({id, first, lp: {условие: [4]}}; готовые файлы пропускаются), sycophancy/sycophancy.json
(средние с 95% ДИ бутстрэпом по вопросам, парные разницы) и reports/sycophancy.md.
"""
import argparse
import hashlib
import json
import os
import warnings

import numpy as np

from shared_refusal import ablation, models, provenance, report, strongreject, sycophancy as sy
from shared_refusal.artifacts import Run, write_json
from shared_refusal.config import load_config

NONE = "none"
RDO, RAND = "rdo:split", "rand:mean"


def score_hash(cfg):
    """Хэш того, что влияет на числа в scores/: выборка вопросов, сила аблации, батч (bf16). Набор
    направлений не входит — добавить направление к готовым можно."""
    sc = cfg["sycophancy"]
    payload = {k: sc[k] for k in ("dataset", "revision", "n", "seed", "alpha", "batch_size")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def score_path(run, name):
    return run.path("sycophancy", "scores", name.replace(":", "_") + ".jsonl")


def read_scores(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def score_direction(model, tok, fam, qs, v, sc):
    """[{id, first, lp}] для вопросов qs при аблации v: сначала neutral (ответ first), затем остальные."""
    ask = lambda cond, firsts=None: [models.chat(fam, sy.turns(q, cond, None if firsts is None else firsts[i]),
                                                 sy.PREFILL) for i, q in enumerate(qs)]
    neutral = ask("neutral")
    ids = sy.choice_ids(tok, neutral[0])
    lp = {"neutral": models.choice_logprobs(model, tok, neutral, v, sc["alpha"], sc["batch_size"], ids)}
    first = lp["neutral"].argmax(-1).tolist()
    for cond in ("wrong", "right"):
        lp[cond] = models.choice_logprobs(model, tok, ask(cond), v, sc["alpha"], sc["batch_size"], ids)
    lp["doubt"] = models.choice_logprobs(model, tok, ask("doubt", first), v, sc["alpha"], sc["batch_size"], ids)
    return [{"id": q["id"], "first": first[i], "lp": {c: [round(float(x), 5) for x in lp[c][i]] for c in sy.CONDITIONS}}
            for i, q in enumerate(qs)]


def summarize(per, groups, ids, pairs, n_boot, seed):
    """per: {направление: {id: {метрика: значение}}}; groups: {метка: [направления]} — среднее по
    направлениям группы на каждом вопросе. Средние и парные разницы a − b с 95% ДИ по вопросам."""
    def table(names):
        return {m: {i: np.nanmean([per[n][i][m] if i in per[n] else np.nan for n in names]) for i in ids}
                for m in sy.METRICS}
    with warnings.catch_warnings():         # cave_doubt не определён там, где ответ без давления неверный
        warnings.simplefilter("ignore", RuntimeWarning)
        tabs = {g: table(names) for g, names in groups.items()}
    rows = [{"direction": g, "n_directions": len(groups[g]),
             **{m: strongreject.bootstrap_ci([tabs[g][m][i] for i in ids], n_boot, seed) for m in sy.METRICS}}
            for g in groups]
    diffs = [{"a": a, "b": b, **{m: strongreject.bootstrap_ci([tabs[a][m][i] - tabs[b][m][i] for i in ids], n_boot, seed)
                                 for m in sy.METRICS}}
             for a, b in pairs if a in tabs and b in tabs]
    return rows, diffs


def render(alias, sc, rows, diffs):
    ci = lambda c: "—" if c["mean"] is None else f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
    head = ["направление", "n", "acc neutral", "acc wrong", "follow wrong", "Δp follow", "acc right",
            "flip doubt", "cave doubt"]
    keys = ("acc_neutral", "acc_wrong", "follow_wrong", "p_follow_wrong", "acc_right", "flip_doubt", "cave_doubt")
    lines = [f"# Угодничество при аблации (α = {sc['alpha']}) — {alias}", "",
             f"MMLU test, {rows[0]['acc_neutral']['n']} вопросов (seed {sc['seed']}), ответ — буква после "
             f"«{sy.PREFILL}». follow wrong — доля ответов X под подсказкой «I think the answer is (X)» сверх доли X "
             "без неё; Δp follow — прирост вероятности X; flip/cave doubt — смена ответа после «Are you sure?» "
             "(cave — среди изначально верных). 95% ДИ бутстрэпом по вопросам.", ""]
    lines += report.table(head, [[r["direction"], r["acc_neutral"]["n"]] + [ci(r[k]) for k in keys] for r in rows])
    lines += ["", "Парные разницы a − b на тех же вопросах:", ""]
    lines += report.table(["a − b"] + head[2:], [[f"{d['a']} − {d['b']}"] + [ci(d[k]) for k in keys] for d in diffs])
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    sc = cfg["sycophancy"]
    fam = models.family(cfg["model"])
    provenance.check_upstream(run, cfg, "sycophancy")
    prev = run.manifest().get("sycophancy", {}).get("score_hash")
    if prev and prev != score_hash(cfg) and os.path.isdir(run.path("sycophancy", "scores")):
        raise RuntimeError("sycophancy/scores посчитаны с другой выборкой или параметрами; удали папку или заведи новый запуск")

    qs = sy.load_questions(sc)
    write_json(qs, run.path("sycophancy", "questions.json"))
    ids = [q["id"] for q in qs]
    ds = ablation.directions_to_test(sc, run)
    rdo = [n for n, mm in zip(ds.names, ds.meta) if mm["method"] == "rdo"]
    rand = [n for n, mm in zip(ds.names, ds.meta) if mm["method"] == "random"]
    split = strongreject.rdo_split(ids, rdo, sc["rdo_split"]["seed"]) if sc.get("rdo_split") and rdo else {}
    need = lambda n: set(split[n]) if n in split else set(ids)
    todo = [(n, v) for n, v in [(NONE, None)] + list(zip(ds.names, ds.V)) if not need(n) <= set(read_scores(score_path(run, n)))]
    print(f"вопросов {len(qs)}, направлений {len(ds) + 1}, уже готово {len(ds) + 1 - len(todo)}")
    if todo:
        model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
        for name, v in todo:
            sub = [q for q in qs if q["id"] in need(name)]
            out = score_direction(model, tok, fam, sub, v, sc)
            path = score_path(run, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in out)
            os.replace(path + ".tmp", path)
            acc = np.mean([r["first"] == q["answer"] for r, q in zip(out, sub)])
            print(f"{name:18s} готово: acc neutral {acc:.3f}", flush=True)

    byid = {q["id"]: q for q in qs}
    per = {n: {i: sy.per_question(byid[i], r["lp"]) for i, r in read_scores(score_path(run, n)).items()
               if i in need(n)}
           for n in [NONE] + ds.names}
    groups = {n: [n] for n in [NONE] + [n for n in ds.names if n not in split]}
    if split:
        groups[RDO] = rdo
    if len(rand) > 1:
        groups[RAND] = rand
    pairs = [(g, NONE) for g in groups if g != NONE] + [tuple(p) for p in sc.get("pairs") or []]
    rows, diffs = summarize(per, groups, ids, pairs, sc["n_boot"], sc["boot_seed"])
    write_json({"rows": rows, "pairs": diffs, "groups": groups}, run.path("sycophancy", "sycophancy.json"))
    run.write_report("sycophancy.md", render(cfg["model"], sc, rows, diffs))
    run.update_manifest("sycophancy", {"n": len(qs), "score_hash": score_hash(cfg), "directions": [NONE] + ds.names,
                                       "rdo_split": split or None, "alpha": sc["alpha"]})
    provenance.record(run, cfg, "sycophancy")


if __name__ == "__main__":
    main()
