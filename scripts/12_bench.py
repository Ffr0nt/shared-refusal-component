"""Этап 12 — полезность при аблации: бенчмарки lm-evaluation-harness через хуки (GPU).

Для каждой задачи секции bench.tasks (имя задачи lm-eval -> num_fewshot, метрика по вопросу и
фильтр ответа) и каждого направления (набор — как в generate: компоненты, при base — DIM и RDO
этапа 1, dims, residuals, случайные) и без вмешательства ("none") — simple_evaluate на модели с
аблацией силы alpha во всех слоях (models.ablation_hooks). Только аблация: в режиме добавления
модель начнёт отказывать, и бенчмарк измерит сам отказ, а не ущерб.

С bench.rdo_split каждый RDO отвечает только на свою 1/20 вопросов каждой подзадачи (samples в
simple_evaluate): вместе сиды покрывают набор один раз — оценка среднего RDO за цену одного
направления. Ловушки кэшей lm-eval — в shared_refusal/bench.py.

Пишет results/<запуск>/bench/<задача>/<направление>.json (метрики lm-eval, версии, хэш конфигов
задачи), bench/<задача>/samples/<направление>.jsonl (по вопросу: подзадача, doc_id, значение,
хэши, ответ), сводку bench/<задача>/summary.json и reports/bench-<задача>.md: среднее с 95% ДИ и
парные разницы бутстрепом по вопросам. Готовые файлы пропускаются.

    --tasks mmlu     только эти задачи из bench.tasks (на числа не влияет)
    --prefetch-only  скачать датасеты задач в HF-кэш и выйти (CPU, нужна сеть)
"""
import argparse
import hashlib
import json
import os

from shared_refusal import ablation, bench, models, provenance
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config

NONE = "none"
KEYS = ("alpha", "apply_chat_template", "fewshot_as_multiturn", "batch_size", "limit", "rdo_split")


def out_path(run, task, name, samples=False):
    fname = name.replace(":", "_") + (".jsonl" if samples else ".json")
    return run.path("bench", task, *(["samples"] if samples else []), fname)


def task_hash(cfg, task):
    """Хэш того, что влияет на числа задачи: её параметры, общие параметры прогона и версия
    lm-eval. batch_size включён: в bf16 разбиение на батчи сдвигает числа. Набор направлений не
    включён — к готовым ответам можно добавить направления."""
    bc = cfg["bench"]
    payload = {"task": task, "params": bc["tasks"][task], **{k: bc.get(k) for k in KEYS},
               "lm_eval": bench.lm_eval_version()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_own(run, cfg, task):
    prev = run.manifest().get("bench", {}).get(task, {}).get("hash")
    if prev and prev != task_hash(cfg, task) and os.path.isdir(run.path("bench", task)):
        raise RuntimeError(f"bench/{task} посчитан с другими параметрами задачи или версией lm-eval; "
                           "удали эту папку или заведи новый запуск")


def done(run, cfg, task, name, samples):
    p = out_path(run, task, name)
    if not (os.path.exists(p) and os.path.exists(out_path(run, task, name, samples=True))):
        return False
    r = read_json(p)
    return r["hash"] == task_hash(cfg, task) and r["samples"] == samples


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate(simple_evaluate, lm, model, cfg, task, v, samples):
    bc, p = cfg["bench"], cfg["bench"]["tasks"][task]
    with models.ablation_hooks(model, v, bc["alpha"]):
        return simple_evaluate(
            model=lm, tasks=[task], num_fewshot=p["num_fewshot"],
            apply_chat_template=bc["apply_chat_template"], fewshot_as_multiturn=bc["fewshot_as_multiturn"],
            log_samples=True, use_cache=None, cache_requests=False,          # см. bench.py
            limit=None if samples else bc["limit"], samples=samples, bootstrap_iters=0,
            random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)


def save(run, cfg, task, name, res, samples):
    p = cfg["bench"]["tasks"][task]
    rows = bench.per_question(res["samples"], p["metric"], p["filter"])
    path = out_path(run, task, name, samples=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    write_json({"direction": name, "task": task, "hash": task_hash(cfg, task), "lm_eval": bench.lm_eval_version(),
                "samples": samples, "results": res["results"], "groups": res.get("groups"),
                "n-samples": res.get("n-samples"), "n-shot": res.get("n-shot"), "versions": res.get("versions"),
                "task_config_hash": bench.config_hash(res.get("configs"))}, out_path(run, task, name))
    return sum(r["value"] for r in rows) / len(rows), len(rows)


def report(run, cfg, task, names, rdo, split, rand):
    """Сводка по готовым файлам задачи; RDO с долями вопросов — одним направлением rdo:split,
    при нескольких случайных — ещё rand:mean (среднее по ним на каждом вопросе)."""
    bc = cfg["bench"]
    values = {}
    for n in [NONE] + names:
        p = out_path(run, task, n, samples=True)
        if os.path.exists(p) and os.path.exists(out_path(run, task, n)):
            values[n] = {(r["task"], r["doc_id"]): r["value"] for r in read_jsonl(p)}
    if split and all(n in values for n in rdo):
        values[bench.SPLIT] = bench.merge_split(values, rdo)
    if split:
        values = {n: v for n, v in values.items() if n not in rdo}
    if len(rand) > 1 and all(n in values for n in rand):
        values[bench.RAND_MEAN] = bench.mean_over([values[n] for n in rand])
    if NONE not in values:
        return
    res = bench.summary(values, [tuple(x) for x in bc["pairs"]], bc["n_boot"], bc["boot_seed"], base=NONE)
    write_json(res, run.path("bench", task, "summary.json"))
    fmt = lambda c: f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c and c["mean"] is not None else "—"
    lines = [f"# {task} при аблации (α = {bc['alpha']}) — {cfg['model']}", "",
             f"Метрика по вопросу: {bc['tasks'][task]['metric']} (фильтр {bc['tasks'][task]['filter']}), "
             f"{bc['tasks'][task]['num_fewshot']}-shot, chat template: {bc['apply_chat_template']}. "
             "Среднее по всем вопросам, 95% ДИ бутстрепом по вопросам; Δ — парная разница на общих вопросах.", "",
             "| направление | n | среднее | Δ к none |", "|---|---|---|---|"]
    lines += [f"| {r['direction']} | {r['n']} | {fmt(r)} | {fmt(r['vs_base'])} |" for r in res["rows"]]
    if res["pairs"]:
        lines += ["", "| a − b | n | Δ |", "|---|---|---|"]
        lines += [f"| {r['a']} − {r['b']} | {r['n']} | {fmt(r)} |" for r in res["pairs"]]
    run.write_report(f"bench-{task}.md", lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="только эти задачи из bench.tasks (на результат не влияет — для разбиения работы)")
    ap.add_argument("--prefetch-only", action="store_true",
                    help="скачать датасеты задач в HF-кэш и выйти (без модели)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    bc = cfg["bench"]
    tasks = args.tasks or list(bc["tasks"])
    unknown = set(tasks) - set(bc["tasks"])
    assert not unknown, f"задач {sorted(unknown)} нет в bench.tasks"

    sizes = {t: bench.leaf_sizes(t) for t in tasks}
    for t, s in sizes.items():
        print(f"{t}: {len(s)} подзадач, {sum(s.values())} вопросов")
    if args.prefetch_only:
        return
    provenance.check_upstream(run, cfg, "bench")
    for t in tasks:
        check_own(run, cfg, t)

    ds = ablation.directions_to_test(bc, run)
    rdo = [n for n, mm in zip(ds.names, ds.meta) if mm["method"] == "rdo"]
    use_split = bool(bc.get("rdo_split")) and bool(rdo)
    splits = {t: bench.split_samples(bench.pool(sizes[t], bc["limit"]), rdo, bc["rdo_split"]["seed"])
              if use_split else {} for t in tasks}
    todo = [(t, n, v) for t in tasks for n, v in [(NONE, None)] + list(zip(ds.names, ds.V))
            if not done(run, cfg, t, n, splits[t].get(n))]
    n_all = len(tasks) * (len(ds) + 1)
    print(f"задач {len(tasks)} × направлений {len(ds) + 1}, уже готово {n_all - len(todo)}"
          + (f"; RDO по 1/{len(rdo)} вопросов" if use_split else ""))

    if todo:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
        lm = HFLM(pretrained=model, tokenizer=tok, backend="causal", batch_size=bc["batch_size"])
        for t, name, v in todo:
            smp = splits[t].get(name)
            res = evaluate(simple_evaluate, lm, model, cfg, t, v, smp)
            mean, n = save(run, cfg, t, name, res, smp)
            print(f"{t:16s} {name:18s} {bc['tasks'][t]['metric']} = {mean:.4f} (n = {n})", flush=True)

    meta = {NONE: {"method": NONE, "domain": None},
            **{n: {"method": mm["method"], "domain": mm["domain"]} for n, mm in zip(ds.names, ds.meta)}}
    m = run.manifest().get("bench", {})
    for t in tasks:
        m[t] = {"hash": task_hash(cfg, t), "lm_eval": bench.lm_eval_version(), "sizes": sizes[t],
                "directions": [NONE] + ds.names, "meta": meta, "rdo_split": splits[t] or None,
                "limit": bc["limit"], "alpha": bc["alpha"]}
    run.update_manifest("bench", m)
    for t in tasks:
        report(run, cfg, t, ds.names, rdo, use_split,
               [n for n, mm in zip(ds.names, ds.meta) if mm["method"] == "random"])
    provenance.record(run, cfg, "bench")


if __name__ == "__main__":
    main()
