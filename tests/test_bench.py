"""Этап 12_bench: раздача вопросов RDO, ответы по вопросам, сводка и этап целиком — lm-eval,
модель и хуки заглушены."""
import contextlib
import os
import sys
import types

import pytest
import torch
import yaml

from shared_refusal import artifacts, bench, models
from shared_refusal.directions import assemble
from test_scripts_cpu import run_main

SIZES = {"mmlu": {"mmlu_a": 6, "mmlu_b": 4}}


def test_split_samples_covers_pool_once():
    p = bench.pool({"a": 7, "b": 5}, None)
    s = bench.split_samples(p, ["r2", "r1"], seed=0)
    for leaf, ids in p.items():
        got = sorted(i for n in s for i in s[n][leaf])
        assert got == ids
    assert bench.pool({"a": 7, "b": 5}, 3) == {"a": [0, 1, 2], "b": [0, 1, 2]}


def test_split_samples_refuses_empty_share():
    """Пустая доля у lm-eval значит «все вопросы» — такую раздачу не допускаем."""
    with pytest.raises(ValueError, match="пустая доля"):
        bench.split_samples({"a": [0, 1]}, ["r1", "r2", "r3"], seed=0)


def test_per_question_takes_one_filter():
    ex = lambda i, f, v: {"doc_id": i, "filter": f, "exact_match": v, "doc_hash": "d", "prompt_hash": "p",
                          "filtered_resps": ["42"]}
    samples = {"gsm8k": [ex(0, "strict-match", 0.0), ex(0, "flexible-extract", 1.0)]}
    rows = bench.per_question(samples, "exact_match", "flexible-extract")
    assert [(r["task"], r["doc_id"], r["value"]) for r in rows] == [("gsm8k", 0, 1.0)]
    with pytest.raises(ValueError, match="strict-match"):
        bench.per_question(samples, "exact_match", "none")


def test_summary_paired():
    values = {"none": {("a", i): 1.0 for i in range(4)},
              "w:all": {("a", i): float(i % 2) for i in range(4)},
              "dim": {("a", i): 0.0 for i in range(4)}}
    res = bench.summary(values, [("w:all", "dim")], n_boot=50, seed=0)
    by = {r["direction"]: r for r in res["rows"]}
    assert by["none"]["vs_base"] is None and by["w:all"]["mean"] == 0.5
    assert by["w:all"]["vs_base"]["mean"] == -0.5 and by["dim"]["vs_base"]["mean"] == -1.0
    assert res["pairs"][0]["mean"] == 0.5 and res["pairs"][0]["n"] == 4


@pytest.fixture
def setup(repo, tmp_path, monkeypatch):
    cfg = {"model": "Org/Qwen2.5-Tiny", "model_revision": "r", "domains": {"A": "a"},
           "rdo": {"seeds": {"A": [1, 2]}},
           "component": {"n_null": 5, "null_seed": 0, "variants": {"all": {}}},
           "bench": {"components": ["all"], "base": True, "select": {}, "rdo_split": {"seed": 0},
                     "n_rand": 1, "rand_seed": 0, "alpha": 1.0,
                     "tasks": {"mmlu": {"num_fewshot": 0, "metric": "acc", "filter": "none"}},
                     "apply_chat_template": True, "fewshot_as_multiturn": True, "batch_size": 2,
                     "limit": None, "n_boot": 50, "boot_seed": 0, "pairs": [["w:all", "rdo:split"]]}}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    run = artifacts.Run(os.path.join(str(repo), "results", "tiny"), cfg["model"])
    g = torch.Generator().manual_seed(0)
    vec = lambda: torch.randn(8, generator=g)
    rdos = {"A": {s: {"vector": vec(), "init_vector": vec(), "add_layer": 1, "lowest_loss": 1.0} for s in (1, 2)}}
    ds, init = assemble({"A": vec()}, rdos, {"pos": -1, "layer": 1})
    ds.save(run.path("directions", "directions.pt"))
    init.save(run.path("directions", "init.pt"))
    run_main("02_component", ["--config", str(path)], monkeypatch)

    state = {"v": None}
    calls = []

    @contextlib.contextmanager
    def fake_hooks(model, v, a):
        state["v"] = v
        yield
        state["v"] = None

    def fake_evaluate(model, tasks, samples=None, limit=None, use_cache=None, cache_requests=False, **kw):
        """Без вмешательства — всё верно, при аблации — неверно; samples — индексы по подзадачам."""
        assert use_cache is None and cache_requests is False
        calls.append((tasks[0], samples))
        val = 1.0 if state["v"] is None else 0.0
        out = {}
        for leaf, n in SIZES[tasks[0]].items():
            ids = samples[leaf] if samples else list(range(n if limit is None else min(n, limit)))
            out[leaf] = [{"doc_id": i, "filter": "none", "acc": val, "doc_hash": f"{leaf}{i}", "prompt_hash": "p",
                          "filtered_resps": [[-1.0, False]]} for i in ids]
        return {"results": {"mmlu": {"acc,none": val}}, "samples": out, "configs": {"mmlu_a": {"f": len}},
                "versions": {"mmlu_a": 1}, "n-shot": {"mmlu_a": 0}, "n-samples": {}}

    fake = types.ModuleType("lm_eval")
    fake.simple_evaluate = fake_evaluate
    hf = types.ModuleType("lm_eval.models.huggingface")
    hf.HFLM = lambda **kw: object()
    for name, mod in [("lm_eval", fake), ("lm_eval.models", types.ModuleType("lm_eval.models")),
                      ("lm_eval.models.huggingface", hf)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(bench, "leaf_sizes", lambda task: SIZES[task])
    monkeypatch.setattr(bench, "lm_eval_version", lambda: "0.0-test")
    monkeypatch.setattr(models, "load_hf", lambda name, rev: (None, None))
    monkeypatch.setattr(models, "ablation_hooks", fake_hooks)
    return str(path), run, calls


def test_bench_stage(setup, monkeypatch):
    cfg_path, run, calls = setup
    run_main("12_bench", ["--config", cfg_path], monkeypatch)
    m = run.manifest()["bench"]["mmlu"]
    assert m["directions"] == ["none", "w:all", "dim_A", "rdo_A_s01", "rdo_A_s02", "rand_0"]
    assert len(calls) == 6
    split = m["rdo_split"]
    assert sorted(split) == ["rdo_A_s01", "rdo_A_s02"]
    assert calls[0] == ("mmlu", None)                                      # none — полный набор
    by_dir = {n: s for (_, s), n in zip(calls, m["directions"])}
    assert by_dir["rdo_A_s01"] == split["rdo_A_s01"] and by_dir["w:all"] is None

    res = artifacts.read_json(run.path("bench", "mmlu", "summary.json"))
    by = {r["direction"]: r for r in res["rows"]}
    assert set(by) == {"none", "w:all", "dim_A", "rand_0", "rdo:split"}      # RDO — одним направлением
    assert by["none"]["mean"] == 1.0 and by["none"]["n"] == 10
    assert by["rdo:split"]["n"] == 10 and by["rdo:split"]["vs_base"]["mean"] == -1.0
    assert res["pairs"][0]["a"] == "w:all" and res["pairs"][0]["mean"] == 0.0
    assert "bench" in run.manifest()["provenance"]
    assert "rdo:split" in open(run.path("reports", "bench-mmlu.md")).read()

    run_main("12_bench", ["--config", cfg_path], monkeypatch)              # всё готово — модель не нужна
    assert len(calls) == 6


def test_bench_adds_random_controls_to_existing(setup, monkeypatch):
    """n_rand 1 -> 3: rand_0 тот же вектор и не пересчитывается, в сводке появляется rand:mean."""
    cfg_path, run, calls = setup
    run_main("12_bench", ["--config", cfg_path], monkeypatch)
    cfg = yaml.safe_load(open(cfg_path))
    cfg["bench"]["n_rand"] = 3
    cfg["bench"]["pairs"].append(["w:all", "rand:mean"])
    open(cfg_path, "w").write(yaml.safe_dump(cfg, allow_unicode=True))
    run_main("12_bench", ["--config", cfg_path], monkeypatch)
    assert len(calls) == 8                                                 # только rand_1 и rand_2
    res = artifacts.read_json(run.path("bench", "mmlu", "summary.json"))
    by = {r["direction"]: r for r in res["rows"]}
    assert by["rand:mean"]["n"] == 10 and by["rand:mean"]["vs_base"]["mean"] == -1.0
    assert res["pairs"][-1]["b"] == "rand:mean"


def test_bench_refuses_stale_results(setup, monkeypatch):
    cfg_path, run, _ = setup
    run_main("12_bench", ["--config", cfg_path], monkeypatch)
    cfg = yaml.safe_load(open(cfg_path))
    cfg["bench"]["batch_size"] = 4
    open(cfg_path, "w").write(yaml.safe_dump(cfg, allow_unicode=True))
    with pytest.raises(RuntimeError, match="другими параметрами"):
        run_main("12_bench", ["--config", cfg_path], monkeypatch)


def test_bench_prefetch_only_needs_no_model(setup, monkeypatch):
    cfg_path, run, calls = setup
    monkeypatch.setattr(models, "load_hf", lambda name, rev: pytest.fail("prefetch не грузит модель"))
    run_main("12_bench", ["--config", cfg_path, "--prefetch-only"], monkeypatch)
    assert calls == [] and "bench" not in run.manifest()
