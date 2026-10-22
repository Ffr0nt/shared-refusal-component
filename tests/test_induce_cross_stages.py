"""Этапы 09_induce и 10_cross целиком: модель — заглушка generate, направления — из фикстуры этапа 06."""
import json
import os

import pytest
import torch
import yaml

from shared_refusal import artifacts, models
from test_scripts_cpu import run_main
from test_strongreject_stages import setup  # noqa: F401 — фикстура

RES = {"dims": ["union"], "residuals": [{"dim": "union", "component": "all"}]}


@pytest.fixture
def staged(setup, monkeypatch):  # noqa: F811
    path, run, _ = setup
    cfg = yaml.safe_load(open(path))
    cfg["induce"] = {"components": ["all"], "base": False, **RES, "n_rand": 1, "rand_seed": 0, "scales": [1, 2],
                     "max_new_tokens": 4, "batch_size": 2}
    cfg["cross"] = {"component": "all", "dim": "union", "n_rand": 1, "rand_seed": 0, "batch_size": 2,
                    "n_boot": 50, "boot_seed": 0}
    open(path, "w").write(yaml.safe_dump(cfg))
    dim_union = 4 * torch.randn(8, generator=torch.Generator().manual_seed(9))
    os.makedirs(run.dim("union"), exist_ok=True)
    torch.save(dim_union, os.path.join(run.dim("union"), "direction.pt"))
    run.update_manifest("dim", {"pos": -1, "layer": 1})
    artifacts.write_json([{"instruction": f"harmless {i}"} for i in range(3)],
                         os.path.join(run.splits("A"), "harmless_test.json"))
    calls = []

    def fake_generate(model, tok, prompts, v, a, max_new_tokens, batch_size, refusal_toks, add=None):
        calls.append({"abl": v, "add": add, "max_new_tokens": max_new_tokens})
        refuse = add is not None and v is None
        return [("I cannot." if refuse else "Sure.", 1.0 if refuse else -1.0) for _ in prompts]

    monkeypatch.setattr(models, "generate", fake_generate)
    return path, run, calls, float(dim_union.norm())


def test_induce_writes_answers_per_direction_and_scale(staged, monkeypatch):
    path, run, calls, alpha0 = staged
    run_main("09_induce", ["--config", path], monkeypatch)
    names = sorted(os.listdir(run.path("induce", "generations")))
    assert names == sorted(["none.jsonl"] + [f"{n}_s{s}.jsonl" for n in ("w_all", "dim_union", "dim_union⊥w_all", "rand_0")
                                             for s in (1, 2)])
    adds = [c["add"] for c in calls]
    assert adds[0] is None and all(c["abl"] is None for c in calls)
    assert {round(a[1] / alpha0, 6) for a in adds[1:]} == {1.0, 2.0} and {a[2] for a in adds[1:]} == {1}
    rows = [json.loads(l) for l in open(run.path("induce", "generations", "dim_union_s1.jsonl"))]
    assert [r["id"] for r in rows] == [0, 1, 2] and rows[0]["response"] == "I cannot."
    calls.clear()
    run_main("09_induce", ["--config", path], monkeypatch)       # всё готово — модель не нужна
    assert calls == []


def test_cross_uses_working_scale_from_stage8(staged, monkeypatch):
    path, run, calls, alpha0 = staged
    rows = [{"direction": "dim_union", "induce": {"0.5": 0.2, "1": 0.9, "2": 0.9, "4": 0.1}},
            {"direction": "w:all", "induce": {"0.5": 0.3, "1": 0.5, "2": 0.95, "4": 0.0}}]
    artifacts.write_json({"rows": rows, "alpha0": alpha0}, run.path("decompose", "decompose.json"))
    run_main("10_cross", ["--config", path], monkeypatch)
    res = artifacts.read_json(run.path("cross", "cross.json"))
    assert all(c["max_new_tokens"] == 1 for c in calls)
    by = {(r["induce"], r["ablate"]): r for r in res["rows"]}
    assert by[("dim_union", "—")]["scale"] == 1.0                # при равенстве — меньший s
    assert by[("w:all", "—")]["scale"] == 2.0
    assert set(a for i, a in by if i == "dim_union") == {"—", "dim_union", "w:all", "dim_union⊥w:all", "rand_0"}
    assert by[("dim_union", "—")]["mean"] == 1.0 and by[("dim_union", "w:all")]["mean"] == 0.0
    assert res["base"]["mean"] == 0.0
    assert os.path.exists(run.path("reports", "cross.md"))


class XJudge:
    """Судья XSTest-заглушка: «I cannot» — полный отказ, «Sure» — выполнение; ответ на «harmless 2» не разбирается."""

    def __init__(self):
        from types import SimpleNamespace as NS
        self.n, self._ns = 0, NS
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, temperature, max_tokens):
        self.n += 1
        text = messages[1]["content"]
        cls = "???" if 'QUESTION: "harmless 2"' in text else (
            "2_full_refusal" if "I cannot" in text else "1_full_compliance")
        return self._ns(model="gpt-4o-mini-x", choices=[self._ns(message=self._ns(content=cls))])


def test_judge_induce_and_summary(staged, monkeypatch):
    from shared_refusal import strongreject, xstest
    path, run, calls, _ = staged
    run_main("09_induce", ["--config", path], monkeypatch)
    judge = XJudge()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: judge)
    monkeypatch.setattr(xstest.time, "sleep", lambda s: None)
    run_main("11_judge_induce", ["--config", path, "--workers", "2"], monkeypatch)
    res = artifacts.read_json(run.path("induce", "induce.json"))
    rows = res["rows"]
    assert rows["none"]["full_refusal"]["mean"] == 0.0
    assert rows["dim_union_s1"]["full_refusal"]["mean"] == 1.0 and rows["dim_union_s1"]["n_unjudged"] == 1
    assert rows["dim_union_s1"]["full_refusal"]["n"] == 2                      # неоценённый — вне нижней границы
    assert rows["none"]["any_refusal_upper"]["mean"] == pytest.approx(1 / 3)   # верхняя — считая его отказом
    pairs = {(d["a"], d["b"]) for d in res["diffs"]}
    assert ("dim_union⊥w:all_s1", "none") in pairs and ("w:all_s2", "dim_union_s2") in pairs
    assert ("rand_0_s1", "none") in pairs
    assert "XSTest" in open(run.path("reports", "induce.md")).read()
    n = judge.n
    run_main("11_judge_induce", ["--config", path], monkeypatch)             # неоценённые досылаются снова
    assert judge.n == n + 2 * len(rows)                                       # retries = 2 на каждый

    out = run.path("summary.md")
    run_main("summary_judge", ["--configs", path, "--out", out], monkeypatch)
    text = open(out).read()
    assert "## 2. Вызов отказа" in text and "| 1 | 0.333 |" not in text   # none: любой отказ 0 по оценённым


def test_xstest_parse():
    from shared_refusal import xstest
    assert xstest.parse("2_full_refusal") == "full_refusal"
    assert xstest.parse("CLASS: 3_partial_refusal.") == "partial_refusal"
    assert xstest.parse("full_compliance") == "full_compliance"
    assert xstest.parse("1") == "full_compliance"
    assert xstest.parse("I'm sorry") is None and xstest.parse(None) is None
    assert '"{question}"' in xstest.PROMPT and xstest.PROMPT.endswith("CLASS:")
    # gpt-4o-mini объясняет перед меткой (22.10): вывод важнее упоминаний других классов
    assert xstest.parse("The response provided is classified as **1_full_compliance**. Not 2_full_refusal.") == "full_compliance"
    assert xstest.parse("The response is a clear refusal to answer... Therefore: 2_full_refusal.") == "full_refusal"
    assert xstest.parse("The response provided is a clear refusal to answer the") is None
    assert xstest.MAX_TOKENS >= 128


def test_paired_diffs_and_kl_fallback(run):
    from shared_refusal import strongreject
    judged = {"a": [{"id": 0, "score": 1.0, "refusal": 0.0}, {"id": 1, "score": None, "refusal": None}],
              "b": [{"id": 0, "score": 0.25, "refusal": 1.0}, {"id": 1, "score": 0.0, "refusal": 1.0}]}
    d = strongreject.paired_diffs(judged, [("a", "b")], [0, 1], 50, 0)[0]
    assert d["score"]["mean"] == 0.75 and d["score"]["n"] == 1
    assert d["score_upper"]["mean"] == pytest.approx((0.75 + 1.0) / 2)
    assert d["non_refusal"]["mean"] == 1.0

    import importlib.util
    spec = importlib.util.spec_from_file_location("j07", os.path.join(os.path.dirname(__file__), "..", "scripts", "07_judge.py"))
    j07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(j07)
    artifacts.write_json({"rows": [{"direction": "w:all", "alpha": 1.0, "kl": 0.05}]}, run.path("ablation", "ablation.json"))
    artifacts.write_json({"rows": [{"direction": "w:all", "kl": 0.9}, {"direction": "dim_union⊥w:all", "kl": 0.07}]},
                         run.path("decompose", "decompose.json"))
    assert j07.kl_at(run, 1.0) == {"w:all": 0.05, "dim_union⊥w:all": 0.07}     # этап 3 важнее этапа 8
    assert j07.kl_at(run, 0.5) == {}
    meta = {"dim_union": {"method": "dim"}, "dim_union⊥w:all": {"method": "residual"}, "rand_0": {"method": "random"}}
    assert [g[0] for g in j07.groups_of(list(meta), meta, [])] == ["none", "dim_union", "dim_union⊥w:all", "rand (среднее)"]
