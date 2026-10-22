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
