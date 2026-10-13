import os

import pytest

from shared_refusal import provenance as P


def cfg(**over):
    base = {"model": "M", "model_revision": "r1", "dataset": {"hf_path": "D"}, "domains": {"A": "a"},
            "splits": {"n_train": 10}, "rdo": {"seeds": {"A": [1]}}, "component": {"variants": {"all": {}}},
            "ablation": {"alphas": [1.0]}, "run_name": "x", "config_path": "configs/x.yaml"}
    base.update(over)
    return base


def test_sections_include_upstream():
    assert P.sections("data") == {"dataset", "domains", "splits"}
    assert P.sections("component") == {"component", "model", "model_revision", "rdo", "dataset", "domains", "splits"}
    assert "component" in P.sections("analysis") and "ablation" not in P.sections("analysis")


@pytest.mark.parametrize("change, affected, unaffected", [
    ({"component": {"variants": {"all": {}, "dim_only": {"methods": ["dim"]}}}}, ["component", "analysis", "ablation"],
     ["data", "directions"]),
    ({"rdo": {"seeds": {"A": [2]}}}, ["directions", "component", "ablation", "analysis"], ["data"]),
    ({"ablation": {"alphas": [0.5]}}, ["ablation"], ["data", "directions", "component", "analysis"]),
    ({"run_name": "y", "config_path": "configs/y.yaml"}, [], list(P.OWN)),
])
def test_stage_hash_changes_only_downstream(change, affected, unaffected):
    a, b = cfg(), cfg(**change)
    for s in affected:
        assert P.stage_hash(a, s) != P.stage_hash(b, s), s
    for s in unaffected:
        assert P.stage_hash(a, s) == P.stage_hash(b, s), s


def test_check_upstream_and_record(run):
    c = cfg()
    P.check_upstream(run, c, "component")                 # ничего не записано — проверять нечего
    P.record(run, c, "directions")
    P.check_upstream(run, c, "component")
    P.check_upstream(run, cfg(component={"variants": {}}), "component")   # своя секция — можно
    with pytest.raises(RuntimeError, match="этап directions"):
        P.check_upstream(run, cfg(rdo={"seeds": {"A": [9]}}), "component")


def test_git_state_of_repo():
    st = P.git_state(P.REPO)
    assert st["commit"] and len(st["commit"]) == 40 and isinstance(st["dirty"], bool)
    assert P.git_state("/") == {"commit": None, "dirty": False}


def test_check_revision_reads_cache_ref(tmp_path, monkeypatch):
    ref = tmp_path / "models--Org--Model" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc\n")
    monkeypatch.setenv("HUGGINGFACE_CACHE_DIR", str(tmp_path))
    assert P.hub_ref("Org/Model") == "abc"
    P.check_revision("Org/Model", "abc")
    with pytest.raises(RuntimeError, match="в конфиге def"):
        P.check_revision("Org/Model", "def")
    with pytest.raises(RuntimeError, match="None"):
        P.check_revision("Org/Missing", "abc")
