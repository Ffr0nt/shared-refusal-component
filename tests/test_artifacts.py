import os

import pytest
import torch

from conftest import SPEC, make_set, rand_unit
from shared_refusal import artifacts
from shared_refusal.artifacts import DirectionSet, Run
from shared_refusal.config import file_sha256, load_config


def test_repo_points_to_repository_root():
    assert os.path.isfile(os.path.join(artifacts.REPO, "pyproject.toml"))
    assert os.path.isdir(os.path.join(artifacts.REPO, "src", "shared_refusal"))


def test_load_config_adds_alias_run_name_and_hash(tmp_path):
    p = tmp_path / "some-run_v2.yaml"
    p.write_text("model: Org/Some-Model-7B\ndomains: {A: a}\n")
    cfg = load_config(str(p))
    assert cfg["model_alias"] == "Some-Model-7B" and cfg["run_name"] == "some-run_v2"
    assert cfg["config_path"].endswith("some-run_v2.yaml")
    h = file_sha256(str(p))
    p.write_text("model: Org/Some-Model-7B\ndomains: {A: b}\n")
    assert file_sha256(str(p)) != h


def test_run_layout_under_results_of_run_name(repo, cfg):
    run = Run.from_config(cfg)
    base = os.path.join(str(repo), "results", "qwen-test")
    assert run.root == base
    assert run.splits("A") == os.path.join(base, "splits", "A")
    assert run.dim("A") == os.path.join(base, "dim", "A")
    assert run.rdo("A") == os.path.join(base, "rdo", "A")
    assert run.component("all") == os.path.join(base, "component", "all")
    assert run.analysis("all") == os.path.join(base, "analysis", "all")


def test_manifest_merges_steps(run, cfg):
    assert run.manifest() == {"model": cfg["model"]}
    run.update_manifest("dim", {"layer": 27})
    run.update_manifest("rdo", {"seeds": [1]})
    run.update_manifest("dim", {"layer": 12})
    assert run.manifest() == {"model": cfg["model"], "dim": {"layer": 12}, "rdo": {"seeds": [1]}}


def test_write_report(run):
    run.write_report("r.md", ["# t", "line"])
    with open(run.path("reports", "r.md")) as f:
        assert f.read() == "# t\nline\n"


def test_from_raw_normalises_and_keeps_order():
    raw = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]
    ds = DirectionSet.from_raw(["a", "b"], raw)
    assert ds.V.dtype == torch.float64
    assert torch.allclose(ds.V, torch.tensor([[0.6, 0.8], [0.0, 1.0]], dtype=torch.float64))
    assert ds.meta == [{}, {}]
    assert torch.equal(ds["b"], ds.V[1]) and ds.hidden == 2 and len(ds) == 2


def test_constructor_rejects_non_unit_duplicates_and_mismatch():
    with pytest.raises(AssertionError, match="единичными"):
        DirectionSet(["a"], torch.tensor([[3.0, 4.0]]))
    with pytest.raises(AssertionError, match="уникальны"):
        DirectionSet(["a", "a"], rand_unit(2, 3))
    with pytest.raises(AssertionError):
        DirectionSet(["a"], rand_unit(2, 3))


def test_select_by_method_domain_and_exclude():
    ds = make_set(SPEC)
    assert ds.select(methods=["dim"]).names == ["dim_A", "dim_B"]
    assert ds.select(domains=["B"]).names == ["dim_B", "rdo_B_s03"]
    assert ds.select(methods=["rdo"], exclude=["rdo_A_s02"]).names == ["rdo_A_s01", "rdo_B_s03"]
    sub = ds.select(domains=["A"])
    assert torch.equal(sub.V, ds.V[:3]) and sub.meta == ds.meta[:3]
    assert len(ds.select(methods=["random"])) == 0


def test_select_rejects_unknown_exclusion():
    with pytest.raises(AssertionError, match="rdo_Z"):
        make_set(SPEC).select(exclude=["rdo_Z"])


def test_where_and_concat():
    ds = make_set(SPEC)
    assert ds.where("dim") == [0, 3]
    extra = DirectionSet.from_raw(["rand_0"], [torch.ones(16)], [{"method": "random", "domain": None}])
    both = ds + extra
    assert both.names[-1] == "rand_0" and both.where("random") == [5]
    with pytest.raises(AssertionError, match="уникальны"):
        ds + ds


def test_save_load_roundtrip_is_bit_exact(tmp_path):
    ds = make_set(SPEC)
    ds.meta[1]["seed"] = 1
    p = str(tmp_path / "d" / "directions.pt")
    ds.save(p)
    back = DirectionSet.load(p)
    assert back.names == ds.names and back.meta == ds.meta
    assert torch.equal(back.V, ds.V)
