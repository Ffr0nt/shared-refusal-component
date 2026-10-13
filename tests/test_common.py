import os

import pytest
import torch

from shared_refusal import common


def test_repo_points_to_repository_root():
    assert os.path.isfile(os.path.join(common.REPO, "pyproject.toml"))
    assert os.path.isdir(os.path.join(common.REPO, "src", "shared_refusal"))


def test_load_config_adds_model_alias(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: Org/Some-Model-7B\ndomains: {A: a}\n")
    cfg = common.load_config(str(p))
    assert cfg["model_alias"] == "Some-Model-7B"


def test_paths_live_under_results_of_model(repo, cfg):
    base = os.path.join(str(repo), "results", "Qwen2.5-3B-Instruct")
    assert common.results_dir(cfg, "x", "y.json") == os.path.join(base, "x", "y.json")
    assert common.splits_dir(cfg, "A") == os.path.join(base, "splits", "A")
    assert common.dim_dir(cfg, "A") == os.path.join(base, "dim", "A")
    assert common.rdo_dir(cfg, "A") == os.path.join(base, "rdo", "A")


def test_update_manifest_merges_steps(repo, cfg):
    common.update_manifest(cfg, "dim", {"layer": 27})
    common.update_manifest(cfg, "rdo", {"seeds": [1]})
    common.update_manifest(cfg, "dim", {"layer": 12})
    assert common.read_manifest(cfg) == {"model": cfg["model"], "dim": {"layer": 12}, "rdo": {"seeds": [1]}}


def test_write_report_creates_markdown(repo, cfg):
    common.write_report(cfg, "r.md", ["# t", "line"])
    with open(common.results_dir(cfg, "reports", "r.md")) as f:
        assert f.read() == "# t\nline\n"


def test_load_vectors_order_names_and_normalisation(repo, cfg):
    hidden = 8
    g = torch.Generator().manual_seed(0)
    saved = {}
    for d in cfg["domains"]:
        os.makedirs(common.dim_dir(cfg, d))
        os.makedirs(common.rdo_dir(cfg, d))
        v = 5 * torch.randn(hidden, generator=g)
        torch.save(v, os.path.join(common.dim_dir(cfg, d), "direction.pt"))
        saved[f"dim_{d}"] = v
        for s in cfg["rdo"]["seeds"][d]:
            r = 3 * torch.randn(hidden, generator=g)
            torch.save({"vector": r, "init_vector": torch.randn(hidden)},
                       os.path.join(common.rdo_dir(cfg, d), f"seed_{s:02d}.pt"))
            saved[f"rdo_{d}_s{s:02d}"] = r

    names, V = common.load_vectors(cfg)
    assert names == ["dim_A", "rdo_A_s01", "rdo_A_s02", "dim_B", "rdo_B_s03"]
    assert V.dtype == torch.float64 and V.shape == (5, hidden)
    assert torch.allclose(V.norm(dim=1), torch.ones(5, dtype=V.dtype))
    for n, row in zip(names, V):
        assert torch.allclose(row, saved[n].double() / saved[n].double().norm())


def test_load_vectors_fails_on_missing_seed(repo, cfg):
    with pytest.raises(FileNotFoundError):
        common.load_vectors(cfg)
