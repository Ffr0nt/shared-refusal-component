"""CPU-этапы целиком (scripts/02_component.py, scripts/04_analysis.py) на синтетическом наборе
во временной папке результатов."""
import importlib.util
import os
import sys

import pytest
import torch
import yaml

from shared_refusal import artifacts
from shared_refusal.directions import assemble

SCRIPTS = os.path.join(artifacts.REPO, "scripts")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stage1(repo, tmp_path):
    """Конфиг и готовый результат этапа 1: 2 домена, DIM + 2 RDO на домен."""
    cfg = {"model": "Org/Tiny-Model", "domains": {"A": "a", "B": "b"},
           "rdo": {"seeds": {"A": [1, 2], "B": [3, 4]}},
           "component": {"n_null": 20, "null_seed": 0,
                         "variants": {"all": {}, "dim_only": {"methods": ["dim"]},
                                      "no_s04": {"exclude": ["rdo_B_s04"]}}},
           "convergence": {"selection": {}, "n_max": 5, "n_samples": 30, "n_orders": 20, "seed": 21,
                           "targets": [0.9, 0.95]}}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(32, generator=g)
    w0 /= w0.norm()
    vec = lambda: w0 + 0.7 * torch.randn(32, generator=g) / 32 ** 0.5
    dims = {"A": vec(), "B": vec()}
    rdos = {d: {s: {"vector": vec(), "init_vector": torch.randn(32, generator=g), "add_layer": 5, "lowest_loss": 1.0}
                for s in seeds} for d, seeds in cfg["rdo"]["seeds"].items()}
    ds, init = assemble(dims, rdos, {"pos": -1, "layer": 5})
    run = artifacts.Run(os.path.join(str(repo), "results", "tiny"), cfg["model"])
    ds.save(run.path("directions", "directions.pt"))
    init.save(run.path("directions", "init.pt"))
    return str(path), run, w0


def run_main(name, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [name] + argv)
    load_script(name).main()


def test_component_then_analysis(stage1, monkeypatch, capsys):
    cfg_path, run, w0 = stage1
    run_main("02_component", ["--config", cfg_path], monkeypatch)
    m = run.manifest()["component"]
    assert set(m) == {"all", "dim_only", "no_s04"}
    assert m["all"]["N"] == 6 and m["dim_only"]["N"] == 2 and m["no_s04"]["N"] == 5
    for v in m:
        assert os.path.exists(run.path("component", v, "component.pt"))
        assert os.path.exists(run.path("reports", f"component-{v}.md"))
    w = torch.load(run.path("component", "all", "component.pt"), weights_only=True)["w"].double()
    assert w @ w0.double() > 0.9

    run_main("04_analysis", ["--config", cfg_path, "--variants", "all"], monkeypatch)
    res = artifacts.read_json(run.path("analysis", "all", "analysis.json"))
    assert set(res["dim_energy"]) == {"A", "B"} and all(e > 0.5 for e in res["dim_energy"].values())
    assert len(res["init_control"]) == 4
    assert set(run.manifest()["analysis"]) == {"all"}
    assert "доля DIM вдоль w" in capsys.readouterr().out


def test_component_requires_config(monkeypatch):
    with pytest.raises(SystemExit):
        run_main("02_component", [], monkeypatch)


def test_provenance_recorded_and_blocks_mixed_config(stage1, monkeypatch):
    cfg_path, run, _ = stage1
    run_main("02_component", ["--config", cfg_path], monkeypatch)
    prov = run.manifest()["provenance"]["component"]
    assert prov["config"].endswith("tiny.yaml") and len(prov["config_hash"]) == 64
    assert "commit" in prov["repo"] and "commit" in prov["fork"]

    # этап 1 будто бы посчитан с другим rdo — этапы 2 и 4 поверх него не запускаются
    m = run.manifest()
    m["provenance"]["directions"] = {"config_hash": "other"}
    artifacts.write_json(m, run.path("manifest.json"))
    with pytest.raises(RuntimeError, match="этап directions"):
        run_main("04_analysis", ["--config", cfg_path], monkeypatch)


def test_convergence_stage(stage1, monkeypatch):
    cfg_path, run, _ = stage1
    run_main("05_convergence", ["--config", cfg_path], monkeypatch)
    res = artifacts.read_json(run.path("convergence", "convergence.json"))
    assert res["n_pool"] == 6 and res["n_dim"] == 2
    assert set(res["split_half"]) == {"1", "2", "3"} and set(res["vs_full"]) == {"1", "2", "3", "4", "5"}
    assert res["split_half"]["1"]["rho2"]["n"] == 15             # все пары при N = 1
    assert set(res["steps"]) == {"1", "2", "3", "4", "5"}
    assert res["fit"]["no_plateau"]["k"] > 0
    assert "convergence" in run.manifest()["provenance"]
    assert os.path.exists(run.path("reports", "convergence.md"))
