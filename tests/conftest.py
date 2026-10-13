"""Общие фикстуры. Тяжёлые части (HF-датасеты, nnsight-модель, scoring из форка) — моки:
тесты идут на CPU за секунды и не ходят в сеть."""
import sys
import types

import pytest
import torch

from shared_refusal import artifacts
from shared_refusal.artifacts import DirectionSet


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Временный корень репозитория: results/ и manifest пишутся в tmp, а не в настоящий."""
    monkeypatch.setattr(artifacts, "REPO", str(tmp_path))
    return tmp_path


@pytest.fixture
def run(repo, cfg):
    return artifacts.Run.from_config(cfg)


@pytest.fixture
def cfg():
    return {"model": "Org/Qwen2.5-3B-Instruct", "model_alias": "Qwen2.5-3B-Instruct",
            "domains": {"A": "Domain A", "B": "Domain B"},
            "rdo": {"seeds": {"A": [1, 2], "B": [3]}}}


@pytest.fixture
def fake_scoring(monkeypatch):
    """Подменяет модуль scoring форка (einops, jaxtyping) простой проекцией."""
    mod = types.ModuleType("scoring")
    mod.projection_einops = lambda act, v: (act @ v)[..., None] * v
    monkeypatch.setitem(sys.modules, "scoring", mod)
    return mod


def rand_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(n, d, generator=g, dtype=torch.float64)
    return V / V.norm(dim=1, keepdim=True)


def make_set(spec, hidden=16, seed=0):
    """DirectionSet из [(имя, method, domain), …] со случайными единичными векторами."""
    V = rand_unit(len(spec), hidden, seed)
    return DirectionSet([n for n, _, _ in spec], V, [{"method": m, "domain": d} for _, m, d in spec])


SPEC = [("dim_A", "dim", "A"), ("rdo_A_s01", "rdo", "A"), ("rdo_A_s02", "rdo", "A"),
        ("dim_B", "dim", "B"), ("rdo_B_s03", "rdo", "B")]
