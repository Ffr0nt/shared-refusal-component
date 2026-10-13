"""Общие фикстуры. Тяжёлые части (HF-датасеты, nnsight-модель, scoring из форка) — моки:
тесты идут на CPU за секунды и не ходят в сеть."""
import sys
import types

import pytest
import torch

from shared_refusal import common


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Временный корень репозитория: results/ и manifest пишутся в tmp, а не в настоящий."""
    monkeypatch.setattr(common, "REPO", str(tmp_path))
    return tmp_path


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
