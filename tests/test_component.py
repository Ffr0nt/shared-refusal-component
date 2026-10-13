import pytest
import torch

from conftest import SPEC, make_set
from shared_refusal import component
from shared_refusal.artifacts import DirectionSet
from shared_refusal.geometry import unit


def planted_set(hidden=32, seed=0):
    """DIM и RDO, у всех общая составляющая w0; DIM смотрят в сторону +w0."""
    g = torch.Generator().manual_seed(seed)
    w0 = unit(torch.randn(hidden, generator=g, dtype=torch.float64))
    vecs = [0.8 * w0 + 0.6 * unit(torch.randn(hidden, generator=g, dtype=torch.float64)) for _ in SPEC]
    vecs[1] = -vecs[1]                                     # знак RDO произволен
    return DirectionSet.from_raw([n for n, _, _ in SPEC], vecs, [{"method": m, "domain": d} for _, m, d in SPEC]), w0


def test_select_applies_spec():
    ds = make_set(SPEC)
    assert component.select(ds, {}).names == ds.names
    assert component.select(ds, {"methods": ["rdo"], "exclude": ["rdo_A_s02"]}).names == ["rdo_A_s01", "rdo_B_s03"]


def test_select_rejects_unknown_keys_and_tiny_sets():
    ds = make_set(SPEC)
    with pytest.raises(AssertionError, match="неизвестные"):
        component.select(ds, {"method": ["dim"]})
    with pytest.raises(AssertionError, match="меньше двух"):
        component.select(ds, {"domains": ["B"], "methods": ["dim"]})


def test_compute_recovers_planted_direction_with_dim_sign():
    ds, w0 = planted_set()
    res = component.compute(ds, n_null=50, null_seed=0)
    assert res["w"] @ w0 > 0.95                            # знак — по среднему DIM
    assert res["lam"][0] == pytest.approx((res["cos"] ** 2).mean().item())
    assert res["lam"][0] > res["null"]["p95"]
    assert res["n_by_method"] == {"dim": 2, "rdo": 3}
    assert res["names"] == ds.names
    assert res["cos"][1] < 0                               # перевёрнутый RDO


def test_compute_without_dim_uses_mean_of_all_for_sign():
    ds, w0 = planted_set()
    sub = ds.select(methods=["rdo"], exclude=["rdo_A_s01"])  # оставшиеся смотрят в +w0
    res = component.compute(sub, n_null=10, null_seed=0)
    assert res["w"] @ w0 > 0


def test_null_level_independent_of_vectors():
    a = component.compute(make_set(SPEC, seed=0), 30, 0)["null"]
    b = component.compute(make_set(SPEC, seed=1), 30, 0)["null"]
    assert a == b                                          # зависит только от N, hidden, n, seed


def test_save_and_load(tmp_path):
    ds, _ = planted_set()
    res = component.compute(ds, n_null=10, null_seed=0)
    out = str(tmp_path / "component" / "all")
    component.save(res, out, {"methods": ["dim", "rdo"]})
    assert torch.allclose(component.load_w(out), res["w"], atol=1e-6)    # сохраняется во float32
    s = component.load_summary(out)
    assert s["selection"] == {"methods": ["dim", "rdo"]} and s["N"] == 5 and s["hidden"] == 32
    assert s["lam"][0] == pytest.approx(float(res["lam"][0]))
    assert list(s["cos"]) == ds.names
