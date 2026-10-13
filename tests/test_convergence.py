import math

import pytest
import torch

from shared_refusal import convergence as C


def planted(n=24, hidden=1024, c2=0.5, seed=0):
    """v_i = ±(c·w + sqrt(1−c²)·n_i), n_i ⟂ w — для неё k = (1 − c²)/c²."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(hidden, generator=g, dtype=torch.float64)
    w /= w.norm()
    N = torch.randn(n, hidden, generator=g, dtype=torch.float64)
    N -= (N @ w)[:, None] * w
    N /= N.norm(dim=1, keepdim=True)
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double()[:, None]
    return sign * (c2 ** 0.5 * w + (1 - c2) ** 0.5 * N), w


def test_combinations_enumerates_small_and_samples_large():
    g = torch.Generator().manual_seed(0)
    assert C.combinations(5, 2, 100, g) == [tuple(c) for c in __import__("itertools").combinations(range(5), 2)]
    big = C.combinations(24, 5, 50, g)
    assert len(big) == 50 == len(set(big)) and all(len(c) == 5 == len(set(c)) for c in big)


def test_disjoint_pairs_are_disjoint_unique_and_complete_when_small():
    g = torch.Generator().manual_seed(0)
    small = C.disjoint_pairs(6, 1, 1000, g)
    assert len(small) == 15                                # C(6,1)·C(5,1)/2
    big = C.disjoint_pairs(24, 6, 200, g)
    assert len(big) == 200 == len(set(big))
    assert all(not set(a) & set(b) and len(a) == len(b) == 6 for a, b in big)
    with pytest.raises(AssertionError):
        C.disjoint_pairs(5, 3, 10, g)


def test_top_of_single_vector_is_itself():
    v = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    lam, w = C.top(v)
    assert lam == pytest.approx(1.0) and abs(float(w @ v[0])) == pytest.approx(1.0)


def test_split_half_recovers_model_and_fit_finds_k():
    V, _ = planted(c2=0.5)
    g = torch.Generator().manual_seed(21)
    ns = list(range(1, 13))
    rho2 = [sum(r["r"] for r in rows) / len(rows) for rows in (C.split_half(V, n, 200, g) for n in ns)]
    assert rho2[0] == pytest.approx(0.5, abs=0.03)        # один вектор: cos² с компонентой = c²
    assert all(a < b for a, b in zip(rho2, rho2[1:]))
    f = C.fit(ns, rho2, plateau=False)
    assert f["k"] == pytest.approx(1.0, abs=0.15)
    fp = C.fit(ns, rho2, plateau=True)
    assert abs(fp["b"]) < 0.05


def test_split_half_tracks_true_component():
    """ρ² из половин ≈ настоящий cos²(w_N, w∞), которого в реальных данных нет."""
    V, w = planted(c2=0.5, seed=3)
    g = torch.Generator().manual_seed(0)
    rows = C.split_half(V, 6, 300, g)
    est = sum(r["r"] for r in rows) / len(rows)
    true = sum(float(C.top(V[list(r["A"])])[1] @ w) ** 2 for r in rows) / len(rows)
    assert est == pytest.approx(true, abs=0.03)
    assert true == pytest.approx(C.rho2_model(6, 1.0), abs=0.05)


def test_steps_shrink_with_n():
    V, _ = planted()
    s = C.steps(V, 50, torch.Generator().manual_seed(0))
    assert len(s) == 50 and len(s[0]) == 23
    mean = [sum(x[i] for x in s) / len(s) for i in range(23)]
    assert mean[0] > mean[5] > mean[20] > 0


def test_vs_full_and_lam1():
    V, _ = planted()
    rows = C.vs_full(V, len(V), 10, torch.Generator().manual_seed(0))
    assert len(rows) == 1 and rows[0]["cos2_full"] == pytest.approx(1.0)
    one = C.vs_full(V, 1, 100, torch.Generator().manual_seed(0))
    assert len(one) == 24 and all(r["lam1"] == pytest.approx(1.0) for r in one)


def test_model_helpers():
    assert C.rho2_model(1, 1.0) == pytest.approx(0.5)
    assert C.n_for(0.9, 1.0) == pytest.approx(9.0)
    assert C.n_for(0.9, 1.0, b=0.2) == math.inf          # плато выше цели
    assert C.gain(10, 1.0) == pytest.approx(11 / 12 - 10 / 11)
