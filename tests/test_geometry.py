import pytest
import torch

from conftest import rand_unit
from shared_refusal import geometry as G


def test_unit_has_norm_one():
    v = torch.tensor([3.0, 4.0])
    assert torch.allclose(G.unit(v), torch.tensor([0.6, 0.8]))


def test_spectrum_matches_direct_eigendecomposition():
    V = rand_unit(6, 20, seed=1)
    lam, W = G.spectrum(V)
    P = V.T @ V / V.shape[0]
    lam_full = torch.linalg.eigvalsh(P).flip(0)
    assert torch.allclose(lam, lam_full[:6], atol=1e-10)
    assert torch.all(lam[:-1] >= lam[1:])
    for k in range(6):                                  # P w_k = lam_k w_k
        assert torch.allclose(P @ W[:, k], lam[k] * W[:, k], atol=1e-10)
    assert torch.allclose(W.norm(dim=0), torch.ones(6, dtype=V.dtype))


def test_shared_component_identical_vectors_give_lam1_one():
    v = rand_unit(1, 16)[0]
    V = torch.stack([v, -v, v])                         # знак вектора не важен
    w, lam, cos = G.shared_component(V, ref=v)
    assert lam[0] == pytest.approx(1.0)
    assert w @ v == pytest.approx(1.0)                  # знак — по ref
    assert torch.allclose(cos, torch.tensor([1.0, -1.0, 1.0], dtype=V.dtype))


def test_shared_component_orthogonal_vectors_give_lower_bound():
    V = torch.eye(4, 10, dtype=torch.float64)
    _, lam, _ = G.shared_component(V, ref=V[0])
    assert torch.allclose(lam, torch.full((4,), 0.25, dtype=V.dtype))


def test_shared_component_finds_planted_direction():
    d = 64
    w0 = rand_unit(1, d, seed=2)[0]
    noise = rand_unit(12, d, seed=3)
    V = torch.stack([G.unit(0.8 * w0 + 0.6 * n) for n in noise])
    w, lam, cos = G.shared_component(V, ref=w0)
    assert w @ w0 > 0.95
    assert lam[0] == pytest.approx((cos ** 2).mean().item())


def test_shared_component_sign_follows_ref():
    V = rand_unit(5, 8, seed=4)
    w, _, _ = G.shared_component(V, ref=V.mean(0))
    w_neg, _, _ = G.shared_component(V, ref=-V.mean(0))
    assert torch.allclose(w, -w_neg)


def test_null_lam1_is_deterministic_and_between_bounds():
    a = G.null_lam1(N=6, hidden=50, n=20, seed=0)
    b = G.null_lam1(N=6, hidden=50, n=20, seed=0)
    assert torch.equal(a, b)
    assert a.shape == (20,)
    assert torch.all(a >= 1 / 6 - 1e-12) and torch.all(a <= 1)
    assert not torch.equal(a, G.null_lam1(N=6, hidden=50, n=20, seed=1))


def test_random_directions_are_unit_and_reproducible():
    rs = G.random_directions(hidden=32, n=3, seed=0)
    assert len(rs) == 3
    assert all(r.norm().item() == pytest.approx(1.0, abs=1e-6) for r in rs)
    assert all(torch.equal(a, b) for a, b in zip(rs, G.random_directions(32, 3, seed=0)))


def test_random_directions_reproduce_original_ablation_controls():
    """Тот же поток генератора, что был в 05_ablate.py до выноса в geometry."""
    g = torch.Generator().manual_seed(0)
    expected = []
    for _ in range(3):
        r = torch.randn(32, generator=g)
        expected.append(r / r.norm())
    assert all(torch.equal(a, b) for a, b in zip(G.random_directions(32, 3, seed=0), expected))


def test_kl_zero_for_same_logits_and_matches_formula():
    base = torch.randn(4, 7, generator=torch.Generator().manual_seed(0))
    new = torch.randn(4, 7, generator=torch.Generator().manual_seed(1))
    assert G.kl(base, base) == pytest.approx(0.0, abs=1e-12)
    p_new = torch.softmax(new.double(), -1)
    manual = (p_new * (torch.log_softmax(new.double(), -1) - torch.log_softmax(base.double(), -1))).sum(-1).mean()
    assert G.kl(base, new) == pytest.approx(manual.item())      # KL(new || base)
    assert G.kl(base, new) > 0
