import pytest
import torch

from shared_refusal import analysis
from shared_refusal.artifacts import DirectionSet

E = torch.eye(4, dtype=torch.float64)


def basis_set():
    """Векторы в базисе e0..e3, косинусы считаются в уме."""
    s = 0.5 ** 0.5
    vecs = [E[0], s * (E[0] + E[1]), E[1], E[2], s * (E[0] + E[2])]
    names = ["dim_A", "rdo_A_s01", "rdo_A_s02", "dim_B", "rdo_B_s03"]
    meta = [{"method": "dim", "domain": "A"}, {"method": "rdo", "domain": "A", "seed": 1},
            {"method": "rdo", "domain": "A", "seed": 2}, {"method": "dim", "domain": "B"},
            {"method": "rdo", "domain": "B", "seed": 3}]
    return DirectionSet(names, torch.stack(vecs), meta)


def test_dim_energy_is_cos2_per_domain():
    assert analysis.dim_energy(E[0], basis_set()) == pytest.approx({"A": 1.0, "B": 0.0})


def test_mean_cos2_by_domain():
    assert analysis.mean_cos2_by_domain(E[0], basis_set(), "rdo") == pytest.approx({"A": 0.25, "B": 0.5})


def test_dim_cosines_matrix():
    m = analysis.dim_cosines(basis_set())
    assert m == {"A": {"A": 1.0, "B": 0.0}, "B": {"A": 0.0, "B": 1.0}}


def test_init_control_per_seed():
    ds = basis_set()
    init = DirectionSet(["rdo_A_s01", "rdo_B_s03"], torch.stack([E[1], E[3]]),
                        [{"method": "init"}, {"method": "init"}])
    rows = analysis.init_control(E[0], ds, init)
    assert [(r["name"], r["domain"], r["seed"]) for r in rows] == [("rdo_A_s01", "A", 1), ("rdo_B_s03", "B", 3)]
    assert rows[0]["cos2_w_init"] == pytest.approx(0.0)
    assert rows[0]["cos_final_init"] == pytest.approx(0.5 ** 0.5)
    assert rows[1]["cos_final_init"] == pytest.approx(0.0)
