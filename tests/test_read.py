import torch

from conftest import rand_unit
from shared_refusal import read, report


def test_group_pairs_type_with_its_contrast():
    assert read.group("homonyms") == read.group("contrast_homonyms") == "homonyms"
    assert read.group("nons_group_real_discr") == read.group("real_group_nons_discr") == read.group("contrast_discr")
    assert read.group("privacy_public") == read.group("privacy_fictional") == read.group("contrast_privacy")


def test_directions_residual_orthogonal_to_w_and_energy():
    V = rand_unit(2, 32, seed=1)
    dirs, energy = read.directions({"dim_union": V[0]}, {"all": V[1]}, n_rand=3, rand_seed=0)
    assert list(dirs) == ["dim_union", "dim_union⊥w:all", "rand_0", "rand_1", "rand_2"]
    assert abs(float(dirs["dim_union⊥w:all"][0] @ V[1].double())) < 1e-10
    assert dirs["rand_0"][1] == "random"
    assert abs(energy["all"]["dim_union"] - float(V[0] @ V[1]) ** 2) < 1e-12


def test_evaluate_pair_separating_direction_beats_random_null():
    g = torch.Generator().manual_seed(0)
    hidden = 64
    v = torch.zeros(hidden, dtype=torch.float64)
    v[0] = 1
    H_pos = torch.randn(60, hidden, generator=g, dtype=torch.float64) + 3 * v
    H_neg = torch.randn(80, hidden, generator=g, dtype=torch.float64)
    dirs = {"dim_union": (v, "dim"), "neg": (-v, "dim_perp_w")}
    for i, r in enumerate(torch.randn(19, hidden, generator=g, dtype=torch.float64)):
        r[0] = 0
        dirs[f"rand_{i}"] = (r / r.norm(), "random")
    gp, gn = ["a"] * 30 + ["b"] * 30, ["a"] * 40 + ["b"] * 40
    res = read.evaluate_pair(H_pos, H_neg, dirs, n_boot=50, boot_seed=0, groups=(gp, gn))
    d = res["rows"]["dim_union"]
    assert d["auroc"] > 0.95 and d["signed"] == d["auroc"] and d["p_rand"] == 1 / 20
    assert res["rows"]["neg"]["auroc"] == d["auroc"] and res["rows"]["neg"]["signed"] < 0.05
    assert res["null"]["n"] == 19 and res["null"]["max"] < d["auroc"]
    assert set(res["by_group"]) == {"a", "b"} and res["by_group"]["a"]["n_pos"] == 30
    assert res["by_group"]["a"]["dim_union"] > res["by_group"]["a"]["rand"]["q95"]


def test_report_read_renders():
    V = rand_unit(2, 16, seed=3)
    dirs, energy = read.directions({"dim_union": V[0]}, {"all": V[1]}, n_rand=4, rand_seed=0)
    H = torch.randn(20, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    pair = read.evaluate_pair(H[:10], H[10:], dirs, n_boot=10, boot_seed=0, groups=(["a"] * 10, ["a"] * 10))
    res = {"xstest": pair, "salad": read.evaluate_pair(H[:10], H[10:], dirs, n_boot=10, boot_seed=0),
           "energy": energy, "point": {"pos": -1, "layer": 3}}
    text = "\n".join(report.read("M", res))
    assert "dim_union⊥w:all" in text and "rand (4)" in text and "| a |" in text
