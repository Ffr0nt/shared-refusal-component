import torch

from conftest import rand_unit
from shared_refusal import decompose, report

TOKS = [0]


def test_residual_is_orthogonal_unit_and_energy_is_cos2():
    v, w = rand_unit(2, 32, seed=1)
    r, e = decompose.residual(v, w)
    assert abs(float(r @ w)) < 1e-12 and abs(float(r.norm()) - 1) < 1e-12
    assert abs(e - float(v @ w) ** 2) < 1e-12


def test_directions_names_kinds_and_energy():
    V = rand_unit(3, 16, seed=2)
    ds, energy = decompose.directions({"dim_union": V[0], "dim_A": V[1]}, {"all": V[2]}, n_rand=2, rand_seed=0)
    assert ds.names == ["dim_union", "dim_A", "w:all", "w:all⊥dim_union", "dim_union⊥w:all", "dim_A⊥w:all",
                        "rand_0", "rand_1"]
    assert abs(float(ds["dim_union⊥w:all"] @ ds["w:all"])) < 1e-10
    assert abs(float(ds["w:all⊥dim_union"] @ ds["dim_union"])) < 1e-10
    assert abs(energy["all"]["dim_A"] - float(V[1] @ V[2]) ** 2) < 1e-12


def test_auroc_basic_cases():
    assert decompose.auroc(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0])) == 1.0
    assert decompose.auroc(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0])) == 0.0
    assert decompose.auroc(torch.tensor([1.0]), torch.tensor([1.0])) == 0.5


def test_read_auroc_sign_free_and_ci_brackets():
    g = torch.Generator().manual_seed(0)
    v = torch.zeros(4, dtype=torch.float64)
    v[0] = 1
    H_pos = torch.randn(200, 4, generator=g, dtype=torch.float64) - 2 * v      # вредные — ниже по v
    H_neg = torch.randn(200, 4, generator=g, dtype=torch.float64)
    r = decompose.read_auroc(H_pos, H_neg, v, n_boot=200, seed=0)
    assert r["auroc"] > 0.9 and r["ci"][0] <= r["auroc"] <= r["ci"][1]
    assert decompose.read_auroc(H_pos, H_neg, -v, n_boot=50, seed=0)["auroc"] == r["auroc"]


def test_evaluate_and_report_with_fake_model():
    """Модель-заглушка: отказ на harmful держится, пока аблируемое направление не совпадает с e0;
    добавление e0 наводит отказ; активации harmful сдвинуты по e1 (read видит e1, не e0)."""
    hidden = 8
    e = torch.eye(hidden, dtype=torch.float64)

    def logits(refuse):
        out = torch.zeros(len(refuse), 4)
        out[:, 0] = torch.tensor([10.0 if x else -10.0 for x in refuse])
        return out

    def forward_abl(prompts, v, a):
        hit = v is not None and abs(float(v.double() @ e[0])) > 0.5
        return logits([p.startswith("h") and not hit for p in prompts])

    def forward_add(prompts, v, c):
        return logits([abs(float(v.double() @ e[0])) * c > 1.0 for _ in prompts])

    def resid(prompts):
        H = torch.zeros(len(prompts), hidden, dtype=torch.float64)
        H[:, 2] = torch.arange(len(prompts), dtype=torch.float64) % 3       # шум
        if prompts[0].startswith("h"):
            H[:, 1] += 5
        return H

    dims = {"dim_union": (e[0] + 2 * e[1]), "dim_A": (e[0] + e[1])}
    ds, energy = decompose.directions(dims, {"all": e[0]}, n_rand=1, rand_seed=0)
    res = decompose.evaluate(forward_abl, forward_add, resid, TOKS, {"A": ["h1", "h2"], "B": ["h3"]}, ["s1", "s2"],
                             ds, [1, 4], alpha0=1.0, n_boot=20, boot_seed=0, log=lambda s: None)
    by = {r["direction"]: r for r in res["rows"]}
    assert res["base"]["asr_mean"] == 0.0 and res["base"]["induce"] == 0.0
    assert by["w:all"]["bypass"] == 1.0 and by["dim_union⊥w:all"]["bypass"] == 0.0
    assert by["dim_union⊥w:all"]["induce"]["4"] == 0.0 and by["w:all"]["induce"]["4"] == 1.0
    assert by["dim_union⊥w:all"]["read"]["auroc"] == 1.0
    res.update({"energy": energy, "alpha0": 1.0, "point": {"pos": -1, "layer": 3}, "scales": [1, 4]})
    text = "\n".join(report.decompose("Tiny", res))
    assert "dim_union⊥w:all" in text and "По доменам" in text
