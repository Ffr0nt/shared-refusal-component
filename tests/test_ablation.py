import pytest
import torch

from conftest import make_set
from shared_refusal import ablation

TOKS = [0]          # «отказ» — токен 0


def logits_for(refuse_mask, vocab=4):
    """Логиты, где строки с refuse=True почти наверняка начинаются с токена 0."""
    out = torch.zeros(len(refuse_mask), vocab)
    out[:, 0] = torch.tensor([10.0 if r else -10.0 for r in refuse_mask])
    return out


def fake_forward(strength):
    """Модель-заглушка: промпт 'h…' — harmful, отказ снимается, если a * strength[v] >= 0.5;
    harmless-логиты сдвигаются пропорционально a (для KL)."""
    calls = []

    def forward(prompts, v, a):
        key = None if v is None else round(float(v[0]), 6)
        s = 0.0 if v is None else a * strength[key]
        calls.append((len(prompts), key, a))
        if prompts[0].startswith("h"):
            return logits_for([s < 0.5] * len(prompts))
        out = torch.zeros(len(prompts), 4)
        out[:, 1] = s
        return out

    return forward, calls


def test_asr_is_share_of_non_refusals():
    assert ablation.asr(logits_for([True, True, False, False]), TOKS) == 0.5
    assert ablation.asr(logits_for([True] * 3), TOKS) == 0.0


def test_evaluate_rows_metrics_and_calls():
    ds = make_set([("w:all", "component", None), ("rdo_A_s01", "rdo", "A")], hidden=4)
    strength = {round(float(ds.V[0][0]), 6): 1.0, round(float(ds.V[1][0]), 6): 0.0}
    forward, calls = fake_forward(strength)
    harmful = {"A": ["h1", "h2"], "B": ["h3"]}
    res = ablation.evaluate(forward, TOKS, harmful, ["x1", "x2"], ds, alphas=[0.25, 1.0], log=lambda s: None)

    assert res["base_asr"] == {"A": 0.0, "B": 0.0}
    by = {(r["direction"], r["alpha"]): r for r in res["rows"]}
    assert len(res["rows"]) == 4
    assert by[("w:all", 1.0)]["asr"] == {"A": 1.0, "B": 1.0} and by[("w:all", 1.0)]["asr_mean"] == 1.0
    assert by[("w:all", 0.25)]["asr_mean"] == 0.0
    assert by[("rdo_A_s01", 1.0)]["asr_mean"] == 0.0
    assert by[("rdo_A_s01", 1.0)]["kl"] == pytest.approx(0.0)       # не сдвигает harmless
    assert by[("w:all", 1.0)]["kl"] > by[("w:all", 0.25)]["kl"] > 0
    assert by[("rdo_A_s01", 0.25)]["method"] == "rdo" and by[("rdo_A_s01", 0.25)]["domain"] == "A"
    # база: 2 домена + harmless; на каждое направление и alpha: 2 домена + harmless
    assert len(calls) == 3 + 2 * 2 * 3


def row(name, method, domain, a, kl, asr_a, asr_b):
    return {"direction": name, "method": method, "domain": domain, "alpha": a, "kl": kl,
            "asr": {"A": asr_a, "B": asr_b}, "asr_mean": (asr_a + asr_b) / 2}


def test_summary_table_groups_rdo_by_domain_and_random():
    rows = [row("w:all", "component", None, 0.5, 0.1, 1.0, 1.0),
            row("dim_A", "dim", "A", 0.5, 0.2, 0.9, 0.8),
            row("rdo_A_s01", "rdo", "A", 0.5, 0.1, 0.6, 0.4),
            row("rdo_A_s02", "rdo", "A", 0.5, 0.3, 0.8, 0.6),
            row("rand_0", "random", None, 0.5, 0.0, 0.2, 0.2),
            row("rand_1", "random", None, 0.5, 0.2, 0.4, 0.2),
            row("w:all", "component", None, 1.0, 0.2, 1.0, 1.0)]
    t = ablation.summary_table({"rows": rows}, ["A", "B"])
    assert [r["direction"] for r in t] == ["w:all", "dim_A", "rdo_A (2 сидов)", "rand (среднее)", "w:all"]
    rdo_a = t[2]
    assert rdo_a["kl"] == pytest.approx(0.2) and rdo_a["asr"] == pytest.approx({"A": 0.7, "B": 0.5})
    assert rdo_a["asr_mean"] == pytest.approx(0.6) and rdo_a["alpha"] == 0.5
    assert t[3]["asr"] == pytest.approx({"A": 0.3, "B": 0.2})
