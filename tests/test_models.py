import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from shared_refusal import models


@pytest.mark.parametrize("name, key", [
    ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5"),
    ("google/gemma-2-2b-it", "gemma"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama-3"),
])
def test_family_by_substring(name, key):
    assert models.family(name) is models.FAMILIES[key]


def test_family_unknown_model_raises():
    with pytest.raises(ValueError, match="FAMILIES"):
        models.family("mistralai/Mistral-7B-Instruct")


@pytest.mark.parametrize("key", list(models.FAMILIES))
def test_templates_have_single_instruction_slot(key):
    f = models.FAMILIES[key]
    assert f["template"].count("{instruction}") == 1
    assert "HELLO" in f["template"].format(instruction="HELLO")
    assert f["refusal_toks"] and all(isinstance(t, int) for t in f["refusal_toks"])


def fake_model(n_layers, batch, seq, hidden, seed=0):
    """Заглушка nnsight-модели: в каждом слое — тензоры там, где nnsight даёт прокси."""
    g = torch.Generator().manual_seed(seed)
    r = lambda: torch.randn(batch, seq, hidden, generator=g)
    layers = [SimpleNamespace(input=r(), self_attn=SimpleNamespace(output=(r(), None)),
                              mlp=SimpleNamespace(output=r())) for _ in range(n_layers)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def snapshot(model):
    return [(l.input.clone(), l.self_attn.output[0].clone(), l.mlp.output.clone()) for l in model.model.layers]


def test_layers_uses_hf_decoder_layout():
    m = fake_model(3, 1, 2, 4)
    assert models.layers(m) is m.model.layers


@pytest.mark.parametrize("a", [0.25, 0.5, 1.0])
def test_ablate_removes_fraction_a_of_projection_everywhere(fake_scoring, a):
    hidden = 6
    v = torch.nn.functional.normalize(torch.randn(hidden, generator=torch.Generator().manual_seed(9)), dim=0)
    m = fake_model(n_layers=3, batch=2, seq=4, hidden=hidden)
    before = snapshot(m)
    models.ablate(m, v, a)
    for (x0, at0, mp0), l in zip(before, m.model.layers):
        for old, new in ((x0, l.input), (at0, l.self_attn.output[0]), (mp0, l.mlp.output)):
            assert torch.allclose(new @ v, (1 - a) * (old @ v), atol=1e-5)          # вдоль v осталось 1 - a
            orth = lambda t: t - (t @ v)[..., None] * v
            assert torch.allclose(orth(new), orth(old), atol=1e-5)                 # ортогональное не тронуто


def test_ablate_with_zero_alpha_is_identity(fake_scoring):
    m = fake_model(2, 1, 3, 5)
    before = snapshot(m)
    models.ablate(m, torch.nn.functional.normalize(torch.ones(5), dim=0), 0.0)
    for (x0, at0, mp0), l in zip(before, m.model.layers):
        assert torch.equal(x0, l.input) and torch.equal(at0, l.self_attn.output[0]) and torch.equal(mp0, l.mlp.output)


def test_load_builds_bf16_model_freezes_and_warms_up(monkeypatch):
    created = {}

    class FakeLM:
        def __init__(self, name, **kw):
            created.update(name=name, **kw)
            self.traced, self.grad = [], None

        def requires_grad_(self, flag):
            self.grad = flag

        @contextlib.contextmanager
        def trace(self, prompt):
            self.traced.append(prompt)
            yield

    monkeypatch.setitem(sys.modules, "nnsight", types.SimpleNamespace(LanguageModel=FakeLM))
    monkeypatch.setenv("HUGGINGFACE_CACHE_DIR", "/cache/hub")

    m = models.load("Org/Qwen2.5-3B-Instruct")
    assert created == {"name": "Org/Qwen2.5-3B-Instruct", "cache_dir": "/cache/hub",
                       "device_map": "auto", "torch_dtype": torch.bfloat16}
    assert m.grad is False
    assert m.traced == ["Hello"]


def test_prompts_fill_template():
    fam = {"template": "<u>{instruction}</u>", "refusal_toks": [0]}
    assert models.prompts(fam, ["a", "b"]) == ["<u>a</u>", "<u>b</u>"]


def test_refusal_score_matches_fork_formula():
    logits = torch.randn(3, 10, generator=torch.Generator().manual_seed(0))
    toks = [2, 5]
    p = torch.softmax(logits.double(), -1)[:, toks].sum(-1)
    expected = torch.log(p + 1e-8) - torch.log(1 - p + 1e-8)
    assert torch.allclose(models.refusal_score(logits, toks), expected)


def test_refusal_score_sign():
    logits = torch.tensor([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    s = models.refusal_score(logits, [0])
    assert s[0] > 0 > s[1]
