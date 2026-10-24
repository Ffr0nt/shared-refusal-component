"""Генерация с аблацией (models.ablation_hooks, models.generate) на крошечной случайной Qwen2."""
import pytest
import torch

from shared_refusal import models


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture
def v():
    return torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(3)), dim=0)


def hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True)


def test_full_ablation_zeroes_projection_on_residual(tiny, v):
    ids = torch.randint(0, 50, (2, 7), generator=torch.Generator().manual_seed(1))
    before = hidden(tiny, ids)
    with models.ablation_hooks(tiny, v, 1.0):
        after = hidden(tiny, ids)
    # hidden_states[1..L-1] — выходы блоков до финальной нормы: вдоль v ничего не осталось
    for h0, h in zip(before.hidden_states[1:-1], after.hidden_states[1:-1]):
        assert (h0 @ v).abs().max() > 1e-2
        assert (h @ v).abs().max() < 1e-5
    assert not torch.allclose(before.logits, after.logits)


def test_hooks_removed_and_noop_cases(tiny, v):
    ids = torch.randint(0, 50, (1, 5), generator=torch.Generator().manual_seed(2))
    base = hidden(tiny, ids).logits
    with models.ablation_hooks(tiny, v, 1.0):
        pass
    for vv, a in ((None, 1.0), (v, 0.0)):
        with models.ablation_hooks(tiny, vv, a):
            assert torch.equal(hidden(tiny, ids).logits, base)
    assert torch.equal(hidden(tiny, ids).logits, base)
    assert all(not l._forward_pre_hooks and not l.mlp._forward_hooks for l in tiny.model.layers)


def test_partial_ablation_keeps_fraction(tiny, v):
    """a = 0.5 на выходе attn: в одном прогоне hook до аблации видит сырой выход, hook после — с
    половиной проекции (хуки torch идут в порядке регистрации)."""
    seen = {}

    def keep(key):
        def hook(m, a, o):
            seen[key] = o[0].clone()             # ничего не возвращаем — выход не подменяется
        return hook

    attn = tiny.model.layers[1].self_attn
    h1 = attn.register_forward_hook(keep("raw"))
    ids = torch.randint(0, 50, (1, 4), generator=torch.Generator().manual_seed(4))
    with models.ablation_hooks(tiny, v, 0.5):
        h2 = attn.register_forward_hook(keep("cut"))
        with torch.no_grad():
            tiny(input_ids=ids)
        h2.remove()
    h1.remove()
    assert torch.allclose(seen["cut"] @ v, 0.5 * (seen["raw"] @ v), atol=1e-6)
    orth = lambda t: t - (t @ v)[..., None] * v
    assert torch.allclose(orth(seen["cut"]), orth(seen["raw"]), atol=1e-6)


class Tok:
    """Токенизатор-заглушка: промпт — строка из цифр через пробел, паддинг слева нулями."""
    pad_token_id = 0
    bos_token = None          # как у Qwen

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        seqs = [[int(t) for t in s.split()] for s in texts]
        n = max(map(len, seqs))
        ids = torch.tensor([[0] * (n - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs])

        class Enc(dict):
            def to(self, device):
                return self
        return Enc(input_ids=ids, attention_mask=mask)

    def batch_decode(self, seqs, skip_special_tokens):
        return [" ".join(map(str, s.tolist())) for s in seqs]


def test_generate_batches_greedy_and_proxy(tiny, v):
    prompts = ["5 6 7", "8 9", "10 11 12 13"]
    out = models.generate(tiny, Tok(), prompts, None, 1.0, max_new_tokens=4, batch_size=2, refusal_toks=[3])
    assert len(out) == 3 and all(len(t.split()) == 4 for t, _ in out)
    again = models.generate(tiny, Tok(), prompts, None, 1.0, max_new_tokens=4, batch_size=3, refusal_toks=[3])
    assert [t for t, _ in out] == [t for t, _ in again]                     # жадно и не зависит от батча
    with torch.no_grad():
        first = tiny(input_ids=torch.tensor([[5, 6, 7]])).logits[:, -1]
    assert out[0][1] == pytest.approx(float(models.refusal_score(first, [3])[0]), abs=1e-4)
    ablated = models.generate(tiny, Tok(), prompts, v, 1.0, max_new_tokens=4, batch_size=2, refusal_toks=[3])
    assert [s for _, s in ablated] != [s for _, s in out]


def test_addition_hooks_add_to_one_block_input(tiny, v):
    seen = {}
    target = tiny.model.layers[1]
    h1 = target.register_forward_pre_hook(lambda m, a, k: seen.__setitem__("in", (a[0] if a else k["hidden_states"]).clone()),
                                          with_kwargs=True)
    ids = torch.randint(0, 50, (2, 5), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        tiny(input_ids=ids)
    raw = seen["in"]
    h1.remove()
    # h2 регистрируется после нашего хука — видит вход уже с прибавкой; норма v не важна
    with models.addition_hooks(tiny, 2 * v, 3.0, 1):
        h2 = target.register_forward_pre_hook(lambda m, a, k: seen.__setitem__("added", (a[0] if a else k["hidden_states"]).clone()),
                                              with_kwargs=True)
        with torch.no_grad():
            tiny(input_ids=ids)
        h2.remove()
    assert torch.allclose(seen["added"] - raw, 3.0 * v.expand_as(raw), atol=1e-5)
    assert not target._forward_pre_hooks
    with models.addition_hooks(tiny, None, 3.0, 1), models.addition_hooks(tiny, v, 0, 1):
        assert not target._forward_pre_hooks


def test_generate_with_addition_and_first_token_only(tiny, v):
    prompts = ["5 6 7", "8 9"]
    base = models.generate(tiny, Tok(), prompts, None, 0, max_new_tokens=1, batch_size=2, refusal_toks=[3])
    assert all(len(t.split()) == 1 for t, _ in base)
    added = models.generate(tiny, Tok(), prompts, None, 0, max_new_tokens=1, batch_size=2, refusal_toks=[3],
                            add=(v, 5.0, 1))
    assert [s for _, s in added] != [s for _, s in base]
    both = models.generate(tiny, Tok(), prompts, v, 1.0, max_new_tokens=1, batch_size=2, refusal_toks=[3],
                           add=(v, 5.0, 1))
    assert [s for _, s in both] != [s for _, s in added]     # вырезание v во всех слоях гасит прибавку дальше
    assert not tiny.model.layers[1]._forward_pre_hooks


def test_needs_bos_by_family():
    class T:
        def __init__(self, bos):
            self.bos_token = bos
    gemma = models.FAMILIES["gemma"]["template"].format(instruction="x")
    llama = models.FAMILIES["llama-3"]["template"].format(instruction="x")
    qwen = models.FAMILIES["qwen2.5"]["template"].format(instruction="x")
    assert models.needs_bos(T("<bos>"), [gemma])                  # Gemma: BOS от токенизатора
    assert not models.needs_bos(T("<|begin_of_text|>"), [llama])  # Llama: BOS уже в шаблоне
    assert not models.needs_bos(T(None), [qwen])                  # Qwen: BOS нет
