"""Всё, что зависит от семейства модели. Единственный источник шаблона и refusal-токенов.

Новое семейство = запись в FAMILIES (ключ ищется подстрокой в имени модели). Шаблон и токены
передаются в rdo.py форка аргументами (directions/rdo.py), а DIM-пайплайн форка сверяется с ними
(directions/dim.py) — обучение RDO, DIM и наши замеры видят модель одинаково.
"""
import contextlib

import torch

FAMILIES = {
    "llama-3": {
        "template": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                     "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        "refusal_toks": [40],                 # 'I'
    },
    "gemma": {
        "template": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        "refusal_toks": [235285],             # 'I'
    },
    "qwen2.5": {
        "template": ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                     "<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"),
        "refusal_toks": [40, 2121],           # 'I', 'As'
    },
}


def family(model_name):
    name = model_name.lower()
    for k, f in FAMILIES.items():
        if k in name:
            return f
    raise ValueError(f"нет шаблона и refusal-токенов для {model_name} — добавь семейство в FAMILIES")


def prompts(fam, instructions):
    return [fam["template"].format(instruction=x) for x in instructions]


def refusal_score(logits, refusal_toks, eps=1e-8):
    """Прокси отказа по первому токену: log p(refusal) - log(1 - p(refusal)) по последней
    позиции; > 0 — модель скорее отказывает. Совпадает со scoring.refusal_metric форка."""
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    p = probs[:, refusal_toks].sum(dim=-1)
    return torch.log(p + eps) - torch.log(1 - p + eps)


def load(model_name, revision):
    """nnsight-модель в bf16 на зафиксированной ревизии HF, веса уже на GPU."""
    import os

    from nnsight import LanguageModel

    model = LanguageModel(model_name, revision=revision, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
                          device_map="auto", torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    with model.trace("Hello"):   # загрузить веса на GPU до обращения к model.device
        pass
    return model


def layers(model):
    """Блоки декодера. HF-раскладка Llama/Qwen/Gemma/Mistral: model.model.layers."""
    return model.model.layers


def ablate(model, v, a):
    """Внутри model.trace: частичная аблация направления v во всех слоях — вход блока,
    выход attn, выход mlp (как в rdo.py): h <- h - a (h·v) v."""
    from scoring import projection_einops

    for layer in layers(model):
        layer.input -= a * projection_einops(layer.input, v)
        at = layer.self_attn.output[0][:]
        layer.self_attn.output[0][:] -= a * projection_einops(at, v)
        mp = layer.mlp.output[:]
        layer.mlp.output[:] -= a * projection_einops(mp, v)


def last_token_logits(model, prompts, v, a, batch_size):
    """Логиты последней позиции [n, vocab] (float32, CPU) с аблацией v силы a во всех слоях;
    v=None — без интервенции. Это forward для ablation.evaluate."""
    if v is not None:
        v = (v / v.norm()).to(model.dtype).to(model.device)
    out = []
    for i in range(0, len(prompts), batch_size):
        with model.trace(prompts[i:i + batch_size]):
            if v is not None:
                ablate(model, v, a)
            logits = model.lm_head.output[:, -1].save()
        out.append(logits.value.detach().float().cpu())
        torch.cuda.empty_cache()
    return torch.cat(out)


def steer_logits(model, prompts, v, coeff, layer, batch_size):
    """Логиты последней позиции [n, vocab] при прибавлении coeff * v/|v| ко входу блока layer
    на всех позициях (activation addition, как induce у Arditi et al.); v=None — без интервенции."""
    if v is not None:
        v = (v / v.norm()).to(model.dtype).to(model.device)
    out = []
    for i in range(0, len(prompts), batch_size):
        with model.trace(prompts[i:i + batch_size]):
            if v is not None:
                layers(model)[layer].input += coeff * v
            logits = model.lm_head.output[:, -1].save()
        out.append(logits.value.detach().float().cpu())
        torch.cuda.empty_cache()
    return torch.cat(out)


def resid_at(model, prompts, layer, pos, batch_size):
    """Активации [n, hidden] (float32, CPU) на входе блока layer в позиции pos (от конца), без
    интервенций — то, по чему извлекается DIM."""
    out = []
    for i in range(0, len(prompts), batch_size):
        with model.trace(prompts[i:i + batch_size]):
            h = layers(model)[layer].input[:, pos].save()
        out.append(h.value.detach().float().cpu())
        torch.cuda.empty_cache()
    return torch.cat(out)


# ---------------------------------------------------------------- генерация (этап 06)
# Для генерации — обычная HF-модель с forward-хуками torch: та же аблация, что ablate()
# (вход блока, выход attn, выход mlp), но на каждом шаге декодирования, с KV-кэшем.

def load_hf(model_name, revision):
    """HF-модель в bf16 на зафиксированной ревизии и токенизатор с паддингом слева."""
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = os.getenv("HUGGINGFACE_CACHE_DIR")
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, cache_dir=cache,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    model.eval()
    return model, tok


def _per_device(v, dtype):
    """Копии v по устройствам: при device_map="auto" на нескольких картах слои стоят на разных
    устройствах, и хук берёт вектор на устройстве своей активации."""
    cache = {}

    def get(device):
        if device not in cache:
            cache[device] = v.to(device=device, dtype=dtype)
        return cache[device]
    return get


@contextlib.contextmanager
def ablation_hooks(model, v, a):
    """Пока открыт контекст, во всех слоях HF-модели h <- h - a (h·v) v на входе блока,
    выходе attn и выходе mlp. v=None или a=0 — без интервенции."""
    if v is None or a == 0:
        yield
        return
    v = _per_device(v / v.norm(), next(model.parameters()).dtype)

    def cut(t):
        u = v(t.device)
        return t - a * (t @ u)[..., None] * u

    def pre(mod, args, kwargs):
        if args:
            return (cut(args[0]),) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": cut(kwargs["hidden_states"])}

    def post(mod, args, out):
        return (cut(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else cut(out)

    handles = []
    try:
        for layer in layers(model):
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(post))
            handles.append(layer.mlp.register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def addition_hooks(model, v, coeff, layer):
    """Пока открыт контекст, h <- h + coeff * v/|v| на входе блока layer HF-модели на всех позициях
    и каждом шаге декодирования (activation addition, как steer_logits). v=None или coeff=0 — без
    интервенции. Вместе с ablation_hooks открывать после него: на входе блока сначала вырезание,
    потом прибавление — как в кросс-нейтрализации common_axis."""
    if v is None or coeff == 0:
        yield
        return
    add = _per_device(coeff * (v / v.norm()), next(model.parameters()).dtype)

    def pre(mod, args, kwargs):
        if args:
            return (args[0] + add(args[0].device),) + tuple(args[1:]), kwargs
        h = kwargs["hidden_states"]
        return args, {**kwargs, "hidden_states": h + add(h.device)}

    h = layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


class _FirstStepLogits:
    """LogitsProcessor: запоминает логиты первого шага декодирования (для прокси) и не меняет их.
    output_scores хранил бы логиты всех шагов: 512 × batch × vocab во float32 — гигабайты."""

    def __init__(self):
        self.first = None

    def __call__(self, input_ids, scores):
        if self.first is None:
            self.first = scores.float().cpu()
        return scores


@torch.no_grad()
def generate(model, tok, prompts, v, a, max_new_tokens, batch_size, refusal_toks, add=None):
    """Жадные ответы на готовые (с шаблоном) промпты при аблации v силы a и, если задано
    add = (u, coeff, layer), прибавлении coeff·û на входе блока layer (addition_hooks).
    Возвращает [(текст ответа, refusal_score первого шага)] — прокси на тех же ответах;
    при max_new_tokens = 1 это просто прокси первого токена."""
    from transformers import LogitsProcessorList

    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(next(model.parameters()).device)
        first = _FirstStepLogits()
        with ablation_hooks(model, v, a), addition_hooks(model, *(add or (None, 0, 0))):
            seqs = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                  top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
                                  logits_processor=LogitsProcessorList([first]))
        texts = tok.batch_decode(seqs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        scores = refusal_score(first.first, refusal_toks)
        out += [(t, float(s)) for t, s in zip(texts, scores)]
        torch.cuda.empty_cache()
    return out
