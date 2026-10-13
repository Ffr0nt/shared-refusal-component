"""Всё, что зависит от семейства модели. Единственный источник шаблона и refusal-токенов.

Новое семейство = запись в FAMILIES (ключ ищется подстрокой в имени модели). Шаблон и токены
передаются в rdo.py форка аргументами (directions/rdo.py), а DIM-пайплайн форка сверяется с ними
(directions/dim.py) — обучение RDO, DIM и наши замеры видят модель одинаково.
"""
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


def load(model_name):
    """nnsight-модель в bf16, веса уже на GPU."""
    import os

    from nnsight import LanguageModel

    model = LanguageModel(model_name, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
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
