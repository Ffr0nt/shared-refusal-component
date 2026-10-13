"""Всё, что зависит от семейства модели.

Новое семейство = запись в FAMILIES (ключ ищется подстрокой в имени модели). Шаблоны и
refusal-токены — те же, что в rdo.py форка: обучение RDO и наши замеры должны видеть модель
одинаково. Шаг 01 берёт модель через construct_model_base форка — там свой список семейств.
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
