"""DIM (difference-in-means, Arditi et al.) через пайплайн refusal_direction форка.

Модель грузится через construct_model_base форка; перед работой check_family сверяет её шаблон
и refusal-токены с models.FAMILIES — расхождение означает, что DIM и RDO видят модель по-разному.
"""
import os
import sys

import torch

from ..artifacts import FORK


def _fork_on_path():
    p = os.path.join(FORK, "refusal_direction")
    if p not in sys.path:
        sys.path.insert(0, p)


def model_base(model_name):
    _fork_on_path()
    from pipeline.model_utils.model_factory import construct_model_base
    return construct_model_base(model_name)


def check_family(mb, fam, probe="Probe instruction"):
    """Шаблон и refusal-токены форка совпадают с FAMILIES — иначе AssertionError."""
    assert list(mb.refusal_toks) == list(fam["refusal_toks"]), \
        f"refusal-токены форка {list(mb.refusal_toks)} != FAMILIES {fam['refusal_toks']}"
    ours = mb.tokenizer(fam["template"].format(instruction=probe), return_tensors="pt").input_ids[0]
    theirs = mb.tokenize_instructions_fn(instructions=[probe]).input_ids[0]
    assert torch.equal(ours, theirs), "шаблон чата форка не совпадает с FAMILIES"


def filter_by_refusal(mb, instructions, refuse):
    """Как в пайплайне Arditi et al.: оставить те, где модель отказывает (refuse=True,
    refusal score > 0) или не отказывает (refuse=False, < 0)."""
    _fork_on_path()
    from pipeline.submodules.select_direction import get_refusal_scores
    scores = get_refusal_scores(mb.model, instructions, mb.tokenize_instructions_fn, mb.refusal_toks)
    return [x for x, s in zip(instructions, scores.tolist()) if (s > 0) == refuse]


def mean_diffs(mb, harmful, harmless, artifact_dir):
    """[n_pos, n_layer, hidden]: среднее harmful минус среднее harmless во всех точках."""
    _fork_on_path()
    from pipeline.submodules.generate_directions import generate_directions
    return generate_directions(mb, harmful, harmless, artifact_dir=artifact_dir).cpu()


def union(mean_diffs_by_domain):
    """Кандидаты объединённой выборки — среднее mean_diffs доменов (равные веса доменов)."""
    return torch.stack(list(mean_diffs_by_domain.values())).mean(0)


def select_point(mb, harmful_val, harmless_val, candidates, artifact_dir):
    """Точка (pos, layer) по select_direction форка (ablation/actadd/KL на val)."""
    _fork_on_path()
    from pipeline.submodules.select_direction import select_direction
    pos, layer, _ = select_direction(mb, harmful_val, harmless_val, candidates.to(mb.model.device),
                                     artifact_dir=artifact_dir)
    return int(pos), int(layer)


def at_point(md, pos, layer):
    return md[pos, layer]
