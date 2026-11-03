"""Этап 12 — полезность модели при аблации: бенчмарки lm-evaluation-harness через хуки.

lm-eval получает уже загруженную HF-модель (HFLM(pretrained=model)) и сам вызывает её forward
(loglikelihood: MMLU, TruthfulQA) и generate (GSM8K); models.ablation_hooks висят на модулях и
срабатывают при каждом вызове — lm-eval оценивает модель с вмешательством, не зная о нём.

Здесь — всё, кроме модели: размеры задач, раздача вопросов RDO, ответы по вопросам и сводка с
парными разницами. Ловушки lm-eval (проверено на 0.4.13):
  use_cache      — SQLite-кэш ответов по имени модели; модель у нас одна, и при втором
                   направлении он вернул бы ответы первого. Не включаем.
  cache_requests — ключ кэша построенных запросов не учитывает samples: RDO с долей вопросов
                   получил бы весь набор (или полный прогон — долю). Не включаем.
  samples        — пустой список для подзадачи lm-eval читает как «все вопросы», поэтому каждая
                   доля RDO обязана быть непустой.
"""
import hashlib
import json

from .strongreject import bootstrap_ci, rdo_split

SPLIT = "rdo:split"
RAND_MEAN = "rand:mean"


def lm_eval_version():
    from importlib.metadata import version
    return version("lm_eval")


def leaf_sizes(task):
    """{подзадача: число вопросов} задачи или группы lm-eval (mmlu — 57 подзадач). Скачивает
    датасеты в HF-кэш."""
    from lm_eval.tasks import TaskManager
    return {name: len(t.eval_docs) for name, t in sorted(TaskManager().load(task)["tasks"].items())}


def pool(sizes, limit):
    """Вопросы, на которых считается задача: в каждой подзадаче первые limit (как limit в lm-eval)
    или все."""
    return {leaf: list(range(n if limit is None else min(n, limit))) for leaf, n in sizes.items()}


def split_samples(pool, names, seed):
    """{RDO: {подзадача: индексы}} — strongreject.rdo_split внутри каждой подзадачи: вместе сиды
    покрывают каждый вопрос ровно один раз, и среднее по объединению оценивает среднее по RDO за
    цену одного направления."""
    out = {n: {} for n in names}
    for leaf, ids in pool.items():
        for n, part in rdo_split(ids, names, seed).items():
            if not part:
                raise ValueError(f"{leaf}: {len(ids)} вопросов на {len(names)} RDO — у {n} пустая доля, "
                                 "а её lm-eval прочтёт как «все вопросы»; увеличь limit")
            out[n][leaf] = part
    return out


def per_question(samples, metric, filt):
    """Строки по вопросам из res['samples'] lm-eval для одного фильтра ответа: подзадача, doc_id,
    значение метрики, хэши документа и промпта (для сверки направлений) и ответ после фильтра."""
    rows, filters = [], set()
    for leaf, exs in sorted(samples.items()):
        for ex in exs:
            filters.add(ex["filter"])
            if ex["filter"] == filt:
                rows.append({"task": leaf, "doc_id": ex["doc_id"], "value": float(ex[metric]),
                             "doc_hash": ex["doc_hash"], "prompt_hash": ex["prompt_hash"],
                             "resps": ex["filtered_resps"]})
    if not rows:
        raise ValueError(f"нет ответов с фильтром {filt!r}; есть: {sorted(filters)}")
    return rows


def config_hash(configs):
    """Хэш конфигов задач, как их видит lm-eval (промпты, метрики, фильтры): меняется между
    версиями lm-eval, поэтому пишется рядом с результатом. Функции — по имени, а не по адресу."""
    s = json.dumps(configs, sort_keys=True, ensure_ascii=False,
                   default=lambda o: getattr(o, "__qualname__", type(o).__name__))
    return hashlib.sha256(s.encode()).hexdigest()


def merge_split(values, names):
    """Ответы RDO на своих долях вопросов -> одно направление rdo:split на всём наборе."""
    merged = {}
    for n in names:
        assert not set(merged) & set(values[n]), f"{n}: вопросы RDO пересекаются"
        merged.update(values[n])
    return merged


def mean_over(dirs):
    """Среднее нескольких направлений на каждом общем вопросе (контроль rand:mean): разброс между
    случайными векторами того же порядка, что сами эффекты, одного мало."""
    keys = set.intersection(*(set(d) for d in dirs))
    return {k: sum(d[k] for d in dirs) / len(dirs) for k in keys}


def summary(values, pairs, n_boot, seed, base="none"):
    """values: {направление: {(подзадача, doc_id): значение}}. Среднее по всем вопросам (как
    группа mmlu в lm-eval — с весом по размеру подзадачи) с 95% ДИ бутстрепом по вопросам и
    парные разницы «a − b» на общих вопросах: каждое направление против base и пары из конфига."""
    def diff(a, b):
        keys = sorted(set(values[a]) & set(values[b]))
        return bootstrap_ci([values[a][k] - values[b][k] for k in keys], n_boot, seed)

    rows = [{"direction": n, **bootstrap_ci(list(v.values()), n_boot, seed),
             "vs_base": diff(n, base) if base in values and n != base else None}
            for n, v in values.items()]
    paired = [{"a": a, "b": b, **diff(a, b)} for a, b in pairs if a in values and b in values]
    return {"rows": rows, "pairs": paired}
