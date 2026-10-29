"""Данные: источник harmful-вопросов, harmless-срез и нарезка сплитов.

Какой датасет и какие колонки — задаёт секция dataset конфига, так что новый датасет это новый
конфиг, а не новый код. Поддержаны два вида источников (dataset.kind):

    kind: labeled (по умолчанию) — у каждой строки одна метка домена, ответов нет:

        dataset:
          name: salad                       # метка source в файлах сплитов
          kind: labeled
          hf_path: OpenSafetyLab/Salad-Data
          hf_config: base_set
          hf_revision: d21a325e…            # коммит датасета на HF — обязателен
          text_col: question
          domain_col: 1-category
          stratum_col: 3-category           # null — без стратификации
        domains: {O1: "O1: Representation & Toxicity", …}

    kind: paired — к запросу приложен готовый ответ без отказа, метки многозначные:

        dataset:
          name: beavertails
          kind: paired
          hf_path: PKU-Alignment/BeaverTails
          hf_revision: 8401fe60…
          path: round0/330k/train.jsonl.xz  # файл в репозитории датасета
          text_col: prompt
          response_col: response
          safe_col: is_safe                 # True — ответ безвредный, такие строки не берём
          category_col: category            # {метка: bool} в каждой строке
          min_target_words: 12
        domains: {B6: [privacy_violation], …}   # порядок доменов = приоритет назначения

В обоих случаях на выходе таблица с колонками text, domain (КОД домена из конфига), stratum и,
у kind: paired, target — ответ, который дальше становится целью обучения RDO (directions/targets.py).
"""
import json
import lzma
import os

import pandas as pd

from .artifacts import FORK, read_json

SPLITS = ("train", "val", "test")


def load_harmful(dc, domains):
    """Таблица (text, domain, stratum[, target]) по доменам конфига {код: спецификация}."""
    kind = dc.get("kind", "labeled")
    if kind == "labeled":
        return _labeled(dc, domains)
    if kind == "paired":
        return _paired(dc, domains)
    raise ValueError(f"dataset.kind: {kind!r} — бывает labeled или paired")


def _labeled(dc, domains):
    """Одна метка домена на строку. Вопрос, попавший в два домена или повторённый,
    неоднозначен — выбрасываются все его копии."""
    from datasets import load_dataset

    code = {label: d for d, label in domains.items()}
    df = load_dataset(dc["hf_path"], name=dc.get("hf_config"), split=dc.get("hf_split", "train"),
                      revision=dc["hf_revision"]).to_pandas()
    df = df[df[dc["domain_col"]].isin(code)]
    df = df[~df[dc["text_col"]].duplicated(keep=False)]
    stratum = dc.get("stratum_col")
    return pd.DataFrame({"text": df[dc["text_col"]].values,
                         "domain": [code[x] for x in df[dc["domain_col"]]],
                         "stratum": df[stratum].values if stratum else "all"})


def download(dc):
    """Файл датасета из HF-кэша на зафиксированной ревизии."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(dc["hf_path"], dc["path"], repo_type="dataset", revision=dc["hf_revision"])


def read_rows(path):
    opener = lzma.open if path.endswith(".xz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f]


def _paired(dc, domains):
    """Запрос с готовым ответом без отказа; метки многозначные.

    Домен запроса — первый домен конфига, чья метка есть среди метк его вредных ответов,
    поэтому порядок доменов в конфиге это приоритет: редкая категория должна стоять раньше
    частой, иначе частая заберёт себе почти всё. Домены получаются непересекающимися по
    построению, но не «чистыми»: у запроса из B5 может быть и метка насилия.

    Страта — первая метка домена, нашедшаяся у запроса. Цель — один ответ на запрос: первый в
    порядке датасета длиной не меньше min_target_words слов, иначе самый длинный.
    """
    df = pd.DataFrame(read_rows(download(dc)))
    df = df[~df[dc["safe_col"]].astype(bool)].reset_index(drop=True)
    text, resp = dc["text_col"], dc["response_col"]
    labels = pd.DataFrame(list(df[dc["category_col"]]), index=df.index).astype(bool)
    missing = sorted({x for labs in domains.values() for x in labs} - set(labels.columns))
    assert not missing, f"в {dc['hf_path']} нет категорий: {missing}"

    words = df[resp].str.split().str.len()
    ok = df.index[words >= dc["min_target_words"]]
    pick = words.groupby(df[text]).idxmax()                                   # запасной: самый длинный
    pick.update(df.index.to_series()[ok].groupby(df.loc[ok, text]).min())     # первый достаточно длинный

    per_prompt = labels.groupby(df[text]).any()
    rows = []
    for d, labs in domains.items():
        for p in per_prompt.index[per_prompt[labs].any(axis=1)]:
            rows.append({"text": p, "domain": d, "stratum": next(c for c in labs if per_prompt.at[p, c]),
                         "target": df.at[pick[p], resp]})
    out = pd.DataFrame(rows)
    return out[~out["text"].duplicated(keep="first")].reset_index(drop=True)  # приоритет доменов


def load_harmless(path):
    """Harmless-инструкции, [{"instruction": ...}]; относительный путь — от форка."""
    return pd.DataFrame(read_json(path if os.path.isabs(path) else os.path.join(FORK, path)))


def stratified_sample(pool, n, col, seed):
    """n строк из pool с сохранением долей col (largest remainder: сумма ровно n)."""
    counts = pool[col].value_counts()
    exact = counts / counts.sum() * n
    alloc = exact.astype(int)
    for cat in (exact - alloc).sort_values(ascending=False).index[: n - int(alloc.sum())]:
        alloc[cat] += 1
    parts = [pool[pool[col] == cat].sample(n=int(k), random_state=seed)
             for cat, k in alloc.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def cut(df, sizes):
    """Нарезает df подряд на непересекающиеся куски {split: df} размеров sizes."""
    out, i = {}, 0
    for s, n in sizes.items():
        out[s] = df.iloc[i:i + n]
        i += n
    return out
