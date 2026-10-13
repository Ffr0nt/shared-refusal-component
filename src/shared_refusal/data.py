"""Данные: источник harmful-вопросов, harmless-срез и нарезка сплитов.

Harmful — любой HF-датасет, где есть колонка текста и колонка домена (и, по желанию,
колонка страты для стратифицированной выборки). Какие колонки — задаёт секция dataset
конфига, так что новый датасет — это новый конфиг, а не новый код:

    dataset:
      name: salad                       # метка source в файлах сплитов
      hf_path: OpenSafetyLab/Salad-Data
      hf_config: base_set
      text_col: question
      domain_col: 1-category
      stratum_col: 3-category           # null — без стратификации
"""
import os

import pandas as pd

from .common import FORK, read_json

SPLITS = ("train", "val", "test")


def load_harmful(dc, labels):
    """Таблица (text, domain, stratum) по доменам labels. Вопрос, попавший в два домена
    или повторённый, неоднозначен — выбрасываются все его копии."""
    from datasets import load_dataset

    df = load_dataset(dc["hf_path"], name=dc.get("hf_config"), split=dc.get("hf_split", "train")).to_pandas()
    df = df[df[dc["domain_col"]].isin(labels)]
    df = df[~df[dc["text_col"]].duplicated(keep=False)]
    stratum = dc.get("stratum_col")
    return pd.DataFrame({"text": df[dc["text_col"]].values,
                         "domain": df[dc["domain_col"]].values,
                         "stratum": df[stratum].values if stratum else "all"})


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
