"""TruthfulQA mc2 при аблации: куда уходит вероятность. Разбор ответов этапа 12 (bench/truthfulqa_mc2/samples).

mc2 не генерирует текст: модель оценивает loglik каждого готового варианта, метрика — доля вероятности
на верных. Здесь вероятности вариантов — softmax по loglik внутри вопроса (та же нормировка, что в mc2).
Считаем среднюю по вопросам вероятность вариантов-«отрицаний» (No…, There is no…, I have no comment, Nothing…)
и вариантов «Yes…» и печатаем вопросы, где направление сильнее всего уронило mc2, с лучшим вариантом
до и после. Отрицание определяется по первым словам варианта — эвристика.

    uv run python scripts/tqa_negation.py --run results/qwen2.5-7b_salad-o1o2o5o6 \
        --dirs w_all,dim_union,dim_union⊥w_all,rand_0 --examples 5

Нужен датасет truthfulqa/truthful_qa (multiple_choice) в HF-кэше — его скачивает этап 12.
"""
import argparse
import json
import math
import re
from pathlib import Path

from datasets import load_dataset

NEG = re.compile(r"^(No\b|I have no comment|There is no\b|There are no\b|Nothing\b|None\b|Nobody\b|No one\b)")
YES = re.compile(r"^Yes\b")


def probs(row):
    ll = [r[0] for r in row["resps"]]
    m = max(ll)
    e = [math.exp(x - m) for x in ll]
    z = sum(e)
    return [x / z for x in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="папка запуска results/<run>")
    ap.add_argument("--dirs", required=True, help="направления через запятую, как имена файлов samples/")
    ap.add_argument("--examples", type=int, default=0, help="сколько худших вопросов показать на направление")
    a = ap.parse_args()

    samples = Path(a.run) / "bench" / "truthfulqa_mc2" / "samples"
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = {d: {r["doc_id"]: r for r in map(json.loads, open(samples / f"{d}.jsonl"))}
            for d in ["none"] + a.dirs.split(",")}

    tgt = [x["mc2_targets"] for x in ds]
    has = lambda true: sum(any(NEG.match(c) for c, l in zip(t["choices"], t["labels"]) if l == true)
                           for t in tgt) / len(tgt)
    print(f"{Path(a.run).name}: вопросов с вариантом-отрицанием среди верных {has(1):.2f}, среди ложных {has(0):.2f}")
    print(f"| направление | mc2 | P(отрицание) | P(Yes) |\n|---|---|---|---|")
    for d, rs in rows.items():
        neg = yes = mc2 = 0.0
        for i, r in rs.items():
            for c, p in zip(tgt[i]["choices"], probs(r)):
                neg += p * bool(NEG.match(c))
                yes += p * bool(YES.match(c))
            mc2 += r["value"]
        n = len(rs)
        print(f"| {d} | {mc2 / n:.3f} | {neg / n:.3f} | {yes / n:.3f} |")

    for d in list(rows)[1:] if a.examples else []:
        print(f"\n## {d}: где mc2 упал сильнее всего")
        worst = sorted(rows["none"], key=lambda i: rows[d][i]["value"] - rows["none"][i]["value"])[:a.examples]
        for i in worst:
            print(f"\n[{i}] {ds[i]['question']}  mc2 {rows['none'][i]['value']:.2f} -> {rows[d][i]['value']:.2f}")
            for name in ("none", d):
                p = probs(rows[name][i])
                k = max(range(len(p)), key=p.__getitem__)
                mark = "верный" if tgt[i]["labels"][k] else "ложный"
                print(f"  {name}: {mark} {p[k]:.2f}  {tgt[i]['choices'][k]}")


if __name__ == "__main__":
    main()
