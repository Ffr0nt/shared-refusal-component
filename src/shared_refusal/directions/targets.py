"""Готовые цели обучения RDO — такие, в которых нет DIM.

По умолчанию rdo.py форка генерирует цели сам, и две из трёх проходят через DIM
(`rdo.py:327-357`): ответ без отказа — генерация при аблации DIM, отказ на безобидный запрос —
при её прибавлении. Поэтому сходимость RDO и DIM к одной оси не независимое свидетельство.
Если файлы целей уже лежат в --targets_dir, rdo.py берёт их как есть и ничего не генерирует
(`rdo.py:328, 352`). Здесь они и пишутся:

    ablation   ответ из датасета (dataset.kind: paired), обрезанный до n_tokens токенов;
    retain     собственный ответ модели на безобидный запрос — через DIM он не проходит и в
               базовом прогоне (`rdo.py:359`);
    addition   пустая строка: цель наведения отказа читает только слагаемое L_add, а оно
               выключено (rdo.hparams.addition_lambda = 0).

Порядок записей совпадает с harmful_train.json и harmless_train.json домена: filter_data в
rdo.py отбирает запросы и цели по одним и тем же индексам, так что пары не разъезжаются.

DIM при этом всё равно читается rdo.py при загрузке (`rdo.py:226-234`), а его слой и норма
используются только слагаемым наведения. При addition_lambda = 0 в градиент они не входят;
единственный путь DIM в результат — цели, и его здесь нет.
"""
import os

import torch

from .. import models
from ..artifacts import read_json, write_json


def paths(targets_dir):
    """Файлы целей, как их ждёт rdo.py."""
    return [os.path.join(targets_dir, f"{k}_targets.json") for k in ("harmful", "harmless")]


def done(targets_dir):
    return all(os.path.exists(p) for p in paths(targets_dir))


def pending(run, domains):
    """Домены, для которых целей ещё нет."""
    return [d for d in domains if not done(os.path.join(run.rdo(d), "targets"))]


def truncate(tok, text, n_tokens):
    """Первые n_tokens токенов — столько же новых токенов генерирует rdo.py (num_target_tokens)."""
    return tok.decode(tok.encode(text, add_special_tokens=False)[:n_tokens])


def write(cfg, run, fam, domains):
    """Цели доменов, у которых их ещё нет. Модель нужна только для retain."""
    tc = cfg["rdo"]["targets"]
    assert cfg["rdo"]["hparams"]["addition_lambda"] == 0, \
        "цели наведения отказа здесь пустые — rdo.hparams.addition_lambda должен быть 0"
    todo = pending(run, domains)
    if not todo:
        print("цели: готовы, пропуск")
        return
    model, tok = models.load_hf(cfg["model"], cfg["model_revision"])
    counts, retain = {}, None
    for d in todo:
        harmful = read_json(os.path.join(run.splits(d), "harmful_train.json"))
        harmless = read_json(os.path.join(run.splits(d), "harmless_train.json"))
        missing = [i for i, r in enumerate(harmful) if not r.get("target")]
        assert not missing, \
            f"{d}/harmful_train.json: нет поля target в {len(missing)} записях — нужен dataset.kind: paired"

        harmful_targets = [{"prompt": p, "ablation": truncate(tok, r["target"], tc["n_tokens"])}
                           for p, r in zip(models.prompts(fam, [r["instruction"] for r in harmful]), harmful)]
        instructions = [r["instruction"] for r in harmless]
        prompts = models.prompts(fam, instructions)
        if retain is None:
            print(f"retain: {len(prompts)} ответов модели на безобидные запросы", flush=True)
            out = models.generate(model, tok, prompts, None, 0, tc["n_tokens"], tc["batch_size"],
                                  fam["refusal_toks"])
            retain = (instructions, [t for t, _ in out])
        assert retain[0] == instructions, "harmless_train доменов должны совпадать — retain общий"
        harmless_targets = [{"prompt": p, "addition": "", "retain": truncate(tok, t, tc["n_tokens"])}
                            for p, t in zip(prompts, retain[1])]

        targets_dir = os.path.join(run.rdo(d), "targets")
        for path, obj in zip(paths(targets_dir), (harmful_targets, harmless_targets)):
            write_json(obj, path)
        counts[d] = {"harmful": len(harmful_targets), "harmless": len(harmless_targets)}
        print(f"цели {d}: harmful {len(harmful_targets)}, harmless {len(harmless_targets)} -> {targets_dir}")

    run.update_manifest("targets", {**run.manifest().get("targets", {}),
                                    "source": tc["source"], "n_tokens": tc["n_tokens"], **counts})
    del model
    torch.cuda.empty_cache()
