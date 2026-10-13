"""Этап 1 — направления отказа. Каждый метод (dim, rdo) — свой модуль; на выходе этапа —
единый DirectionSet, который дальше потребляют компонента, аблация и анализ.

Новый метод получения направлений — новый модуль здесь, который отдаёт векторы с метаданными
(method, domain, …) для assemble.
"""
from ..artifacts import DirectionSet


def assemble(dims, rdos, point):
    """Единый набор из сырых результатов, порядок: по доменам — DIM, затем RDO по сидам.

    dims:  {домен: вектор DIM}
    rdos:  {домен: {сид: словарь из seed_XX.pt (vector, init_vector, lowest_loss, …)}}
    point: {"pos": …, "layer": …} — точка извлечения DIM (она же слой добавления RDO)
    Возвращает (directions, init): init — стартовые векторы RDO под теми же именами.
    """
    names, vecs, meta = [], [], []
    init_names, init_vecs, init_meta = [], [], []
    for d, dim in dims.items():
        names.append(f"dim_{d}")
        vecs.append(dim)
        meta.append({"method": "dim", "domain": d, "pos": point["pos"], "layer": point["layer"]})
        for s, r in rdos.get(d, {}).items():
            n = f"rdo_{d}_s{s:02d}"
            names.append(n)
            vecs.append(r["vector"])
            meta.append({"method": "rdo", "domain": d, "seed": s, "layer": r["add_layer"],
                         "lowest_loss": float(r["lowest_loss"])})
            init_names.append(n)
            init_vecs.append(r["init_vector"])
            init_meta.append({"method": "init", "domain": d, "seed": s})
    hidden = vecs[0].shape[0]
    return (DirectionSet.from_raw(names, vecs, meta),
            DirectionSet.from_raw(init_names, init_vecs, init_meta, hidden=hidden))
