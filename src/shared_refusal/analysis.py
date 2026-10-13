"""Этап 4 — свойства общей компоненты (CPU, без модели).

Все функции принимают w и наборы направлений и возвращают числа; что из этого попадает в
отчёт — решает скрипт.
"""


def dim_energy(w, ds):
    """Доля энергии DIM каждого домена вдоль w: для единичных векторов это cos^2(w, DIM_d);
    остаток 1 - cos^2 лежит в ортогональном дополнении к w."""
    return {ds.meta[i]["domain"]: float((w @ ds.V[i]) ** 2) for i in ds.where("dim")}


def mean_cos2_by_domain(w, ds, method):
    """Средний cos^2(w, v) по направлениям метода в каждом домене."""
    out = {}
    for i in ds.where(method):
        out.setdefault(ds.meta[i]["domain"], []).append(float((w @ ds.V[i]) ** 2))
    return {d: sum(x) / len(x) for d, x in out.items()}


def dim_cosines(ds):
    """cos между DIM разных доменов."""
    idx = ds.where("dim")
    doms = [ds.meta[i]["domain"] for i in idx]
    return {a: {b: float(ds.V[i] @ ds.V[j]) for b, j in zip(doms, idx)} for a, i in zip(doms, idx)}


def init_control(w, ds, init):
    """Не объясняется ли совпадение RDO общим стартом: для каждого сида cos^2(w, старт) и
    cos(итог, старт). init — DirectionSet стартовых векторов под теми же именами, что в ds."""
    rows = []
    for n in init.names:
        i = ds.names.index(n)
        v, v0 = ds.V[i], init[n]
        rows.append({"name": n, "domain": ds.meta[i].get("domain"), "seed": ds.meta[i].get("seed"),
                     "cos2_w_init": float((w @ v0) ** 2), "cos_final_init": float(v @ v0)})
    return rows
