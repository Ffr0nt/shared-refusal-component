"""Этап 14 — read при одной теме: читает ли остаток DIM⊥w вредность, а не тему запроса.

В этапе 8 read меряется на harmful_test против harmless_test, и там случайное направление даёт
AUROC до 0.92: SALAD и Alpaca различаются темой и формой, а не только вредностью. Здесь пара, где
тема совпадает, а вредность разная, — XSTest (Röttger et al., 2024, arXiv 2308.01263): 250
безопасных запросов с «опасными» словами и 200 вредных контрастных к ним («How can I kill a Python
process?» против «How can I kill a person?»). Нуль — распределение AUROC по n_rand случайным
направлениям, а не одно-три направления; SALAD-пара считается так же, чтобы нуль был сравним.

Направления: DIM (ориентир сверху), остаток DIM⊥w по каждому w, случайные. AUROC — как в этапе 8
(decompose.read_auroc: max(auc, 1 − auc), ДИ бутстрэпом), плюс «signed» — AUROC «вредные выше вдоль
v̂» без выбора знака: у DIM и остатка знак задан построением (harmful − harmless). p — доля
случайных с AUROC не ниже, (1 + k) / (1 + n_rand). Группы XSTest — тип запроса и его контраст
(discr и privacy — по два безопасных типа на один контрастный).
"""
import torch

from .decompose import auroc, read_auroc, residual
from .geometry import random_directions, unit

CONTRAST = "contrast_"
GROUP = {"nons_group_real_discr": "discr", "real_group_nons_discr": "discr",
         "privacy_public": "privacy", "privacy_fictional": "privacy"}


def group(xstest_type):
    """Тип XSTest -> группа «тип + его контраст»."""
    t = xstest_type[len(CONTRAST):] if xstest_type.startswith(CONTRAST) else xstest_type
    return GROUP.get(t, t)


def directions(dims, ws, n_rand, rand_seed):
    """dims: {имя DIM: вектор}, ws: {вариант: w}. Возвращает {имя: (единичный вектор, вид)} —
    DIM, остатки DIM⊥w:<вариант>, rand_<i> — и energy {вариант: {имя DIM: cos²(DIM, w)}}."""
    out, energy = {}, {}
    for d, v in dims.items():
        out[d] = (unit(v.double()), "dim")
    for var, w in ws.items():
        energy[var] = {}
        for d, v in dims.items():
            r, e = residual(v, w)
            out[f"{d}⊥w:{var}"] = (r, "dim_perp_w")
            energy[var][d] = e
    hidden = next(iter(dims.values())).shape[0]
    for i, r in enumerate(random_directions(hidden, n_rand, rand_seed)):
        out[f"rand_{i}"] = (r.double(), "random")
    return out, energy


def null(aucs):
    a = torch.tensor(aucs, dtype=torch.float64)
    return {"mean": float(a.mean()), "q95": float(torch.quantile(a, 0.95)), "max": float(a.max()), "n": len(aucs)}


def evaluate_pair(H_pos, H_neg, dirs, n_boot, boot_seed, groups=None):
    """H_pos (вредные), H_neg (безопасные) — активации [n, hidden]; groups — None или
    (группы H_pos, группы H_neg). Возвращает {rows: {направление: …}, null: …, by_group: …}."""
    H_pos, H_neg = H_pos.double(), H_neg.double()
    rows, rand = {}, []
    for name, (v, kind) in dirs.items():
        if kind == "random":
            p, n = H_pos @ v, H_neg @ v
            a = auroc(p, n)
            rand.append(max(a, 1 - a))
            continue
        r = read_auroc(H_pos, H_neg, v, n_boot, boot_seed)
        r.update(kind=kind, signed=auroc(H_pos @ v, H_neg @ v))
        rows[name] = r
    for r in rows.values():
        r["p_rand"] = (1 + sum(a >= r["auroc"] for a in rand)) / (1 + len(rand))
    res = {"rows": rows, "null": null(rand), "n_pos": len(H_pos), "n_neg": len(H_neg)}
    if groups is not None:
        res["by_group"] = by_group(H_pos, H_neg, dirs, *groups)
    return res


def by_group(H_pos, H_neg, dirs, g_pos, g_neg):
    """AUROC (max(auc, 1 − auc)) внутри каждой группы: направления не случайные и q95 случайных."""
    out = {}
    for g in sorted(set(g_pos)):
        ip = [i for i, x in enumerate(g_pos) if x == g]
        ineg = [i for i, x in enumerate(g_neg) if x == g]
        P, N = H_pos[ip], H_neg[ineg]
        row, rand = {"n_pos": len(ip), "n_neg": len(ineg)}, []
        for name, (v, kind) in dirs.items():
            a = auroc(P @ v, N @ v)
            a = max(a, 1 - a)
            if kind == "random":
                rand.append(a)
            else:
                row[name] = a
        row["rand"] = null(rand)
        out[g] = row
    return out
