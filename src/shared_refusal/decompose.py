"""Этап 8 — разложение DIM на общую компоненту w и остаток DIM⊥w: что из DIM снимает отказ,
что наводит его и что читает вредность.

Для каждого DIM (объединённый и по доменам) и каждого варианта компоненты w:
  остаток  dim⊥w = dim - (dim·ŵ)ŵ  — часть DIM вне общей компоненты;
  w⊥dim    = w - (w·d̂)d̂           — часть компоненты вне DIM (обратный контроль).
Три свойства направления (все — прокси по первому токену или по активациям):
  bypass — ASR на harmful_test при аблации во всех слоях (как этап 3), KL на harmless_test;
  induce — доля harmless_test с refusal_score > 0 при h <- h + s·α0·v̂ на входе блока слоя DIM;
  read   — AUROC проекции активации в точке DIM (pos, layer) на v̂: harmful_test против
           harmless_test, знак произволен — max(auc, 1 - auc); 95% ДИ — бутстрэп по промптам.
Модель видна через forward-функции (как в ablation.evaluate): математика здесь проверяется на CPU.
"""
import torch

from .artifacts import DirectionSet
from .geometry import random_directions, unit
from .models import refusal_score


def residual(v, ref):
    """Часть v, ортогональная ref, — единичный вектор, и доля энергии v вдоль ref (cos²)."""
    v, r = unit(v.double()), unit(ref.double())
    c = float(v @ r)
    return unit(v - c * r), c * c


def directions(dims, ws, n_rand, rand_seed):
    """dims: {имя DIM: вектор} (первым — объединённый), ws: {вариант: w}.
    Возвращает DirectionSet: DIM, их остатки по каждому w, сами w и w⊥(первый DIM), случайные;
    и energy: {вариант: {имя DIM: cos²(DIM, w)}}."""
    names, vecs, meta = [], [], []

    def add(name, v, **m):
        names.append(name)
        vecs.append(unit(v.double()))
        meta.append(m)

    first = next(iter(dims))
    for d, v in dims.items():
        add(d, v, kind="dim", dim=d)
    energy = {}
    for var, w in ws.items():
        add(f"w:{var}", w, kind="w", variant=var)
        wp, _ = residual(w, dims[first])
        add(f"w:{var}⊥{first}", wp, kind="w_perp_dim", variant=var, dim=first)
        energy[var] = {}
        for d, v in dims.items():
            r, e = residual(v, w)
            add(f"{d}⊥w:{var}", r, kind="dim_perp_w", variant=var, dim=d)
            energy[var][d] = e
    hidden = vecs[0].shape[0]
    for i, r in enumerate(random_directions(hidden, n_rand, rand_seed)):
        add(f"rand_{i}", r, kind="random")
    return DirectionSet.from_raw(names, vecs, meta), energy


def induce_rate(logits, refusal_toks):
    return float((refusal_score(logits, refusal_toks) > 0).float().mean())


def auroc(pos, neg):
    """AUROC «pos выше neg» без порога: доля пар (pos, neg) с pos > neg, равные — по 0.5."""
    d = pos.double()[:, None] - neg.double()[None, :]
    return float((d > 0).double().mean() + 0.5 * (d == 0).double().mean())


def read_auroc(H_pos, H_neg, v, n_boot, seed):
    """AUROC проекции на v̂ (знак произволен: max(auc, 1 - auc)) и 95% ДИ бутстрэпом —
    промпты harmful и harmless пересэмплируются отдельно, знак фиксируется по полной выборке."""
    v = unit(v.double())
    p, n = H_pos.double() @ v, H_neg.double() @ v
    auc = auroc(p, n)
    flip = auc < 0.5
    g = torch.Generator().manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        bp = p[torch.randint(len(p), (len(p),), generator=g)]
        bn = n[torch.randint(len(n), (len(n),), generator=g)]
        a = auroc(bp, bn)
        boots.append(1 - a if flip else a)
    boots = torch.tensor(boots)
    return {"auroc": 1 - auc if flip else auc,
            "ci": [float(torch.quantile(boots, 0.025)), float(torch.quantile(boots, 0.975))]}


def evaluate(forward_abl, forward_add, resid, refusal_toks, harmful, harmless, ds, scales, alpha0,
             n_boot, boot_seed, log=print):
    """forward_abl(prompts, v, a) и forward_add(prompts, v, coeff) — логиты последней позиции;
    resid(prompts) — активации в точке DIM. harmful: {домен: промпты}, harmless: промпты.
    Возвращает {"base": …, "rows": [{direction, kind, …, bypass, kl, induce: {s: …}, read}]}."""
    from .ablation import asr
    from .geometry import kl

    base_logits = forward_abl(harmless, None, 0)
    base_asr = {d: asr(forward_abl(p, None, 0), refusal_toks) for d, p in harmful.items()}
    base = {"asr": base_asr, "asr_mean": sum(base_asr.values()) / len(base_asr),
            "induce": induce_rate(base_logits, refusal_toks)}
    log(f"без вмешательства: ASR {base['asr_mean']:.3f}, отказ на harmless {base['induce']:.3f}")
    H_pos = torch.cat([resid(p) for p in harmful.values()])
    H_neg = resid(harmless)

    rows = []
    for name, v, m in zip(ds.names, ds.V, ds.meta):
        by_domain = {d: asr(forward_abl(p, v, 1.0), refusal_toks) for d, p in harmful.items()}
        row = {"direction": name, **m, "bypass": sum(by_domain.values()) / len(by_domain), "bypass_by_domain": by_domain,
               "kl": kl(base_logits, forward_abl(harmless, v, 1.0)),
               "induce": {str(s): induce_rate(forward_add(harmless, v, s * alpha0), refusal_toks) for s in scales},
               "read": read_auroc(H_pos, H_neg, v, n_boot, boot_seed)}
        rows.append(row)
        log(f"{name:28s} bypass={row['bypass']:.3f} KL={row['kl']:.4f} induce=" +
            " ".join(f"{s}:{x:.3f}" for s, x in row["induce"].items()) + f" read={row['read']['auroc']:.3f}")
    return {"base": base, "rows": rows, "n_harmful": len(H_pos), "n_harmless": len(H_neg)}
