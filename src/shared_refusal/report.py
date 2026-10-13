"""Markdown-отчёты этапов. Функции получают уже посчитанные числа и возвращают строки."""


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def component(alias, variant, spec, summary):
    lam, null, N = summary["lam"], summary["null"], summary["N"]
    by = ", ".join(f"{n} {m.upper()}" for m, n in summary["n_by_method"].items())
    lines = [f"# Общая компонента — {alias}, вариант `{variant}`", "",
             f"Отбор: `{spec or 'все направления'}`.", "",
             f"N = {N} векторов ({by}), равные веса, hidden = {summary['hidden']}.", ""]
    top = [f"**{lam[0]:.3f}**"] + [f"{x:.3f}" for x in lam[1:3]]
    top += ["—"] * (3 - len(top))                       # у варианта из двух векторов два с.з.
    lines += table(["", "λ1", "λ2", "λ3", "нуль: среднее", "нуль: p95", "нижняя граница 1/N"],
                   [[""] + top + [f"{null['mean']:.3f}", f"{null['p95']:.3f}", f"{1 / N:.3f}"]])
    lines += ["", "cos(w, v) по векторам (знак w — по среднему DIM):", ""]
    lines += table(["вектор", "cos", "cos²"], [[n, f"{c:+.3f}", f"{c * c:.3f}"] for n, c in summary["cos"].items()])
    return lines


def analysis(alias, variant, hidden, energy, rdo_cos2, dim_cos, init_rows):
    domains = list(dim_cos)
    lines = [f"# Доля общей компоненты в DIM — {alias}, вариант `{variant}`", "",
             f"Случайный уровень cos² = 1/hidden = {1 / hidden:.5f}.", ""]
    lines += table(["домен", "доля энергии DIM вдоль w", "остаток", "средний cos²(w, RDO)"],
                   [[d, f"**{e:.3f}**", f"{1 - e:.3f}", f"{rdo_cos2[d]:.3f}" if d in rdo_cos2 else "—"]
                    for d, e in energy.items()])
    lines += ["", "cos между DIM доменов (одна точка извлечения):", ""]
    lines += table([""] + domains, [[a] + [f"{dim_cos[a][b]:.3f}" for b in domains] for a in domains])
    if init_rows:
        c2 = [r["cos2_w_init"] for r in init_rows]
        cf = [r["cos_final_init"] for r in init_rows]
        lines += ["", f"Контроль общего старта ({len(init_rows)} сидов):", "",
                  f"- cos²(w, старт): среднее {sum(c2) / len(c2):.5f}, максимум {max(c2):.5f}",
                  f"- cos(итог, старт): среднее {sum(cf) / len(cf):.3f}, от {min(cf):.3f} до {max(cf):.3f}"]
    return lines


def ablation(alias, base_asr, summary_rows, domains):
    lines = [f"# Аблация — {alias}", "",
             "ASR — доля harmful_test с refusal_score < 0; KL — на harmless_test. "
             "**Сравнивать направления только при близком KL.**", "",
             "Без аблации: " + ", ".join(f"{d} {x:.3f}" for d, x in base_asr.items()) + ".", ""]
    lines += table(["направление", "a", "KL"] + [f"ASR {d}" for d in domains] + ["ASR среднее"],
                   [[r["direction"], f"{r['alpha']:.2f}", f"{r['kl']:.4f}"] +
                    [f"{r['asr'][d]:.3f}" for d in domains] + [f"**{r['asr_mean']:.3f}**"] for r in summary_rows])
    return lines
