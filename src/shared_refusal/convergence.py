"""Сходимость общей компоненты по числу векторов (CPU, без модели).

Вопрос: сколько направлений (DIM и RDO вперемешку) нужно, чтобы компонента их пересечения
была достаточно чистой, и когда очередной вектор уже почти ничего не даёт.

Модель «общая часть плюс независимый шум»: v_i = c·w∞ + sqrt(1 − c²)·n_i. Тогда для компоненты
w_N из N векторов ρ_N² = cos²(w_N, w∞) = 1 / (1 + k/N), k = (1 − c²)/c², то есть
1/ρ² − 1 = k/N. Вариант с плато: 1/ρ² − 1 = k/N + b, b > 0 — систематическая часть, которую
добавление векторов не убирает.

Метрики:
  split_half — основная: cos(w_A, w_B) для непересекающихся A, B по N векторов. Ошибки
               независимы, поэтому |cos(w_A, w_B)| ≈ ρ_A ρ_B = ρ_N²; w∞ знать не нужно.
               N ≤ пул/2;
  steps      — сдвиг компоненты от ещё одного вектора: 1 − cos²(w_N, w_{N+1}) по префиксам
               случайных порядков;
  vs_full    — cos²(w_S, w_всех) и λ1(S) для случайных S размера N. Завышена при больших N:
               S почти совпадает с полным набором.
"""
import itertools
import math

import torch

from .geometry import spectrum


def top(V):
    """(λ1, w) пересечения строк V; для одного вектора — он сам."""
    lam, W = spectrum(V)
    return float(lam[0]), W[:, 0]


def _unique_draws(n_total, n_samples, draw):
    """Не больше n_samples различных случайных выборок (draw() -> hashable)."""
    seen = set()
    tries = 0
    while len(seen) < min(n_samples, n_total) and tries < 50 * n_samples:
        seen.add(draw())
        tries += 1
    return sorted(seen)


def combinations(n, k, n_samples, gen):
    """Все сочетания k из n, если их не больше n_samples, иначе n_samples различных случайных."""
    total = math.comb(n, k)
    if total <= n_samples:
        return list(itertools.combinations(range(n), k))
    return _unique_draws(total, n_samples, lambda: tuple(sorted(torch.randperm(n, generator=gen)[:k].tolist())))


def disjoint_pairs(n, k, n_samples, gen):
    """Неупорядоченные пары непересекающихся сочетаний (A, B) по k из n: все, если их не
    больше n_samples, иначе n_samples различных случайных."""
    assert 2 * k <= n, f"две непересекающиеся половины по {k} из {n} невозможны"
    total = math.comb(n, k) * math.comb(n - k, k) // 2
    if total <= n_samples:
        out = set()
        for a in itertools.combinations(range(n), k):
            rest = [i for i in range(n) if i not in a]
            for b in itertools.combinations(rest, k):
                out.add(tuple(sorted((a, b))))
        return sorted(out)

    def draw():
        p = torch.randperm(n, generator=gen).tolist()
        return tuple(sorted((tuple(sorted(p[:k])), tuple(sorted(p[k:2 * k])))))
    return _unique_draws(total, n_samples, draw)


def split_half(V, k, n_samples, gen):
    """|cos(w_A, w_B)| для непересекающихся A, B по k векторов — оценка ρ_k²."""
    out = []
    for a, b in disjoint_pairs(len(V), k, n_samples, gen):
        _, wa = top(V[list(a)])
        _, wb = top(V[list(b)])
        out.append({"A": list(a), "B": list(b), "r": abs(float(wa @ wb))})
    return out


def steps(V, n_orders, gen):
    """Для n_orders случайных порядков: 1 − cos²(w_N, w_{N+1}), N = 1 … len(V)−1."""
    n = len(V)
    out = []
    for _ in range(n_orders):
        p = torch.randperm(n, generator=gen)
        ws = [top(V[p[:m]])[1] for m in range(1, n + 1)]
        out.append([1 - float(ws[m] @ ws[m + 1]) ** 2 for m in range(n - 1)])
    return out


def vs_full(V, k, n_samples, gen):
    """Для сочетаний k векторов: cos²(w_S, w_всех) и λ1(S)."""
    _, w_all = top(V)
    out = []
    for s in combinations(len(V), k, n_samples, gen):
        lam, w = top(V[list(s)])
        out.append({"S": list(s), "cos2_full": float(w @ w_all) ** 2, "lam1": lam})
    return out


def fit(ns, rho2, plateau):
    """МНК для 1/ρ² − 1 = k/N (+ b при plateau). Возвращает {"k", "b"}."""
    x = torch.tensor([1 / n for n in ns], dtype=torch.float64)
    y = torch.tensor([1 / r - 1 for r in rho2], dtype=torch.float64)
    if not plateau:
        return {"k": float((x * y).sum() / (x * x).sum()), "b": 0.0}
    A = torch.stack([x, torch.ones_like(x)], 1)
    k, b = torch.linalg.lstsq(A, y[:, None]).solution[:, 0].tolist()
    return {"k": k, "b": b}


def rho2_model(n, k, b=0.0):
    return 1 / (1 + k / n + b)


def n_for(level, k, b=0.0):
    """Сколько векторов нужно для ρ² ≥ level (inf, если плато не пускает)."""
    room = 1 / level - 1 - b
    return k / room if room > 0 else math.inf


def gain(n, k, b=0.0):
    """Прирост ρ² от (n+1)-го вектора."""
    return rho2_model(n + 1, k, b) - rho2_model(n, k, b)
