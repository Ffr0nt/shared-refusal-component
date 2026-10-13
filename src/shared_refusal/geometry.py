"""Математика над векторами направлений — не знает ни модели, ни данных."""
import torch


def unit(v):
    return v / v.norm()


def spectrum(V):
    """Спектр P = (1/N) sum_i v_i v_i^T для строк V [N, hidden], по убыванию.

    Считается через малую матрицу Грама G = V V^T / N (N x N): ненулевые с.з. P и G
    совпадают, собственные векторы P — W = V^T U, нормированные по столбцам.
    Возвращает (lam [N], W [hidden, N]).
    """
    N = V.shape[0]
    lam, U = torch.linalg.eigh(V @ V.T / N)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order], U[:, order]
    W = V.T @ U
    return lam, W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)


def shared_component(V, ref):
    """Общая компонента строк V: первый собственный вектор P, знак — cos(w, ref) > 0.
    Возвращает (w, lam, cos), где cos = V @ w и lam[0] = mean(cos^2)."""
    lam, W = spectrum(V)
    w = W[:, 0]
    if w @ ref < 0:
        w = -w
    cos = V @ w
    assert torch.allclose((cos ** 2).mean(), lam[0], atol=1e-8)
    return w, lam, cos


def null_lam1(N, hidden, n, seed):
    """Нулевой уровень lam_1: n раз берём N независимых случайных единичных векторов."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        R = torch.randn(N, hidden, generator=g, dtype=torch.float64)
        out.append(spectrum(R / R.norm(dim=1, keepdim=True))[0][0])
    return torch.stack(out)


def random_directions(hidden, n, seed):
    """n случайных единичных векторов (контроль в аблации)."""
    g = torch.Generator().manual_seed(seed)
    return [unit(torch.randn(hidden, generator=g)) for _ in range(n)]


def kl(base_logits, new_logits):
    """kl_div(log_softmax(base), softmax(new), 'batchmean') — та же конвенция, что в rdo.py
    (по определению torch это KL(new || base))."""
    return float(torch.nn.functional.kl_div(torch.log_softmax(base_logits.double(), -1),
                                            torch.softmax(new_logits.double(), -1), reduction="batchmean"))
