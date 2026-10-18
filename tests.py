"""Проверки корректности системы сравнения.

Запуск: python tests.py
Каждая проверка печатает результат; в конце выводится сводка.
"""

from itertools import pairwise

import numpy as np
from scipy.sparse import csr_matrix

from expander_codes import (
    BSC,
    ExpanderCode,
    FixedErrors,
    evaluate,
    girth,
    random_regular_bipartite,
    sweep,
    wilson_interval,
)

PASSED = []


def check(name, condition, detail=""):
    PASSED.append(bool(condition))
    mark = "OK  " if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def test_graph_is_regular():
    """Построенный граф регулярен и не содержит кратных рёбер."""
    rng = np.random.default_rng(1)
    for n, dv, dc in [(1008, 3, 6), (1008, 4, 8), (1008, 3, 4), (504, 5, 10)]:
        H = random_regular_bipartite(n, dv, dc, rng).toarray()
        ok = (
            H.shape == (n * dv // dc, n)
            and set(H.sum(axis=0)) == {dv}
            and set(H.sum(axis=1)) == {dc}
            and H.max() == 1
        )
        check(f"граф ({n}, {dv}, {dc}) регулярен и прост", ok)


def test_girth_known_graphs():
    """Обхват совпадает с известным ответом на трёх малых графах."""
    cycle4 = csr_matrix(np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int32))
    cycle6 = csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int32))
    tree = csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int32))
    check("обхват графа с циклом длины 4", girth(cycle4) == 4, f"получено {girth(cycle4)}")
    check("обхват графа с циклом длины 6", girth(cycle6) == 6, f"получено {girth(cycle6)}")
    check("дерево: циклов нет", girth(tree) == np.inf)


def test_generator_matrix():
    """Строки G лежат в ядре H, а закодированное слово имеет нулевой синдром."""
    code = ExpanderCode.random_regular(n=120, dv=3, dc=6, seed=2)
    H = code.H.toarray().astype(np.int32)
    product = (H @ code.G.T.astype(np.int32)) % 2
    check("G H^T = 0 над GF(2)", not product.any())
    check("размерность согласована", code.G.shape == (code.k, code.n),
          f"k={code.k}, n={code.n}")

    rng = np.random.default_rng(3)
    messages = rng.integers(0, 2, size=(50, code.k), dtype=np.uint8)
    syndrome = (code.H @ code.encode(messages).T.astype(np.int32)) % 2
    check("синдром закодированных слов нулевой", not syndrome.any())


def test_single_error_is_corrected():
    """Одна ошибка исправляется правилами serial и max."""
    for mode in ("serial", "max"):
        code = ExpanderCode.random_regular(n=504, dv=4, dc=8, seed=4, flip_mode=mode)
        X = code.zero_codeword(code.n)
        Y = X.copy()
        Y[np.arange(code.n), np.arange(code.n)] ^= 1   # по одной ошибке в каждой позиции
        X_hat, info = code.decode(Y)
        check(f"правило {mode}: исправлены все одиночные ошибки",
              (X_hat == X).all() and info["converged"].all())


def test_no_noise_is_identity():
    """При нулевом шуме декодер ничего не портит."""
    code = ExpanderCode.random_regular(n=504, dv=3, dc=6, seed=5, flip_mode="max")
    rng = np.random.default_rng(6)
    X = code.encode(rng.integers(0, 2, size=(200, code.k), dtype=np.uint8))
    X_hat, info = code.decode(X)
    check("нулевой шум: слово не меняется",
          (X_hat == X).all() and info["converged"].all()
          and info["iterations"].max() == 0)


def test_zero_codeword_assumption():
    """FER на нулевом слове совпадает с FER на случайных словах.

    Это проверка допущения, на котором держится ускорение прогона.
    Интервалы Уилсона должны перекрываться.
    """
    code = ExpanderCode.random_regular(n=504, dv=3, dc=6, seed=7, flip_mode="max")
    args = {"batch_size": 500, "max_blocks": 10_000, "target_block_errors": 10 ** 9}

    zero = evaluate(code, BSC(0.03), np.random.default_rng(8),
                    use_zero_codeword=True, **args)
    rand = evaluate(code, BSC(0.03), np.random.default_rng(8),
                    use_zero_codeword=False, **args)
    overlap = zero["fer_lo"] <= rand["fer_hi"] and rand["fer_lo"] <= zero["fer_hi"]
    check("нулевое слово даёт тот же FER, что и случайные", overlap,
          f"нулевое {zero['fer']:.4f} [{zero['fer_lo']:.4f}, {zero['fer_hi']:.4f}], "
          f"случайные {rand['fer']:.4f} [{rand['fer_lo']:.4f}, {rand['fer_hi']:.4f}]")


def test_monotonicity():
    """С ростом шума FER не убывает."""
    code = ExpanderCode.random_regular(n=504, dv=3, dc=6, seed=9, flip_mode="max")
    rows = sweep([code], BSC, [0.01, 0.02, 0.03, 0.04, 0.05], seed=10,
                 batch_size=500, max_blocks=5000, target_block_errors=10 ** 9)
    fers = [row["fer"] for row in rows]
    check("FER растёт вместе с уровнем шума",
          all(a <= b for a, b in pairwise(fers)),
          ", ".join(f"{value:.4f}" for value in fers))


def test_fixed_errors_weight():
    """Канал FixedErrors вносит ровно t ошибок."""
    channel = FixedErrors(7)
    rng = np.random.default_rng(11)
    X = np.zeros((500, 200), dtype=np.uint8)
    weights = channel.transmit(X, rng).sum(axis=1)
    check("FixedErrors вносит ровно t ошибок", set(weights.tolist()) == {7})


def test_reproducibility():
    """Одно зерно даёт тот же граф и тот же результат прогона."""
    a = ExpanderCode.random_regular(n=504, dv=3, dc=6, seed=12, flip_mode="max")
    b = ExpanderCode.random_regular(n=504, dv=3, dc=6, seed=12, flip_mode="max")
    check("один seed — одна матрица H", (a.H.toarray() == b.H.toarray()).all())

    args = {"batch_size": 500, "max_blocks": 2000, "target_block_errors": 10 ** 9}
    first = sweep([a], BSC, [0.02, 0.03], seed=13, **args)
    second = sweep([b], BSC, [0.02, 0.03], seed=13, **args)
    check("один seed — тот же FER",
          [row["fer"] for row in first] == [row["fer"] for row in second])


def test_wilson_interval():
    """Интервал Уилсона ведёт себя корректно в крайних случаях."""
    lo, hi = wilson_interval(0, 20_000)
    check("нуль ошибок даёт ненулевую верхнюю границу", lo == 0.0 and 0 < hi < 0.001,
          f"[{lo}, {hi:.2e}]")
    lo, hi = wilson_interval(500, 1000)
    check("доля 0,5 лежит внутри своего интервала", lo < 0.5 < hi,
          f"[{lo:.4f}, {hi:.4f}]")


def main():
    for test in (
        test_graph_is_regular,
        test_girth_known_graphs,
        test_generator_matrix,
        test_single_error_is_corrected,
        test_no_noise_is_identity,
        test_zero_codeword_assumption,
        test_monotonicity,
        test_fixed_errors_weight,
        test_reproducibility,
        test_wilson_interval,
    ):
        print(f"\n{test.__doc__.splitlines()[0]}")
        test()

    print(f"\nпройдено {sum(PASSED)} из {len(PASSED)}")
    return 0 if all(PASSED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
