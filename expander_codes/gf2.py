"""Линейная алгебра над полем GF(2)."""

import numpy as np


def rref(A):
    """Приведённый ступенчатый вид матрицы над GF(2).

    Возвращает (R, pivots): R — матрица в ступенчатом виде,
    pivots — список индексов ведущих столбцов.
    """
    R = (np.asarray(A) % 2).astype(np.uint8).copy()
    m, n = R.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        nz = np.nonzero(R[row:, col])[0]
        if nz.size == 0:
            continue
        i = row + nz[0]
        if i != row:
            R[[row, i]] = R[[i, row]]
        mask = R[:, col].astype(bool).copy()
        mask[row] = False
        if mask.any():
            R[mask] ^= R[row]
        pivots.append(col)
        row += 1
    return R, pivots


def rank(A):
    """Ранг матрицы над GF(2)."""
    return len(rref(A)[1])


def generator_from_parity_check(H):
    """Порождающая матрица G по проверочной матрице H.

    Строки G образуют базис ядра H, то есть G @ H.T = 0 над GF(2).
    Возвращает (G, free, pivots): free — позиции информационных бит,
    pivots — позиции проверочных бит.
    """
    R, pivots = rref(H)
    n = H.shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    G = np.zeros((len(free), n), dtype=np.uint8)
    for j, f in enumerate(free):
        G[j, f] = 1
        for i, p in enumerate(pivots):
            G[j, p] = R[i, f]
    return G, np.array(free, dtype=np.int64), np.array(pivots, dtype=np.int64)
