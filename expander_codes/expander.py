"""Экспандерный (LDPC) код с декодером перевёрткой битов."""

import numpy as np
from scipy.sparse import csr_matrix

from .base import Code
from .gf2 import generator_from_parity_check, rref
from .graphs import expansion_estimate, girth, random_regular_bipartite

EXPANSION_FRACTION = 0.02   # доля вершин в подмножестве при оценке расширения
EXPANSION_TRIALS = 500      # число случайных подмножеств
EXPANSION_SEED = 0          # отдельное зерно, чтобы оценка не зависела от прогона


class ExpanderCode(Code):
    """Код, заданный проверочной матрицей H разреженного двудольного графа.

    flip_mode задаёт правило декодирования:
      'serial'   — за итерацию переворачивается один бит с наибольшим
                   числом невыполненных проверок (правило Сипсера–Спилмана);
      'max'      — за итерацию переворачиваются все биты, набравшие
                   максимальное число невыполненных проверок;
      'majority' — за итерацию переворачиваются все биты, у которых
                   невыполненных проверок больше, чем выполненных.
    """

    def __init__(self, H, name=None, max_iter=50, flip_mode="serial", meta=None):
        self.H = csr_matrix(H, dtype=np.int32)
        self.Ht = csr_matrix(self.H.T, dtype=np.int32)
        self.m, self._n = self.H.shape
        self.degree = np.asarray(self.H.sum(axis=0)).ravel().astype(np.int32)
        self.max_iter = max_iter
        self.flip_mode = flip_mode
        self.meta = dict(meta or {})
        self._name = name or f"expander_n{self._n}_m{self.m}_{flip_mode}"
        self._k = None
        self._G = None
        self._girth = None
        self._expansion = None

    @classmethod
    def random_regular(cls, n, dv, dc, seed, max_iter=50, flip_mode="serial", name=None):
        """Случайный регулярный код; seed — зерно генератора графа.

        Зерно задаётся числом, а не готовым генератором, чтобы граф строился
        одинаково при любом порядке вызовов и записывался в результаты.
        """
        rng = np.random.default_rng(seed)
        H = random_regular_bipartite(n, dv, dc, rng)
        return cls(
            H,
            name=name or f"expander_n{n}_dv{dv}_dc{dc}_{flip_mode}",
            max_iter=max_iter,
            flip_mode=flip_mode,
            meta={"dv": dv, "dc": dc, "graph_seed": seed},
        )

    @property
    def name(self):
        return self._name

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        if self._k is None:
            self._k = self._n - len(rref(self.H.toarray())[1])
        return self._k

    @property
    def design_rate(self):
        """Проектная скорость 1 - m/n: совпадает с истинной, если строки H независимы."""
        return 1.0 - self.m / self._n

    @property
    def girth(self):
        """Обхват графа Таннера; считается один раз и запоминается."""
        if self._girth is None:
            self._girth = girth(self.H)
        return self._girth

    @property
    def expansion(self):
        """Оценка расширения на подмножествах размера EXPANSION_FRACTION * n."""
        if self._expansion is None:
            set_size = max(2, round(EXPANSION_FRACTION * self._n))
            self._expansion = expansion_estimate(
                self.H, set_size, EXPANSION_TRIALS,
                np.random.default_rng(EXPANSION_SEED),
            )
        return self._expansion

    @property
    def G(self):
        """Порождающая матрица; вычисляется лениво, так как она плотная."""
        if self._G is None:
            G, free, pivots = generator_from_parity_check(self.H.toarray())
            self._G = G
            self._k = G.shape[0]
            self.info_positions = free
            self.check_positions = pivots
        return self._G

    def encode(self, messages):
        M = np.asarray(messages, dtype=np.int32)
        return ((M @ self.G.astype(np.int32)) & 1).astype(np.uint8)

    def decode(self, received):
        Y = np.asarray(received, dtype=np.uint8)
        X = Y.T.astype(np.int32).copy()          # (n, B): работаем по столбцам
        batch = X.shape[1]
        iterations = np.full(batch, self.max_iter, dtype=np.int64)
        converged = np.zeros(batch, dtype=bool)

        for it in range(self.max_iter + 1):
            S = self.H @ X & 1                   # (m, B) синдромы
            ok = ~S.any(axis=0)
            newly = ok & ~converged
            iterations[newly] = it
            converged |= ok
            if converged.all() or it == self.max_iter:
                break

            unsat = self.Ht @ S                  # (n, B) невыполненных проверок на бит
            active = ~converged

            if self.flip_mode == "majority":
                candidates = (2 * unsat > self.degree[:, None]) & active[None, :]
            else:
                candidates = (unsat > 0) & active[None, :]
            if not candidates.any():
                break

            if self.flip_mode == "majority":
                flip = candidates
            elif self.flip_mode == "max":
                best = np.where(candidates, unsat, -1).max(axis=0)
                flip = candidates & (unsat == best[None, :])
            elif self.flip_mode == "serial":
                scores = np.where(candidates, unsat, -1)
                best = scores.argmax(axis=0)
                flip = np.zeros_like(candidates)
                movable = candidates.any(axis=0)
                flip[best[movable], np.nonzero(movable)[0]] = True
            else:
                raise ValueError(f"неизвестный flip_mode: {self.flip_mode}")

            X ^= flip.astype(np.int32)

        return X.T.astype(np.uint8), {"iterations": iterations, "converged": converged}

    def params(self):
        return {
            "n": self.n,
            "m": self.m,
            "design_rate": round(self.design_rate, 6),
            "max_iter": self.max_iter,
            "flip_mode": self.flip_mode,
            "girth": self.girth,
            "expansion": round(self.expansion, 6),
            "expansion_set_size": max(2, round(EXPANSION_FRACTION * self._n)),
            "expansion_trials": EXPANSION_TRIALS,
            **self.meta,
        }
