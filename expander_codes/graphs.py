"""Построение двудольных графов для экспандерных кодов."""

import numpy as np
from scipy.sparse import csr_matrix


def random_regular_bipartite(n, dv, dc, rng, max_tries=50):
    """Случайный (dv, dc)-регулярный двудольный граф без кратных рёбер.

    n   — число переменных вершин (длина кода);
    dv  — степень переменной вершины (число проверок на бит);
    dc  — степень проверочной вершины (число бит в проверке).
    Требуется, чтобы n * dv делилось на dc; число проверок m = n * dv / dc.

    Переменные вершины обрабатываются по очереди: для каждой выбираются dv
    различных проверок с вероятностью, пропорциональной остатку их степени.
    Так все проверки получают ровно по dc рёбер, а кратные рёбра невозможны
    по построению. Если в конце остаток степени сосредоточен менее чем в dv
    проверках, попытка повторяется с начала.

    Возвращает H — разреженную проверочную матрицу формы (m, n).
    """
    if (n * dv) % dc != 0:
        raise ValueError(f"n*dv={n * dv} не делится на dc={dc}")
    if dv > n * dv // dc:
        raise ValueError("dv не может превосходить число проверок m = n*dv/dc")
    m = n * dv // dc

    for _ in range(max_tries):
        capacity = np.full(m, dc, dtype=np.int64)
        rows = np.empty(n * dv, dtype=np.int64)
        cols = np.repeat(np.arange(n, dtype=np.int64), dv)
        failed = False

        for position, v in enumerate(rng.permutation(n)):
            available = np.nonzero(capacity)[0]
            if available.size < dv:
                failed = True
                break
            weights = capacity[available] / capacity[available].sum()
            chosen = rng.choice(available, size=dv, replace=False, p=weights)
            capacity[chosen] -= 1
            rows[position * dv:(position + 1) * dv] = chosen
            cols[position * dv:(position + 1) * dv] = v

        if not failed:
            data = np.ones(n * dv, dtype=np.int32)
            return csr_matrix((data, (rows, cols)), shape=(m, n))

    raise RuntimeError(
        "не удалось построить граф за отведённое число попыток; "
        "проверьте совместимость n, dv и dc"
    )


def girth(H):
    """Обхват графа Таннера — длина кратчайшего цикла. inf, если циклов нет.

    Граф двудольный, поэтому все циклы чётной длины и обхват не меньше 4.
    Цикл длины 4 означает, что два бита входят в одни и те же две проверки:
    такая пара битов неразличима для декодера по этим проверкам, поэтому
    короткие циклы ухудшают декодирование.

    Алгоритм: обход в ширину из каждой вершины. Если встречается ребро (u, v),
    где v уже посещена и не является предком u, найден цикл длины
    d[u] + d[v] + 1, где d — расстояние от стартовой вершины.
    Минимум по всем стартам и есть обхват.
    """
    H = csr_matrix(H)
    m, n = H.shape
    Hc = H.tocsc()

    # Единая нумерация вершин: 0..m-1 проверки, m..m+n-1 биты.
    neighbours = [None] * (m + n)
    for i in range(m):
        neighbours[i] = H.indices[H.indptr[i]:H.indptr[i + 1]] + m
    for j in range(n):
        neighbours[m + j] = Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]

    total = m + n
    best = np.inf
    for start in range(total):
        depth = np.full(total, -1, dtype=np.int64)
        parent = np.full(total, -1, dtype=np.int64)
        depth[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if 2 * depth[u] >= best - 1:
                break
            for v in neighbours[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    best = min(best, depth[u] + depth[v] + 1)
    return int(best) if np.isfinite(best) else np.inf


def expansion_estimate(H, set_size, trials, rng):
    """Оценка расширения снизу по случайным подмножествам переменных вершин.

    Для trials случайных подмножеств размера set_size считается число соседей
    среди проверочных вершин; возвращается минимальное отношение
    |N(S)| / (dv * |S|) по всем испытаниям.

    Это именно оценка сверху на худший случай: точное значение расширения
    требует перебора всех подмножеств и вычислительно недоступно.
    """
    if set_size < 1 or set_size > H.shape[1]:
        raise ValueError(f"размер подмножества вне диапазона: {set_size}")
    n = H.shape[1]
    Hc = H.tocsc()
    dv = np.diff(Hc.indptr).mean()
    worst = np.inf
    for _ in range(trials):
        S = rng.choice(n, size=set_size, replace=False)
        neighbours = np.unique(np.concatenate([
            Hc.indices[Hc.indptr[v]:Hc.indptr[v + 1]] for v in S
        ]))
        worst = min(worst, neighbours.size / (dv * set_size))
    return float(worst)
