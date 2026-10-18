"""Вероятностное оценивание кодов методом Монте-Карло."""

import math
import time

import numpy as np


def wilson_interval(successes, trials, z=1.96):
    """Доверительный интервал Уилсона для доли successes/trials.

    successes — число наступлений события, trials — число испытаний,
    z — квантиль нормального распределения (1.96 соответствует 95%).
    В отличие от нормального интервала остаётся корректным при малых долях,
    а это как раз режим редких ошибок декодирования.
    """
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(code, channel, rng, batch_size=1000, max_blocks=100_000,
             target_block_errors=100, use_zero_codeword=True):
    """Прогон кода через канал до накопления статистики.

    Останавливается, когда набрано target_block_errors ошибочных блоков
    либо пройдено max_blocks блоков — что наступит раньше.

    use_zero_codeword=True передаёт нулевое кодовое слово. Для линейного кода
    и симметричного канала вероятность ошибки от переданного слова не зависит,
    а кодирование пакета плотной матрицей G — самая дорогая операция в прогоне.

    Возвращает словарь метрик:
      fer — доля блоков, декодированных неверно (с границами интервала);
      ber — доля неверных бит;
      undetected — доля блоков, где декодер сошёлся к чужому кодовому слову;
      not_converged — доля блоков без нулевого синдрома за max_iter итераций;
      mean_iterations, std_iterations — среднее и СКО числа итераций;
      blocks, block_errors, bit_errors, seconds, blocks_per_second.
    """
    blocks = 0
    block_errors = 0
    bit_errors = 0
    undetected = 0
    not_converged = 0
    iterations_sum = 0
    iterations_sq_sum = 0
    started = time.perf_counter()

    while blocks < max_blocks and block_errors < target_block_errors:
        size = min(batch_size, max_blocks - blocks)
        if use_zero_codeword:
            X = code.zero_codeword(size)
        else:
            messages = rng.integers(0, 2, size=(size, code.k), dtype=np.uint8)
            X = code.encode(messages)

        Y = channel.transmit(X, rng)
        X_hat, info = code.decode(Y)

        wrong_bits = (X_hat != X)
        wrong_blocks = wrong_bits.any(axis=1)

        blocks += size
        block_errors += int(wrong_blocks.sum())
        bit_errors += int(wrong_bits.sum())
        undetected += int((wrong_blocks & info["converged"]).sum())
        not_converged += int((~info["converged"]).sum())
        iterations_sum += int(info["iterations"].sum())
        iterations_sq_sum += int((info["iterations"].astype(np.int64) ** 2).sum())

    seconds = time.perf_counter() - started
    fer = block_errors / blocks
    lo, hi = wilson_interval(block_errors, blocks)

    # Среднеквадратичное отклонение числа итераций через сумму квадратов:
    # D = E[X^2] - (E[X])^2, корень из неё и есть СКО.
    mean_iterations = iterations_sum / blocks
    variance = max(0.0, iterations_sq_sum / blocks - mean_iterations ** 2)

    return {
        "blocks": blocks,
        "block_errors": block_errors,
        "bit_errors": bit_errors,
        "fer": fer,
        "fer_lo": lo,
        "fer_hi": hi,
        "ber": bit_errors / (blocks * code.n),
        "undetected": undetected / blocks,
        "not_converged": not_converged / blocks,
        "mean_iterations": mean_iterations,
        "std_iterations": math.sqrt(variance),
        "seconds": seconds,
        "blocks_per_second": blocks / seconds if seconds > 0 else float("nan"),
    }


def sweep(codes, channel_factory, levels, seed, **kwargs):
    """Прогон списка кодов по развёртке уровней шума.

    channel_factory — функция уровня, возвращающая канал (например, BSC);
    seed — зерно генератора шума; для каждой пары «код + уровень» из него
    выводится своё зерно, которое записывается в строку результата.
    Поэтому любую отдельную точку можно переиграть, не повторяя всю развёртку.

    Возвращает список строк-словарей, готовых к записи в CSV.
    """
    rows = []
    parent = np.random.SeedSequence(seed)
    children = parent.spawn(len(codes) * len(levels))

    for index, (code, level) in enumerate(
        (code, level) for code in codes for level in levels
    ):
        child = children[index]
        channel = channel_factory(level)
        metrics = evaluate(code, channel, np.random.default_rng(child), **kwargs)
        rows.append({
            "code": code.name,
            "n": code.n,
            "k": code.k,
            "rate": code.rate,
            **{f"code_{key}": value for key, value in code.params().items()},
            **channel.params(),
            "sweep_seed": seed,
            "point_seed": child.entropy if child.spawn_key == () else
                          f"{seed}:{child.spawn_key[0]}",
            **metrics,
        })
    return rows


def noise_threshold(rows, code_name, target_fer=1e-3):
    """Наибольший уровень шума, при котором FER кода не превосходит target_fer.

    Оценка по сетке: возвращается последний уровень развёртки, прошедший порог,
    поэтому точность ограничена шагом сетки.
    """
    levels = sorted(
        row["level"] for row in rows
        if row["code"] == code_name and row["fer"] <= target_fer
    )
    return levels[-1] if levels else None
