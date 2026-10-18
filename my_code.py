"""Пример: как подключить свой код к системе сравнения.

Код повторения: каждый информационный бит передаётся r раз подряд.
Декодер — голосование большинством внутри каждой группы из r бит.

Запуск: python my_code.py
"""

import numpy as np

from expander_codes import BSC, Code, ExpanderCode, print_table, save_rows, sweep


class RepetitionCode(Code):
    """Код повторения: k информационных бит, каждый повторён r раз."""

    def __init__(self, k, r):
        if r % 2 == 0:
            raise ValueError("r должно быть нечётным, иначе голосование неоднозначно")
        self._k = k
        self.r = r

    # --- четыре обязательных свойства -------------------------------------

    @property
    def name(self):
        return f"repetition_k{self._k}_r{self.r}"

    @property
    def n(self):
        return self._k * self.r

    @property
    def k(self):
        return self._k

    # --- кодирование ------------------------------------------------------

    def encode(self, messages):
        """(B, k) -> (B, n): каждый бит повторяется r раз."""
        M = np.asarray(messages, dtype=np.uint8)
        return np.repeat(M, self.r, axis=1)

    # --- декодирование ----------------------------------------------------

    def decode(self, received):
        """(B, n) -> (B, n) плюс служебная информация.

        Принятое слово режется на группы по r бит; в каждой группе
        побеждает тот символ, которого больше.
        """
        Y = np.asarray(received, dtype=np.uint8)
        batch = Y.shape[0]

        groups = Y.reshape(batch, self._k, self.r)      # (B, k, r)
        winners = (groups.sum(axis=2) * 2 > self.r)     # (B, k) — большинство единиц
        X_hat = np.repeat(winners.astype(np.uint8), self.r, axis=1)

        # Результат голосования всегда является кодовым словом, поэтому
        # декодер формально всегда «сходится» и делает это за одну итерацию.
        return X_hat, {
            "iterations": np.ones(batch, dtype=np.int64),
            "converged": np.ones(batch, dtype=bool),
        }

    # --- что записать в results.csv ---------------------------------------

    def params(self):
        return {"n": self.n, "r": self.r}


def main():
    codes = [
        RepetitionCode(k=336, r=3),
        ExpanderCode.random_regular(n=1008, dv=3, dc=6, seed=2027, flip_mode="max"),
    ]

    rows = sweep(
        codes,
        BSC,
        [0.01, 0.02, 0.05, 0.1],
        seed=1,
        batch_size=500,
        max_blocks=20_000,
        target_block_errors=100,
    )

    print_table(rows, columns=("code", "n", "k", "rate", "level",
                               "fer", "fer_lo", "fer_hi",
                               "undetected", "not_converged", "blocks"))
    save_rows(rows, "my_results.csv")


if __name__ == "__main__":
    main()
