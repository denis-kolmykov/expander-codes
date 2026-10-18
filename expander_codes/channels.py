"""Модели канала: вносят ошибки в пакет кодовых слов."""

from abc import ABC, abstractmethod

import numpy as np


class Channel(ABC):
    @property
    @abstractmethod
    def name(self):
        """Имя канала для записи в результаты."""

    @property
    @abstractmethod
    def level(self):
        """Числовой уровень шума — то, по чему строится развёртка."""

    @abstractmethod
    def transmit(self, codewords, rng):
        """(B, n) -> (B, n): принятые слова."""

    def params(self):
        return {"channel": self.name, "level": self.level}


class BSC(Channel):
    """Двоичный симметричный канал: каждый бит инвертируется с вероятностью p."""

    def __init__(self, p):
        self.p = float(p)

    @property
    def name(self):
        return "BSC"

    @property
    def level(self):
        return self.p

    def transmit(self, codewords, rng):
        noise = (rng.random(codewords.shape) < self.p).astype(np.uint8)
        return codewords ^ noise


class FixedErrors(Channel):
    """Ровно t ошибок в каждом блоке, позиции выбираются равномерно.

    Нужен для проверки гарантий вида «код исправляет до t ошибок»:
    в отличие от BSC здесь вес ошибки не случаен.
    """

    def __init__(self, t):
        self.t = int(t)

    @property
    def name(self):
        return "FixedErrors"

    @property
    def level(self):
        return self.t

    def transmit(self, codewords, rng):
        batch, n = codewords.shape
        noise = np.zeros((batch, n), dtype=np.uint8)
        positions = np.argsort(rng.random((batch, n)), axis=1)[:, : self.t]
        np.put_along_axis(noise, positions, 1, axis=1)
        return codewords ^ noise
