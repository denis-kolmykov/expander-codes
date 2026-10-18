"""Общий интерфейс кода: всё, что умеет кодировать и декодировать пакеты слов."""

from abc import ABC, abstractmethod

import numpy as np


class Code(ABC):
    """Базовый класс для любого кода, подключаемого к системе сравнения.

    Соглашения по форматам массивов:
      сообщения  — uint8, форма (B, k);
      кодовые слова и принятые слова — uint8, форма (B, n);
    где B — размер пакета.
    """

    @property
    @abstractmethod
    def name(self):
        """Короткое имя кода для таблиц и графиков."""

    @property
    @abstractmethod
    def n(self):
        """Длина кодового слова."""

    @property
    @abstractmethod
    def k(self):
        """Размерность кода (число информационных бит)."""

    @property
    def rate(self):
        return self.k / self.n

    @abstractmethod
    def encode(self, messages):
        """(B, k) -> (B, n)."""

    @abstractmethod
    def decode(self, received):
        """(B, n) -> (X_hat, info).

        X_hat — оценки кодовых слов, форма (B, n).
        info — словарь с полями:
          'iterations' — int-массив (B,), число итераций декодера;
          'converged'  — bool-массив (B,), найдено ли слово с нулевым синдромом.
        """

    @abstractmethod
    def params(self):
        """Словарь параметров кода для записи в результаты."""

    def zero_codeword(self, batch_size):
        """Нулевое кодовое слово — лежит в любом линейном коде."""
        return np.zeros((batch_size, self.n), dtype=np.uint8)
