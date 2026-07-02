"""Chromosome representation for row and column aisle tendencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

def ensure_binary_vector(
    vector: Any,
    expected_length: int | None = None,
    name: str = "vector",
) -> np.ndarray:
    """Return *vector* as a validated 1D uint8 binary numpy array."""
    array = np.asarray(vector)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")

    if expected_length is not None and len(array) != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}; got {len(array)}."
        )

    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary 0/1 values.")

    return array.astype(np.uint8, copy=True)

@dataclass
class Chromosome:
    """Binary row/column aisle tendency chromosome."""

    h: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        self.h = ensure_binary_vector(self.h, name="h")
        self.v = ensure_binary_vector(self.v, name="v")

    def validate(self, rows: int | None = None, cols: int | None = None) -> None:
        """Validate chromosome vectors and optional expected dimensions."""
        self.h = ensure_binary_vector(self.h, expected_length=rows, name="h")
        self.v = ensure_binary_vector(self.v, expected_length=cols, name="v")

    def copy(self) -> Chromosome:
        """Return an independent copy of this chromosome."""
        return Chromosome(h=self.h.copy(), v=self.v.copy())

    def active_h_indices(self) -> list[int]:
        """Return active row tendency indices."""
        return np.flatnonzero(self.h).astype(int).tolist()

    def active_v_indices(self) -> list[int]:
        """Return active column tendency indices."""
        return np.flatnonzero(self.v).astype(int).tolist()

    def as_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return a stable immutable signature of active indices."""
        return (
            tuple(self.active_h_indices()),
            tuple(self.active_v_indices()),
        )

    def active_count(self) -> tuple[int, int]:
        """Return counts of active row and column tendencies."""
        return int(self.h.sum()), int(self.v.sum())

    @staticmethod
    def from_indices(
        rows: int,
        cols: int,
        h_indices: list[int],
        v_indices: list[int],
    ) -> Chromosome:
        """Build a chromosome from active row and column index lists."""
        if rows <= 0:
            raise ValueError("rows must be positive.")
        if cols <= 0:
            raise ValueError("cols must be positive.")

        h = np.zeros(rows, dtype=np.uint8)
        v = np.zeros(cols, dtype=np.uint8)

        for index in h_indices:
            if index < 0 or index >= rows:
                raise ValueError(f"h index {index} is outside [0, {rows}).")
            h[index] = 1

        for index in v_indices:
            if index < 0 or index >= cols:
                raise ValueError(f"v index {index} is outside [0, {cols}).")
            v[index] = 1

        return Chromosome(h=h, v=v)

def chromosome_signature(
    chromosome: Chromosome,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the stable signature for a chromosome."""
    return chromosome.as_tuple()

__all__ = [
    "Chromosome",
    "chromosome_signature",
    "ensure_binary_vector",
]
