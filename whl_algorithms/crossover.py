"""Crossover operators for chromosome search."""

from __future__ import annotations

import numpy as np
from whl_core.chromosome import Chromosome, ensure_binary_vector


def two_point_crossover_vector(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two child vectors made by swapping one two-point segment."""
    vector_a = ensure_binary_vector(parent_a, name="parent_a")
    vector_b = ensure_binary_vector(parent_b, name="parent_b")

    if len(vector_a) != len(vector_b):
        raise ValueError("parent vectors must have the same length.")

    length = len(vector_a)
    child_a = vector_a.copy()
    child_b = vector_b.copy()
    if length < 2:
        return child_a, child_b

    cut_1, cut_2 = sorted(rng.choice(np.arange(1, length + 1), size=2, replace=False))
    child_a[cut_1:cut_2] = vector_b[cut_1:cut_2]
    child_b[cut_1:cut_2] = vector_a[cut_1:cut_2]

    return child_a.astype(np.uint8, copy=False), child_b.astype(np.uint8, copy=False)


def chromosome_two_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: np.random.Generator,
    crossover_prob: float = 1.0,
) -> tuple[Chromosome, Chromosome]:
    """Apply independent two-point crossover to parent ``h`` and ``v`` vectors."""
    if not 0.0 <= crossover_prob <= 1.0:
        raise ValueError("crossover_prob must be in [0, 1].")

    parent_a.validate()
    parent_b.validate()
    if len(parent_a.h) != len(parent_b.h) or len(parent_a.v) != len(parent_b.v):
        raise ValueError("parents must have matching h and v dimensions.")

    if rng.random() > crossover_prob:
        return parent_a.copy(), parent_b.copy()

    child_a_h, child_b_h = two_point_crossover_vector(parent_a.h, parent_b.h, rng)
    child_a_v, child_b_v = two_point_crossover_vector(parent_a.v, parent_b.v, rng)

    child_a = Chromosome(h=child_a_h, v=child_a_v)
    child_b = Chromosome(h=child_b_h, v=child_b_v)
    child_a.validate(rows=len(parent_a.h), cols=len(parent_a.v))
    child_b.validate(rows=len(parent_b.h), cols=len(parent_b.v))
    return child_a, child_b


def repair_empty_chromosome(
    chromosome: Chromosome,
    rng: np.random.Generator,
) -> Chromosome:
    """Activate one random bit if both chromosome vectors are empty."""
    repaired = chromosome.copy()
    if sum(repaired.active_count()) > 0:
        repaired.validate()
        return repaired

    total_length = len(repaired.h) + len(repaired.v)
    if total_length == 0:
        repaired.validate()
        return repaired

    index = int(rng.integers(0, total_length))
    if index < len(repaired.h):
        repaired.h[index] = 1
    else:
        repaired.v[index - len(repaired.h)] = 1

    repaired.validate()
    return repaired


def make_offspring_pair(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: np.random.Generator,
    crossover_prob: float = 1.0,
) -> tuple[Chromosome, Chromosome]:
    """Create a valid repaired offspring pair from two parents."""
    child_a, child_b = chromosome_two_point_crossover(
        parent_a,
        parent_b,
        rng,
        crossover_prob=crossover_prob,
    )
    child_a = repair_empty_chromosome(child_a, rng)
    child_b = repair_empty_chromosome(child_b, rng)
    child_a.validate(rows=len(parent_a.h), cols=len(parent_a.v))
    child_b.validate(rows=len(parent_b.h), cols=len(parent_b.v))
    return child_a, child_b


__all__ = [
    "chromosome_two_point_crossover",
    "make_offspring_pair",
    "repair_empty_chromosome",
    "two_point_crossover_vector",
]
