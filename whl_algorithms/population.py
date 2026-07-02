"""Population initialization for chromosome-based layout search."""

from __future__ import annotations

import math

import numpy as np
from whl_core.chromosome import Chromosome, chromosome_signature


def compute_aspect_ratio_betas(rows: int, cols: int) -> tuple[float, float]:
    """Return row/column spacing betas based on layout aspect ratio."""
    if rows <= 0:
        raise ValueError("rows must be positive.")
    if cols <= 0:
        raise ValueError("cols must be positive.")

    ratio = rows / cols
    if ratio > 1.2:
        return 0.75, 0.35
    if ratio < 0.8:
        return 0.35, 0.75
    return 0.50, 0.50


def adaptive_spacing(
    index: int,
    population_size: int,
    dimension: int,
    beta: float,
    alpha: float = 0.5,
) -> int:
    """Compute adaptive minimum spacing for one chromosome position."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if index < 0 or index >= population_size:
        raise ValueError("index must be inside the population range.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    if population_size == 1:
        decay = 1.0
    else:
        decay = math.exp(-alpha * index / (population_size - 1))

    spacing = math.floor(beta * dimension / math.sqrt(population_size) * decay)
    return max(1, int(spacing))


def sample_indices_with_spacing(
    length: int,
    spacing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a binary vector whose active indices respect minimum spacing."""
    if length < 0:
        raise ValueError("length must be non-negative.")
    if spacing < 1:
        raise ValueError("spacing must be at least 1.")

    vector = np.zeros(length, dtype=np.uint8)
    if length == 0:
        return vector

    selected: list[int] = []
    selection_probability = min(0.75, max(0.20, 1.0 / (spacing + 1)))
    for candidate in rng.permutation(length):
        candidate_int = int(candidate)
        if (
            all(abs(candidate_int - existing) >= spacing for existing in selected)
            and rng.random() <= selection_probability
        ):
            selected.append(candidate_int)

    if not selected:
        selected.append(int(rng.integers(0, length)))

    vector[selected] = 1
    return vector


def initialize_population(
    rows: int,
    cols: int,
    population_size: int,
    seed: int | None = None,
    alpha: float = 0.5,
) -> list[Chromosome]:
    """Initialize a deterministic adaptive population of chromosomes."""
    if rows <= 0:
        raise ValueError("rows must be positive.")
    if cols <= 0:
        raise ValueError("cols must be positive.")
    if population_size <= 0:
        raise ValueError("population_size must be positive.")

    rng = np.random.default_rng(seed)
    beta_h, beta_v = compute_aspect_ratio_betas(rows, cols)
    population: list[Chromosome] = []
    signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    max_retries = 25

    for index in range(population_size):
        h_spacing = adaptive_spacing(index, population_size, rows, beta_h, alpha)
        v_spacing = adaptive_spacing(index, population_size, cols, beta_v, alpha)

        accepted: Chromosome | None = None
        fallback: Chromosome | None = None
        for _ in range(max_retries):
            chromosome = Chromosome(
                h=sample_indices_with_spacing(rows, h_spacing, rng),
                v=sample_indices_with_spacing(cols, v_spacing, rng),
            )
            fallback = chromosome
            signature = chromosome_signature(chromosome)
            if signature not in signatures:
                accepted = chromosome
                break

        if accepted is None:
            if fallback is None:
                raise RuntimeError("failed to sample a chromosome.")
            accepted = fallback

        population.append(accepted)
        signatures.add(chromosome_signature(accepted))

    return population


def population_signatures(
    population: list[Chromosome],
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return unique chromosome signatures for a population."""
    return {chromosome_signature(chromosome) for chromosome in population}


__all__ = [
    "adaptive_spacing",
    "compute_aspect_ratio_betas",
    "initialize_population",
    "population_signatures",
    "sample_indices_with_spacing",
]
