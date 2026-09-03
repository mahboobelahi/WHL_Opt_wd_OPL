"""Configuration-only parameter policies for warehouse layout experiments."""

from __future__ import annotations

import math


BENCHMARK_PARAMETER_OVERRIDES = {
    (56, 22): {
        "population_size": 11,
        "generations": 16,
        "beam_width": 4,
        "max_depth": 30,
    },
    (56, 26): {
        "population_size": 12,
        "generations": 18,
        "beam_width": 3,
        "max_depth": 30,
    },
}


def adaptive_mutation_probability(
    generation: int,
    max_generations: int,
    base: float = 0.1,
    min_val: float = 0.01,
) -> float:
    """Return a linearly decayed mutation probability with a lower bound."""
    if generation < 0:
        raise ValueError("generation must be non-negative.")
    if max_generations <= 0:
        raise ValueError("max_generations must be positive.")
    if base < 0:
        raise ValueError("base must be non-negative.")
    if min_val < 0:
        raise ValueError("min_val must be non-negative.")

    return max(min_val, base * (1 - generation / max_generations))


def min_fragment_size(aisle_width: int) -> int:
    """Return the minimum fragment size derived from an aisle width."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    return max(2, math.ceil(aisle_width * 1.5 / 2))


def auto_hyperparams(rows: int, cols: int) -> dict[str, int | float]:
    """Return search parameters for a warehouse grid.

    The two Kovacs benchmark geometries use the fixed parameter tuples reported
    for the paper experiments. Other grid sizes retain the generic size-based
    policy.
    """
    if rows <= 0:
        raise ValueError("rows must be positive.")
    if cols <= 0:
        raise ValueError("cols must be positive.")

    override = BENCHMARK_PARAMETER_OVERRIDES.get((rows, cols))
    if override is None:
        area = rows * cols
        population_size = max(10, min(40, area // 1000 + 10))
        generations = int(1.5 * population_size)
        beam_width = max(3, min(8, area // 1500 + 3))
        max_depth = min(30, max(rows, cols))
    else:
        population_size = int(override["population_size"])
        generations = int(override["generations"])
        beam_width = int(override["beam_width"])
        max_depth = int(override["max_depth"])

    crossover_prob = 1.0
    num_offspring = int(population_size * crossover_prob)

    return {
        "population_size": population_size,
        "generations": generations,
        "beam_width": beam_width,
        "max_depth": max_depth,
        "crossover_prob": crossover_prob,
        "num_offspring": num_offspring,
    }


__all__ = [
    "BENCHMARK_PARAMETER_OVERRIDES",
    "adaptive_mutation_probability",
    "auto_hyperparams",
    "min_fragment_size",
]
