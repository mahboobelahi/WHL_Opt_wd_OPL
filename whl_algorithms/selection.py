"""Parent-selection utilities for chromosome search."""

from __future__ import annotations

import numpy as np
from whl_core.chromosome import Chromosome, chromosome_signature


def _validate_population(population: list[Chromosome]) -> None:
    if not population:
        raise ValueError("population must not be empty.")


def _candidate_indices(
    population_size: int,
    rng: np.random.Generator,
    tournament_size: int,
) -> np.ndarray:
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive.")
    effective_tournament_size = min(tournament_size, population_size)
    return rng.choice(population_size, size=effective_tournament_size, replace=False)


def random_parent_pair(
    population: list[Chromosome],
    rng: np.random.Generator,
    allow_same: bool = False,
) -> tuple[Chromosome, Chromosome]:
    """Return two randomly selected parent references from ``population``."""
    _validate_population(population)
    if not allow_same and len(population) < 2:
        raise ValueError("at least two parents are required when allow_same=False.")

    replace = allow_same
    indices = rng.choice(len(population), size=2, replace=replace)
    while not allow_same and int(indices[0]) == int(indices[1]):
        indices = rng.choice(len(population), size=2, replace=False)
    return population[int(indices[0])], population[int(indices[1])]


def tournament_select(
    population: list[Chromosome],
    rng: np.random.Generator,
    scores: list[float] | None = None,
    tournament_size: int = 2,
    minimize: bool = True,
) -> Chromosome:
    """Select one parent by tournament."""
    _validate_population(population)
    if scores is not None and len(scores) != len(population):
        raise ValueError("scores length must match population length.")

    candidates = _candidate_indices(len(population), rng, tournament_size)
    if scores is None:
        return population[int(rng.choice(candidates))]

    candidate_scores = [(int(index), scores[int(index)]) for index in candidates]
    if minimize:
        best_score = min(score for _, score in candidate_scores)
    else:
        best_score = max(score for _, score in candidate_scores)

    tied = [index for index, score in candidate_scores if score == best_score]
    return population[int(rng.choice(tied))]


def nsga2_tournament_select(
    population: list[Chromosome],
    rng: np.random.Generator,
    ranks: list[int],
    crowding_distances: list[float],
    tournament_size: int = 2,
) -> Chromosome:
    """Select one parent using NSGA-II rank and crowding tournament rules."""
    _validate_population(population)
    if len(ranks) != len(population):
        raise ValueError("ranks length must match population length.")
    if len(crowding_distances) != len(population):
        raise ValueError("crowding_distances length must match population length.")

    candidates = _candidate_indices(len(population), rng, tournament_size)
    best_rank = min(ranks[int(index)] for index in candidates)
    rank_tied = [int(index) for index in candidates if ranks[int(index)] == best_rank]
    best_crowding = max(crowding_distances[index] for index in rank_tied)
    crowding_tied = [
        index for index in rank_tied if crowding_distances[index] == best_crowding
    ]
    return population[int(rng.choice(crowding_tied))]


def nsga2_parent_pair(
    population: list[Chromosome],
    rng: np.random.Generator,
    ranks: list[int],
    crowding_distances: list[float],
    allow_same: bool = False,
) -> tuple[Chromosome, Chromosome]:
    """Return two NSGA-II tournament-selected parent references."""
    if not allow_same and len(population) < 2:
        raise ValueError("at least two parents are required when allow_same=False.")

    first = nsga2_tournament_select(population, rng, ranks, crowding_distances)
    second = nsga2_tournament_select(population, rng, ranks, crowding_distances)
    if allow_same:
        return first, second

    first_signature = chromosome_signature(first)
    for _ in range(25):
        if second is not first and chromosome_signature(second) != first_signature:
            return first, second
        second = nsga2_tournament_select(population, rng, ranks, crowding_distances)

    for candidate in population:
        if (
            candidate is not first
            and chromosome_signature(candidate) != first_signature
        ):
            return first, candidate

    return first, second


__all__ = [
    "nsga2_parent_pair",
    "nsga2_tournament_select",
    "random_parent_pair",
    "tournament_select",
]
