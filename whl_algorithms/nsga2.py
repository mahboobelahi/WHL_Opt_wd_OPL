"""NSGA-II objective ranking utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

ObjectiveDirection = Literal["min", "max"]


def validate_objective_directions(directions: list[str]) -> None:
    """Validate objective direction strings."""
    if not directions:
        raise ValueError("directions must not be empty.")
    invalid = [direction for direction in directions if direction not in {"min", "max"}]
    if invalid:
        raise ValueError("objective directions must be 'min' or 'max'.")


def _as_2d_objectives(objectives: np.ndarray) -> np.ndarray:
    array = np.asarray(objectives, dtype=float)
    if array.ndim != 2:
        raise ValueError("objectives must be a 2D array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("objectives must have at least one row and one column.")
    return array


def normalize_objectives_for_minimization(
    objectives: np.ndarray,
    directions: list[str],
) -> np.ndarray:
    """Convert mixed min/max objective values to minimization form."""
    validate_objective_directions(directions)
    normalized = _as_2d_objectives(objectives).copy()
    if normalized.shape[1] != len(directions):
        raise ValueError("direction count must match objective column count.")

    for column, direction in enumerate(directions):
        if direction == "max":
            normalized[:, column] *= -1.0

    return normalized


def dominates(a: np.ndarray, b: np.ndarray, directions: list[str]) -> bool:
    """Return True if objective vector ``a`` dominates objective vector ``b``."""
    validate_objective_directions(directions)
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)
    if vector_a.shape != vector_b.shape:
        raise ValueError("objective vectors must have the same shape.")
    if vector_a.ndim != 1:
        raise ValueError("objective vectors must be 1D.")
    if len(vector_a) != len(directions):
        raise ValueError("direction count must match objective vector length.")

    no_worse = []
    strictly_better = []
    for value_a, value_b, direction in zip(vector_a, vector_b, directions, strict=True):
        if direction == "min":
            no_worse.append(value_a <= value_b)
            strictly_better.append(value_a < value_b)
        else:
            no_worse.append(value_a >= value_b)
            strictly_better.append(value_a > value_b)

    return bool(all(no_worse) and any(strictly_better))


def non_dominated_sort(
    objectives: np.ndarray,
    directions: list[str],
) -> list[list[int]]:
    """Return NSGA-II non-dominated fronts for objective rows."""
    objective_array = _as_2d_objectives(objectives)
    normalize_objectives_for_minimization(objective_array, directions)

    n_solutions = objective_array.shape[0]
    dominates_sets: list[list[int]] = [[] for _ in range(n_solutions)]
    domination_counts = [0 for _ in range(n_solutions)]
    fronts: list[list[int]] = [[]]

    for p in range(n_solutions):
        for q in range(n_solutions):
            if p == q:
                continue
            if dominates(objective_array[p], objective_array[q], directions):
                dominates_sets[p].append(q)
            elif dominates(objective_array[q], objective_array[p], directions):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    current_front_index = 0
    while current_front_index < len(fronts) and fronts[current_front_index]:
        next_front: list[int] = []
        for p in fronts[current_front_index]:
            for q in dominates_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current_front_index += 1

    return fronts


def assign_ranks(fronts: list[list[int]], n: int) -> list[int]:
    """Return rank per solution index from non-dominated fronts."""
    if n <= 0:
        raise ValueError("n must be positive.")

    ranks = [-1 for _ in range(n)]
    seen: set[int] = set()
    for rank, front in enumerate(fronts):
        for index in front:
            if index < 0 or index >= n:
                raise ValueError(f"front index {index} is outside [0, {n}).")
            if index in seen:
                raise ValueError(f"front index {index} appears more than once.")
            seen.add(index)
            ranks[index] = rank

    if len(seen) != n:
        missing = sorted(set(range(n)) - seen)
        raise ValueError(f"fronts do not cover all indices; missing {missing}.")

    return ranks


def crowding_distance(
    objectives: np.ndarray,
    front: list[int],
    directions: list[str],
) -> dict[int, float]:
    """Compute NSGA-II crowding distance for one front."""
    objective_array = normalize_objectives_for_minimization(objectives, directions)
    n_solutions = objective_array.shape[0]
    for index in front:
        if index < 0 or index >= n_solutions:
            raise ValueError(f"front index {index} is outside objective rows.")

    distances = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}

    front_array = objective_array[front]
    for column in range(objective_array.shape[1]):
        column_values = front_array[:, column]
        order = np.argsort(column_values, kind="mergesort")
        sorted_front = [front[int(position)] for position in order]
        min_value = float(column_values[order[0]])
        max_value = float(column_values[order[-1]])

        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")
        if np.isclose(max_value, min_value):
            continue

        denominator = max_value - min_value
        for order_position in range(1, len(sorted_front) - 1):
            index = sorted_front[order_position]
            if np.isinf(distances[index]):
                continue
            previous_value = float(column_values[order[order_position - 1]])
            next_value = float(column_values[order[order_position + 1]])
            distances[index] += (next_value - previous_value) / denominator

    return distances


def crowding_distances_for_fronts(
    objectives: np.ndarray,
    fronts: list[list[int]],
    directions: list[str],
) -> list[float]:
    """Return crowding distances aligned with solution index."""
    objective_array = _as_2d_objectives(objectives)
    distances = [0.0 for _ in range(objective_array.shape[0])]
    seen: set[int] = set()

    for front in fronts:
        front_distances = crowding_distance(objective_array, front, directions)
        for index, distance in front_distances.items():
            if index in seen:
                raise ValueError(f"front index {index} appears more than once.")
            seen.add(index)
            distances[index] = distance

    if len(seen) != objective_array.shape[0]:
        missing = sorted(set(range(objective_array.shape[0])) - seen)
        raise ValueError(f"fronts do not cover all indices; missing {missing}.")

    return distances


def sort_by_rank_and_crowding(
    indices: list[int],
    ranks: list[int],
    crowding: list[float],
) -> list[int]:
    """Sort indices by lower rank, then higher crowding distance."""
    return sorted(indices, key=lambda index: (ranks[index], -crowding[index]))


def environmental_selection(
    population: list,
    objectives: np.ndarray,
    directions: list[str],
    target_size: int,
) -> tuple[list, np.ndarray, list[int], list[float]]:
    """Select survivors by non-dominated rank and crowding distance."""
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    if target_size > len(population):
        raise ValueError("target_size must not exceed population size.")

    objective_array = _as_2d_objectives(objectives)
    if len(population) != objective_array.shape[0]:
        raise ValueError("population length must match objective rows.")

    fronts = non_dominated_sort(objective_array, directions)
    ranks = assign_ranks(fronts, len(population))
    crowding = crowding_distances_for_fronts(objective_array, fronts, directions)

    selected_indices: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= target_size:
            selected_indices.extend(front)
        else:
            remaining = target_size - len(selected_indices)
            sorted_front = sort_by_rank_and_crowding(front, ranks, crowding)
            selected_indices.extend(sorted_front[:remaining])
            break

    selected_population = [population[index] for index in selected_indices]
    selected_objectives = objective_array[selected_indices].copy()
    selected_ranks = [ranks[index] for index in selected_indices]
    selected_crowding = [crowding[index] for index in selected_indices]
    return selected_population, selected_objectives, selected_ranks, selected_crowding


__all__ = [
    "ObjectiveDirection",
    "assign_ranks",
    "crowding_distance",
    "crowding_distances_for_fronts",
    "dominates",
    "environmental_selection",
    "non_dominated_sort",
    "normalize_objectives_for_minimization",
    "sort_by_rank_and_crowding",
    "validate_objective_directions",
]
