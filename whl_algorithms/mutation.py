"""Mutation operators for chromosome search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from whl_core.chromosome import Chromosome

DEFAULT_MUTATION_PROBABILITIES = {
    "bit_flip": 0.20,
    "spacing_aware_flip": 0.20,
    "swap_segments": 0.10,
    "insert_random_aisle": 0.15,
    "remove_random_aisle": 0.15,
    "break_symmetry": 0.20,
}

UNIFORM_MUTATION_PROBABILITIES = {
    "bit_flip": 1 / 6,
    "spacing_aware_flip": 1 / 6,
    "swap_segments": 1 / 6,
    "insert_random_aisle": 1 / 6,
    "remove_random_aisle": 1 / 6,
    "break_symmetry": 1 / 6,
}

BIT_FLIP_ONLY_PROBABILITIES = {
    "bit_flip": 1.0,
    "spacing_aware_flip": 0.0,
    "swap_segments": 0.0,
    "insert_random_aisle": 0.0,
    "remove_random_aisle": 0.0,
    "break_symmetry": 0.0,
}

NO_SYMMETRY_MUTATION_PROBABILITIES = {
    "bit_flip": 0.25,
    "spacing_aware_flip": 0.25,
    "swap_segments": 0.125,
    "insert_random_aisle": 0.1875,
    "remove_random_aisle": 0.1875,
    "break_symmetry": 0.0,
}

MUTATION_OPERATOR_NAMES = tuple(DEFAULT_MUTATION_PROBABILITIES)


def _choose_axis(rng: np.random.Generator, axis: str | None = None) -> str:
    if axis is None:
        return str(rng.choice(["h", "v"]))
    if axis not in {"h", "v"}:
        raise ValueError("axis must be 'h', 'v', or None.")
    return axis


def _vector_for_axis(chromosome: Chromosome, axis: str) -> np.ndarray:
    return chromosome.h if axis == "h" else chromosome.v


def _inactive_indices(vector: np.ndarray) -> np.ndarray:
    return np.flatnonzero(vector == 0)


def _active_indices(vector: np.ndarray) -> np.ndarray:
    return np.flatnonzero(vector == 1)


def _valid_spacing_activation(
    vector: np.ndarray,
    index: int,
    spacing: int,
) -> bool:
    active_indices = _active_indices(vector)
    return bool(np.all(np.abs(active_indices - index) >= spacing))


def validate_mutation_probabilities(probabilities: Mapping[str, float]) -> None:
    """Validate a mutation probability scheme."""
    unknown_names = set(probabilities) - set(MUTATION_OPERATOR_NAMES)
    if unknown_names:
        unknown_text = ", ".join(sorted(unknown_names))
        raise ValueError(f"unknown mutation operator(s): {unknown_text}")

    missing_names = set(MUTATION_OPERATOR_NAMES) - set(probabilities)
    if missing_names:
        missing_text = ", ".join(sorted(missing_names))
        raise ValueError(f"missing mutation operator(s): {missing_text}")

    for name, probability in probabilities.items():
        if probability < 0:
            raise ValueError(f"probability for {name!r} must be non-negative.")

    total = sum(probabilities.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"mutation probabilities must sum to 1.0; got {total}.")


def mutate_bit_flip(
    chromosome: Chromosome,
    rng: np.random.Generator,
    axis: str | None = None,
) -> Chromosome:
    """Flip one random bit in ``h`` or ``v``."""
    mutated = chromosome.copy()
    chosen_axis = _choose_axis(rng, axis)
    vector = _vector_for_axis(mutated, chosen_axis)
    if len(vector) == 0:
        return mutated

    index = int(rng.integers(0, len(vector)))
    vector[index] = 1 - vector[index]
    mutated.validate()
    return mutated


def mutate_spacing_aware_flip(
    chromosome: Chromosome,
    rng: np.random.Generator,
    min_h_spacing: int = 1,
    min_v_spacing: int = 1,
    max_attempts: int = 50,
) -> Chromosome:
    """Activate one inactive bit while respecting axis-specific spacing."""
    if min_h_spacing < 1 or min_v_spacing < 1:
        raise ValueError("minimum spacing values must be at least 1.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    chosen_axis = _choose_axis(rng)
    spacing = min_h_spacing if chosen_axis == "h" else min_v_spacing
    vector = _vector_for_axis(chromosome, chosen_axis)
    candidates = [
        int(index)
        for index in _inactive_indices(vector)
        if _valid_spacing_activation(vector, int(index), spacing)
    ]

    if candidates:
        mutated = chromosome.copy()
        target = int(rng.choice(candidates))
        _vector_for_axis(mutated, chosen_axis)[target] = 1
        mutated.validate()
        return mutated

    other_axis = "v" if chosen_axis == "h" else "h"
    other_spacing = min_v_spacing if other_axis == "v" else min_h_spacing
    other_vector = _vector_for_axis(chromosome, other_axis)
    other_candidates = [
        int(index)
        for index in _inactive_indices(other_vector)
        if _valid_spacing_activation(other_vector, int(index), other_spacing)
    ]
    if other_candidates:
        mutated = chromosome.copy()
        target = int(rng.choice(other_candidates))
        _vector_for_axis(mutated, other_axis)[target] = 1
        mutated.validate()
        return mutated

    return mutate_bit_flip(chromosome, rng, axis=chosen_axis)


def mutate_swap_segments(
    chromosome: Chromosome,
    rng: np.random.Generator,
    axis: str | None = None,
) -> Chromosome:
    """Swap two same-length non-overlapping random segments on one axis."""
    mutated = chromosome.copy()
    chosen_axis = _choose_axis(rng, axis)
    vector = _vector_for_axis(mutated, chosen_axis)
    length = len(vector)
    if length < 4:
        return mutate_bit_flip(chromosome, rng, axis=chosen_axis)

    max_segment_length = max(1, length // 3)
    for _ in range(50):
        segment_length = int(rng.integers(1, max_segment_length + 1))
        first_start = int(rng.integers(0, length - segment_length + 1))
        second_start = int(rng.integers(0, length - segment_length + 1))
        first_end = first_start + segment_length
        second_end = second_start + segment_length
        if first_end <= second_start or second_end <= first_start:
            first_segment = vector[first_start:first_end].copy()
            vector[first_start:first_end] = vector[second_start:second_end]
            vector[second_start:second_end] = first_segment
            mutated.validate()
            return mutated

    return mutate_bit_flip(chromosome, rng, axis=chosen_axis)


def mutate_insert_random_aisle(
    chromosome: Chromosome,
    rng: np.random.Generator,
    axis: str | None = None,
) -> Chromosome:
    """Set one random inactive bit to active."""
    mutated = chromosome.copy()
    chosen_axis = _choose_axis(rng, axis)
    vector = _vector_for_axis(mutated, chosen_axis)
    inactive = _inactive_indices(vector)
    if len(inactive) == 0:
        mutated.validate()
        return mutated

    vector[int(rng.choice(inactive))] = 1
    mutated.validate()
    return mutated


def mutate_remove_random_aisle(
    chromosome: Chromosome,
    rng: np.random.Generator,
    axis: str | None = None,
) -> Chromosome:
    """Set one random active bit to inactive."""
    mutated = chromosome.copy()
    chosen_axis = _choose_axis(rng, axis)
    vector = _vector_for_axis(mutated, chosen_axis)
    active = _active_indices(vector)
    if len(active) == 0:
        mutated.validate()
        return mutated

    vector[int(rng.choice(active))] = 0
    mutated.validate()
    return mutated


def mutate_break_symmetry(
    chromosome: Chromosome,
    rng: np.random.Generator,
) -> Chromosome:
    """Apply a small asymmetric shift to one active bit where possible."""
    chosen_axis = _choose_axis(rng)
    source_vector = _vector_for_axis(chromosome, chosen_axis)
    active = _active_indices(source_vector)

    if len(active) > 0:
        for index in rng.permutation(active):
            direction = int(rng.choice([-1, 1]))
            for candidate in (int(index) + direction, int(index) - direction):
                if (
                    0 <= candidate < len(source_vector)
                    and source_vector[candidate] == 0
                ):
                    mutated = chromosome.copy()
                    vector = _vector_for_axis(mutated, chosen_axis)
                    vector[int(index)] = 0
                    vector[candidate] = 1
                    mutated.validate()
                    return mutated

    vector_length = len(source_vector)
    if vector_length > 0:
        mutated = chromosome.copy()
        vector = _vector_for_axis(mutated, chosen_axis)
        side = 0 if rng.random() < 0.5 else vector_length - 1
        vector[side] = 1 - vector[side]
        mutated.validate()
        return mutated

    return chromosome.copy()


def choose_mutation_operator(
    probabilities: Mapping[str, float],
    rng: np.random.Generator,
) -> str:
    """Choose a mutation operator according to a validated probability scheme."""
    validate_mutation_probabilities(probabilities)
    names = list(MUTATION_OPERATOR_NAMES)
    weights = np.array([probabilities[name] for name in names], dtype=float)
    return str(rng.choice(names, p=weights))


def mutate_chromosome_with_info(
    chromosome: Chromosome,
    rng: np.random.Generator,
    probabilities: Mapping[str, float] | None = None,
    min_h_spacing: int = 1,
    min_v_spacing: int = 1,
) -> tuple[Chromosome, dict[str, Any]]:
    """Mutate a chromosome and return basic metadata about the operator used."""
    selected_probabilities = probabilities or DEFAULT_MUTATION_PROBABILITIES
    operator = choose_mutation_operator(selected_probabilities, rng)

    if operator == "bit_flip":
        mutated = mutate_bit_flip(chromosome, rng)
    elif operator == "spacing_aware_flip":
        mutated = mutate_spacing_aware_flip(
            chromosome,
            rng,
            min_h_spacing=min_h_spacing,
            min_v_spacing=min_v_spacing,
        )
    elif operator == "swap_segments":
        mutated = mutate_swap_segments(chromosome, rng)
    elif operator == "insert_random_aisle":
        mutated = mutate_insert_random_aisle(chromosome, rng)
    elif operator == "remove_random_aisle":
        mutated = mutate_remove_random_aisle(chromosome, rng)
    elif operator == "break_symmetry":
        mutated = mutate_break_symmetry(chromosome, rng)
    else:
        raise ValueError(f"unsupported mutation operator: {operator}")

    mutated.validate(rows=len(chromosome.h), cols=len(chromosome.v))
    return mutated, {"operator": operator}


def mutate_chromosome(
    chromosome: Chromosome,
    rng: np.random.Generator,
    probabilities: Mapping[str, float] | None = None,
    min_h_spacing: int = 1,
    min_v_spacing: int = 1,
) -> Chromosome:
    """Apply one selected mutation operator and return a new chromosome."""
    mutated, _ = mutate_chromosome_with_info(
        chromosome,
        rng,
        probabilities=probabilities,
        min_h_spacing=min_h_spacing,
        min_v_spacing=min_v_spacing,
    )
    return mutated


for _probabilities in (
    DEFAULT_MUTATION_PROBABILITIES,
    UNIFORM_MUTATION_PROBABILITIES,
    BIT_FLIP_ONLY_PROBABILITIES,
    NO_SYMMETRY_MUTATION_PROBABILITIES,
):
    validate_mutation_probabilities(_probabilities)


__all__ = [
    "BIT_FLIP_ONLY_PROBABILITIES",
    "DEFAULT_MUTATION_PROBABILITIES",
    "NO_SYMMETRY_MUTATION_PROBABILITIES",
    "UNIFORM_MUTATION_PROBABILITIES",
    "choose_mutation_operator",
    "mutate_bit_flip",
    "mutate_break_symmetry",
    "mutate_chromosome",
    "mutate_chromosome_with_info",
    "mutate_insert_random_aisle",
    "mutate_remove_random_aisle",
    "mutate_spacing_aware_flip",
    "mutate_swap_segments",
    "validate_mutation_probabilities",
]
