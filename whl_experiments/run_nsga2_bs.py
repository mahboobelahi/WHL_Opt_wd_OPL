"""Minimal NSGA-II plus Beam Search runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from whl_algorithms.beam_node import BeamNode, layout_signature
from whl_algorithms.beam_scoring import DEFAULT_BEAM_WEIGHTS, sample_beam_weights
from whl_algorithms.beam_search import BeamSearchConfig, run_beam_search
from whl_algorithms.beam_sorting import load_sorting_rules, sample_sorting_rule
from whl_algorithms.carving import (
    find_feasible_global_horizontal_starts,
    find_feasible_global_vertical_starts,
)
from whl_algorithms.crossover import make_offspring_pair
from whl_algorithms.mutation import (
    DEFAULT_MUTATION_PROBABILITIES,
    NO_SYMMETRY_MUTATION_PROBABILITIES,
    UNIFORM_MUTATION_PROBABILITIES,
    mutate_chromosome_with_info,
)
from whl_algorithms.nsga2 import (
    assign_ranks,
    crowding_distances_for_fronts,
    non_dominated_sort,
    sort_by_rank_and_crowding,
)
from whl_algorithms.parameter_policy import min_fragment_size
from whl_algorithms.population import (
    adaptive_spacing,
    compute_aspect_ratio_betas,
    initialize_population,
)
from whl_algorithms.selection import nsga2_parent_pair
from whl_core.chromosome import Chromosome
from whl_core.feasibility import (
    access_anchor_mask_from_grid_and_masks,
    check_layout_feasible,
    oriented_aisle_thickness_violations,
)
from whl_core.layout_io import fixed_aisle_mask_from_masks, load_mask, mask_to_grid
from whl_core.paths import FIGURES_DIR, MASK_DIR, PROCESSED_RESULTS_DIR
from whl_core.registry import load_layouts
from whl_core.connectivity import access_anchor_connectivity_report
from whl_core.scoring import score_layout

DEFAULT_NSGA2_BS_CSV = PROCESSED_RESULTS_DIR / "nsga2_bs_summary.csv"
DEFAULT_NSGA2_FIGURE_DIR = FIGURES_DIR / "nsga2_bs"
SORTING_RULE_MODES = ("sampled_pool", "fixed")
ADAPTIVE_WEIGHT_MODES = ("adaptive", "fixed")
DEFAULT_SORTING_RULE_MODE = "sampled_pool"
DEFAULT_SORTING_RULE = "PF_LS_RP"
DEFAULT_ADAPTIVE_WEIGHT_MODE = "adaptive"
DEFAULT_INITIALIZATION_MODE = "feasible_start_adaptive_spacing"
RANDOM_FEASIBLE_INITIALIZATION_MODE = "random_feasible_start_no_adaptive_spacing"
INITIALIZATION_SPACING_MODES = (
    DEFAULT_INITIALIZATION_MODE,
    RANDOM_FEASIBLE_INITIALIZATION_MODE,
)
MUTATION_MODES = ("weighted", "uniform", "weighted_no_symmetry_breaking")
DEFAULT_MUTATION_MODE = "weighted"
REFERENCE_ONLY_LAYOUT_FILENAMES = {f"AT_{index}.npz" for index in range(1, 14)}

OBJECTIVE_KEYS = (
    "interior_storage",
    "retrieval_penalty",
    "pick_faces",
)
OBJECTIVE_DIRECTIONS = ["min", "min", "max"]

CSV_COLUMNS = [
    "run_id",
    "instance",
    "seed",
    "generation",
    "candidate_id",
    "parent_chromosome_id",
    "depth",
    "trace",
    "status",
    "rank",
    "crowding_distance",
    "storage_total",
    "pick_faces",
    "interior_storage",
    "retrieval_penalty",
    "door_connectivity_index",
    "access_anchor_connectivity_index",
    "has_door_connected_aisle",
    "has_access_anchor_connected_aisle",
    "aisle_components",
    "anchor_connected_components",
    "unanchored_aisle_components",
    "single_aisle_component",
    "access_network_components",
    "aisle_access_network_components",
    "unreachable_aisle_components",
    "unreachable_aisle_cells",
    "has_access_anchor_reachable_aisle_network",
    "exact_width_ok",
    "exact_width_violation_count",
    "chromosome_h_active_count",
    "chromosome_v_active_count",
    "active_h_count",
    "active_v_count",
    "chromosome_index",
    "sorting_rule",
    "sorting_rule_mode",
    "uses_scalar_score",
    "rho",
    "beam_w1",
    "beam_w2",
    "beam_lambda",
    "adaptive_weight_mode",
    "mutation_operator",
    "initialization_mode",
    "initialization_spacing_mode",
    "adaptive_spacing_used",
    "feasible_h_start_count",
    "feasible_v_start_count",
    "h_active_starts",
    "v_active_starts",
    "chromosome_signature",
    "layout_signature",
]


@dataclass
class LayoutCandidate:
    """Decoded layout-level record with its chromosome state."""

    node: BeamNode
    chromosome: Chromosome
    parent_chromosome_id: int
    metrics: dict[str, Any]
    feasibility: dict[str, Any]
    exact_width_violations: list[str]
    decode_metadata: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    crowding_distance: float | None = None
    selected: bool = False
    candidate_id: int = 0

    @property
    def is_feasible(self) -> bool:
        return bool(self.feasibility.get("is_feasible", False))


def _scalar_int(value: Any, default: int = 1) -> int:
    if value is None:
        return int(default)
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(value)


def _scalar_text(value: Any, default: str = "layout") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def _signature_digest(signature: bytes) -> str:
    return hashlib.sha1(signature).hexdigest()


def chromosome_signature_text(chromosome: Chromosome) -> str:
    h_indices = ",".join(str(index) for index in chromosome.active_h_indices())
    v_indices = ",".join(str(index) for index in chromosome.active_v_indices())
    return f"H[{h_indices}]|V[{v_indices}]"


def chromosome_active_text(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def is_reference_only_layout(path: Path) -> bool:
    """Return whether a mask is a plotting/editor reference layout."""
    return path.name in REFERENCE_ONLY_LAYOUT_FILENAMES


def discover_instance_masks(
    limit: int | None = None,
    *,
    include_reference: bool = False,
) -> list[Path]:
    """Return available local optimization mask files."""
    masks = sorted(path for path in MASK_DIR.glob("*.npz") if path.is_file())
    if not include_reference:
        masks = [path for path in masks if not is_reference_only_layout(path)]
    if limit is None:
        return masks
    if int(limit) < 0:
        raise ValueError("limit must be non-negative or None.")
    return masks[: int(limit)]


def resolve_instance_paths(
    instance: str | None = None,
    limit_instances: int | None = 1,
) -> list[Path]:
    """Resolve an optional filename, registry ID, or path into mask paths."""
    if instance:
        raw = Path(instance)
        if raw.exists():
            return [raw]

        registry = load_layouts()
        filename = registry.get(int(instance)) if str(instance).isdigit() else None
        if filename is None:
            filename = instance
        if not filename.endswith(".npz"):
            filename = f"{filename}.npz"
        candidate = MASK_DIR / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Instance mask not found: {candidate}")
        return [candidate]

    masks = discover_instance_masks(limit=limit_instances)
    if not masks:
        raise FileNotFoundError(f"No .npz masks found in {MASK_DIR}")
    return masks


def _sample_starts(
    starts: list[int],
    rng: np.random.Generator,
    max_active: int,
    allow_empty: bool,
) -> list[int]:
    if not starts:
        return []
    upper = min(max_active, len(starts))
    lower = 0 if allow_empty else 1
    count = int(rng.integers(lower, upper + 1))
    if count == 0:
        return []
    selected = rng.choice(np.asarray(starts, dtype=int), size=count, replace=False)
    return sorted(int(value) for value in selected.tolist())


def _tag_chromosome(
    chromosome: Chromosome,
    *,
    initialization_mode: str,
    initialization_spacing_mode: str | None = None,
    adaptive_spacing_used: bool | None = None,
    h_spacing: int | None = None,
    v_spacing: int | None = None,
    h_feasible_count: int | None = None,
    v_feasible_count: int | None = None,
) -> Chromosome:
    chromosome.initialization_mode = initialization_mode
    chromosome.initialization_spacing_mode = initialization_spacing_mode or initialization_mode
    chromosome.adaptive_spacing_used = adaptive_spacing_used
    chromosome.h_spacing = h_spacing
    chromosome.v_spacing = v_spacing
    chromosome.h_feasible_count = h_feasible_count
    chromosome.v_feasible_count = v_feasible_count
    chromosome.feasible_h_start_count = h_feasible_count
    chromosome.feasible_v_start_count = v_feasible_count
    chromosome.active_h_count = len(chromosome.active_h_indices())
    chromosome.active_v_count = len(chromosome.active_v_indices())
    return chromosome


def mutation_probabilities_for_mode(mutation_mode: str) -> dict[str, float]:
    """Return the mutation probability scheme for one ablation mode."""
    if mutation_mode == "weighted":
        return dict(DEFAULT_MUTATION_PROBABILITIES)
    if mutation_mode == "uniform":
        return dict(UNIFORM_MUTATION_PROBABILITIES)
    if mutation_mode == "weighted_no_symmetry_breaking":
        return dict(NO_SYMMETRY_MUTATION_PROBABILITIES)
    raise ValueError(f"mutation_mode must be one of {MUTATION_MODES}.")


def symmetry_breaking_enabled_for_mode(mutation_mode: str) -> bool:
    probabilities = mutation_probabilities_for_mode(mutation_mode)
    return bool(probabilities.get("break_symmetry", 0.0) > 0.0)


def _chromosome_initialization_mode(chromosome: Chromosome) -> str:
    return str(getattr(chromosome, "initialization_mode", "offspring"))


def _sample_feasible_starts_with_spacing(
    starts: list[int],
    spacing: int,
    rng: np.random.Generator,
    *,
    max_active: int = 3,
    allow_empty: bool = False,
) -> tuple[list[int], bool]:
    """Sample feasible starts while respecting adaptive spacing when possible."""
    if not starts:
        return [], False
    if spacing < 1:
        raise ValueError("spacing must be at least 1.")
    if max_active <= 0:
        raise ValueError("max_active must be positive.")

    selected: list[int] = []
    selection_probability = min(0.75, max(0.20, 1.0 / (spacing + 1)))
    for value in rng.permutation(np.asarray(starts, dtype=int)):
        candidate = int(value)
        if all(abs(candidate - existing) >= spacing for existing in selected):
            if rng.random() <= selection_probability:
                selected.append(candidate)
                if len(selected) >= max_active:
                    break

    repaired = False
    if not selected and not allow_empty:
        selected.append(int(rng.choice(np.asarray(starts, dtype=int))))
        repaired = True
    return sorted(selected), repaired


def _sample_random_feasible_starts(
    starts: list[int],
    rng: np.random.Generator,
    *,
    max_active: int = 3,
    allow_empty: bool = False,
) -> list[int]:
    """Sample feasible starts without chromosome-index adaptive spacing."""
    if not starts:
        return []
    if max_active <= 0:
        raise ValueError("max_active must be positive.")

    max_count = min(int(max_active), len(starts))
    min_count = 0 if allow_empty else 1
    count = int(rng.integers(min_count, max_count + 1))
    if count == 0:
        return []
    selected = rng.choice(np.asarray(starts, dtype=int), size=count, replace=False)
    return sorted(int(value) for value in selected)


def create_initial_population_for_grid(
    grid: np.ndarray,
    aisle_width: int,
    population_size: int,
    seed: int = 1,
    initialization_spacing_mode: str = DEFAULT_INITIALIZATION_MODE,
) -> list[Chromosome]:
    """Create a deterministic chromosome population biased to feasible starts."""
    if initialization_spacing_mode not in INITIALIZATION_SPACING_MODES:
        raise ValueError(
            f"initialization_spacing_mode must be one of {INITIALIZATION_SPACING_MODES}."
        )
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if population_size <= 0:
        raise ValueError("population_size must be positive.")

    rows, cols = layout.shape
    h_starts = find_feasible_global_horizontal_starts(layout, int(aisle_width))
    v_starts = find_feasible_global_vertical_starts(layout, int(aisle_width))
    beta_h, beta_v = compute_aspect_ratio_betas(rows, cols)

    if not h_starts and not v_starts:
        return [
            _tag_chromosome(
                chromosome,
                initialization_mode="fallback_adaptive_spacing",
                initialization_spacing_mode=initialization_spacing_mode,
                adaptive_spacing_used=False,
                h_feasible_count=0,
                v_feasible_count=0,
            )
            for chromosome in initialize_population(rows, cols, population_size, seed=seed)
        ]

    rng = np.random.default_rng(seed)
    population: list[Chromosome] = []
    signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    max_retries = 25
    for index in range(population_size):
        h_spacing = adaptive_spacing(index, population_size, rows, beta_h)
        v_spacing = adaptive_spacing(index, population_size, cols, beta_v)
        accepted: Chromosome | None = None
        fallback: Chromosome | None = None
        for _ in range(max_retries):
            allow_empty_h = bool(v_starts)
            allow_empty_v = bool(h_starts)
            if initialization_spacing_mode == RANDOM_FEASIBLE_INITIALIZATION_MODE:
                h_indices = _sample_random_feasible_starts(
                    h_starts,
                    rng,
                    max_active=3,
                    allow_empty=allow_empty_h,
                )
                v_indices = _sample_random_feasible_starts(
                    v_starts,
                    rng,
                    max_active=3,
                    allow_empty=allow_empty_v,
                )
                h_repaired = False
                v_repaired = False
                init_mode = RANDOM_FEASIBLE_INITIALIZATION_MODE
                tag_h_spacing = None
                tag_v_spacing = None
                adaptive_spacing_used = False
            else:
                h_indices, h_repaired = _sample_feasible_starts_with_spacing(
                    h_starts,
                    h_spacing,
                    rng,
                    max_active=3,
                    allow_empty=allow_empty_h,
                )
                v_indices, v_repaired = _sample_feasible_starts_with_spacing(
                    v_starts,
                    v_spacing,
                    rng,
                    max_active=3,
                    allow_empty=allow_empty_v,
                )
                init_mode = DEFAULT_INITIALIZATION_MODE
                tag_h_spacing = h_spacing
                tag_v_spacing = v_spacing
                adaptive_spacing_used = True
            if not h_indices and not v_indices:
                if h_starts and (not v_starts or index % 2 == 0):
                    h_indices = [int(rng.choice(h_starts))]
                elif v_starts:
                    v_indices = [int(rng.choice(v_starts))]
                init_mode = (
                    RANDOM_FEASIBLE_INITIALIZATION_MODE
                    if initialization_spacing_mode == RANDOM_FEASIBLE_INITIALIZATION_MODE
                    else "fallback_random_or_repair"
                )
                adaptive_spacing_used = initialization_spacing_mode != RANDOM_FEASIBLE_INITIALIZATION_MODE
            elif h_repaired or v_repaired:
                init_mode = "fallback_random_or_repair"

            chromosome = Chromosome.from_indices(rows, cols, h_indices, v_indices)
            chromosome = _tag_chromosome(
                chromosome,
                initialization_mode=init_mode,
                initialization_spacing_mode=initialization_spacing_mode,
                adaptive_spacing_used=adaptive_spacing_used,
                h_spacing=tag_h_spacing,
                v_spacing=tag_v_spacing,
                h_feasible_count=len(h_starts),
                v_feasible_count=len(v_starts),
            )
            fallback = chromosome
            signature = chromosome.as_tuple()
            if signature not in signatures:
                accepted = chromosome
                break
        if accepted is None:
            if fallback is None:
                raise RuntimeError("failed to sample a chromosome.")
            accepted = fallback
        accepted.validate(rows=rows, cols=cols)
        population.append(accepted)
        signatures.add(accepted.as_tuple())
    return population


def _node_chromosome_state(node: BeamNode, fallback: Chromosome) -> Chromosome:
    h = node.remaining_h if node.remaining_h is not None else fallback.h
    v = node.remaining_v if node.remaining_v is not None else fallback.v
    chromosome = Chromosome(h=np.asarray(h).copy(), v=np.asarray(v).copy())
    chromosome.validate(rows=len(fallback.h), cols=len(fallback.v))
    return chromosome


def sorting_rule_uses_scalar_score(
    sorting_rule: str,
    sorting_rules: dict[str, list[tuple[str, str]]] | None = None,
) -> bool:
    rules = load_sorting_rules() if sorting_rules is None else sorting_rules
    if sorting_rule not in rules:
        raise KeyError(f"unknown sorting rule: {sorting_rule}")
    return any(metric == "scalar_score" for metric, _ in rules[sorting_rule])


def choose_sorting_rule_for_decode(
    *,
    sorting_rule_mode: str,
    fixed_sorting_rule: str,
    sorting_rules: dict[str, list[tuple[str, str]]],
    rng: np.random.Generator,
) -> str:
    if sorting_rule_mode not in SORTING_RULE_MODES:
        raise ValueError(f"sorting_rule_mode must be one of {SORTING_RULE_MODES}.")
    if sorting_rule_mode == "fixed":
        if fixed_sorting_rule not in sorting_rules:
            raise KeyError(f"unknown sorting rule: {fixed_sorting_rule}")
        return fixed_sorting_rule
    return sample_sorting_rule(sorting_rules, rng)


def beam_weights_for_decode(
    *,
    generation: int,
    total_generations: int,
    chromosome_index: int,
    seed: int,
    adaptive_weight_mode: str,
    fixed_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    if adaptive_weight_mode not in ADAPTIVE_WEIGHT_MODES:
        raise ValueError(
            f"adaptive_weight_mode must be one of {ADAPTIVE_WEIGHT_MODES}."
        )
    if total_generations <= 0:
        raise ValueError("total_generations must be positive.")
    rho = float((int(generation) + 1) / int(total_generations))
    if adaptive_weight_mode == "fixed":
        return dict(fixed_weights or DEFAULT_BEAM_WEIGHTS), rho
    rng_seed = (
        int(seed) * 1_000_003
        + int(generation) * 10_007
        + int(chromosome_index) * 101
    )
    rng = np.random.default_rng(rng_seed)
    return sample_beam_weights(rho, rng), rho


def decode_chromosome_with_beam_search(
    chromosome: Chromosome,
    grid: np.ndarray,
    aisle_width: int,
    beam_width: int,
    max_depth: int,
    seed: int = 1,
    sorting_rule: str = DEFAULT_SORTING_RULE,
    sorting_rules: dict[str, list[tuple[str, str]]] | None = None,
    weights: dict[str, float] | None = None,
) -> list[BeamNode]:
    """Decode one chromosome through the current Beam Search decoder."""
    config = BeamSearchConfig(
        aisle_width=int(aisle_width),
        beam_width=int(beam_width),
        max_depth=int(max_depth),
        min_fragment_size=min_fragment_size(int(aisle_width)),
        sorting_rule=sorting_rule,
        allowed_sorting_rules=sorted(sorting_rules) if sorting_rules is not None else None,
    )
    rng = np.random.default_rng(seed)
    return run_beam_search(
        chromosome,
        grid,
        config,
        rng=rng,
        sorting_rules=sorting_rules,
        weights=weights,
    )


def extract_objective_metrics(node: BeamNode) -> dict[str, float]:
    """Return objective values used by the NSGA-II adapter."""
    metrics = dict(node.metrics or {})
    metrics.update(score_layout(node.layout))
    return {key: float(metrics[key]) for key in OBJECTIVE_KEYS}


def objective_array(candidates: list[LayoutCandidate]) -> np.ndarray:
    """Return objective matrix aligned with candidate order."""
    return np.asarray(
        [[candidate.metrics[key] for key in OBJECTIVE_KEYS] for candidate in candidates],
        dtype=float,
    )


def _add_profile_time(profile_times: dict[str, float] | None, key: str, elapsed: float) -> None:
    if profile_times is None:
        return
    profile_times[key] = float(profile_times.get(key, 0.0)) + max(0.0, float(elapsed))


def build_layout_candidates(
    population: list[Chromosome],
    grid: np.ndarray,
    masks: dict[str, Any],
    aisle_width: int,
    beam_width: int,
    max_depth: int,
    seed: int,
    generation: int = 0,
    total_generations: int = 1,
    sorting_rule_mode: str = DEFAULT_SORTING_RULE_MODE,
    sorting_rule: str = DEFAULT_SORTING_RULE,
    adaptive_weight_mode: str = DEFAULT_ADAPTIVE_WEIGHT_MODE,
    fixed_weights: dict[str, float] | None = None,
    sorting_rules: dict[str, list[tuple[str, str]]] | None = None,
    profile_times: dict[str, float] | None = None,
) -> tuple[list[LayoutCandidate], int]:
    """Decode a population and return unique layout records."""
    records: list[LayoutCandidate] = []
    seen_layouts: set[bytes] = set()
    decoded_count = 0
    fixed_aisle_mask = fixed_aisle_mask_from_masks(masks)
    selected_sorting_rules = load_sorting_rules() if sorting_rules is None else sorting_rules

    for chromosome_index, chromosome in enumerate(population, start=1):
        decode_rng = np.random.default_rng(
            int(seed) * 1_000_003 + int(generation) * 10_007 + chromosome_index
        )
        selected_sorting_rule = choose_sorting_rule_for_decode(
            sorting_rule_mode=sorting_rule_mode,
            fixed_sorting_rule=sorting_rule,
            sorting_rules=selected_sorting_rules,
            rng=decode_rng,
        )
        uses_scalar_score = sorting_rule_uses_scalar_score(
            selected_sorting_rule,
            selected_sorting_rules,
        )
        beam_weights, rho = beam_weights_for_decode(
            generation=generation,
            total_generations=total_generations,
            chromosome_index=chromosome_index,
            seed=seed,
            adaptive_weight_mode=adaptive_weight_mode,
            fixed_weights=fixed_weights,
        )
        weights_for_sorting = beam_weights if uses_scalar_score else None
        decode_metadata = {
            "chromosome_index": chromosome_index,
            "sorting_rule": selected_sorting_rule,
            "sorting_rule_mode": sorting_rule_mode,
            "uses_scalar_score": bool(uses_scalar_score),
            "rho": rho,
            "beam_w1": beam_weights["w1"],
            "beam_w2": beam_weights["w2"],
            "beam_lambda": beam_weights["lambda"],
            "adaptive_weight_mode": adaptive_weight_mode,
            "mutation_operator": getattr(chromosome, "mutation_operator", ""),
            "initialization_mode": _chromosome_initialization_mode(chromosome),
            "initialization_spacing_mode": getattr(
                chromosome,
                "initialization_spacing_mode",
                _chromosome_initialization_mode(chromosome),
            ),
            "adaptive_spacing_used": getattr(chromosome, "adaptive_spacing_used", ""),
            "active_h_count": getattr(chromosome, "active_h_count", ""),
            "active_v_count": getattr(chromosome, "active_v_count", ""),
            "feasible_h_start_count": getattr(chromosome, "feasible_h_start_count", ""),
            "feasible_v_start_count": getattr(chromosome, "feasible_v_start_count", ""),
            "h_active_starts": chromosome_active_text(chromosome.active_h_indices()),
            "v_active_starts": chromosome_active_text(chromosome.active_v_indices()),
        }
        beam_started = time.perf_counter()
        nodes = decode_chromosome_with_beam_search(
            chromosome,
            grid,
            aisle_width,
            beam_width,
            max_depth,
            seed=seed + chromosome_index,
            sorting_rule=selected_sorting_rule,
            sorting_rules=selected_sorting_rules,
            weights=weights_for_sorting,
        )
        _add_profile_time(
            profile_times,
            "beam_expansion_time_seconds",
            time.perf_counter() - beam_started,
        )
        decoded_count += len(nodes)

        for node in nodes:
            signature = layout_signature(node.layout)
            if signature in seen_layouts:
                continue
            seen_layouts.add(signature)

            objective_started = time.perf_counter()
            metrics = dict(node.metrics or {})
            metrics.update(score_layout(node.layout))
            access_anchor_mask = access_anchor_mask_from_grid_and_masks(
                node.layout,
                masks=masks,
            )
            metrics.update(
                access_anchor_connectivity_report(
                    node.layout,
                    access_anchor_mask=access_anchor_mask,
                )
            )
            _add_profile_time(
                profile_times,
                "objective_evaluation_time_seconds",
                time.perf_counter() - objective_started,
            )
            feasibility_started = time.perf_counter()
            feasibility = check_layout_feasible(
                node.layout,
                masks=masks,
                aisle_width=int(aisle_width),
                require_access_anchor_connected=True,
                require_single_aisle_component=True,
                enforce_aisle_width=True,
                enforce_exact_aisle_width=False,
                fixed_aisle_mask=fixed_aisle_mask,
            )
            exact_width_violations = oriented_aisle_thickness_violations(
                node.layout,
                int(aisle_width),
                exact=True,
                fixed_aisle_mask=fixed_aisle_mask,
            )
            _add_profile_time(
                profile_times,
                "feasibility_filter_time_seconds",
                time.perf_counter() - feasibility_started,
            )
            records.append(
                LayoutCandidate(
                    node=node,
                    chromosome=_node_chromosome_state(node, chromosome),
                    parent_chromosome_id=chromosome_index,
                    metrics=metrics,
                    feasibility=feasibility,
                    exact_width_violations=exact_width_violations,
                    decode_metadata=dict(decode_metadata),
                )
            )

    for candidate_id, record in enumerate(records, start=1):
        record.candidate_id = candidate_id
    return records, decoded_count


def select_nsga2_survivors(
    candidates: list[LayoutCandidate],
    population_size: int,
) -> list[LayoutCandidate]:
    """Assign NSGA-II rank/crowding and select feasible layout survivors."""
    feasible = [candidate for candidate in candidates if candidate.is_feasible]
    if not feasible:
        return []

    objectives = objective_array(feasible)
    fronts = non_dominated_sort(objectives, OBJECTIVE_DIRECTIONS)
    ranks = assign_ranks(fronts, len(feasible))
    crowding = crowding_distances_for_fronts(objectives, fronts, OBJECTIVE_DIRECTIONS)

    for index, candidate in enumerate(feasible):
        candidate.rank = ranks[index]
        candidate.crowding_distance = crowding[index]

    selected_indices: list[int] = []
    target_size = min(int(population_size), len(feasible))
    for front in fronts:
        if len(selected_indices) + len(front) <= target_size:
            selected_indices.extend(front)
        else:
            remaining = target_size - len(selected_indices)
            selected_indices.extend(
                sort_by_rank_and_crowding(front, ranks, crowding)[:remaining]
            )
            break

    selected = [feasible[index] for index in selected_indices]
    for candidate in selected:
        candidate.selected = True
    return selected


def make_next_generation(
    survivors: list[LayoutCandidate],
    fallback_population: list[Chromosome],
    population_size: int,
    rng: np.random.Generator,
    aisle_width: int,
    mutation_mode: str = DEFAULT_MUTATION_MODE,
) -> list[Chromosome]:
    """Create the next chromosome generation from selected layout survivors."""
    mutation_probabilities = mutation_probabilities_for_mode(mutation_mode)
    if survivors:
        parent_pool = [candidate.chromosome for candidate in survivors]
        ranks = [int(candidate.rank or 0) for candidate in survivors]
        crowding = [
            float(candidate.crowding_distance)
            if candidate.crowding_distance is not None
            else 0.0
            for candidate in survivors
        ]
    else:
        parent_pool = [chromosome.copy() for chromosome in fallback_population]
        ranks = [0 for _ in parent_pool]
        crowding = [float("inf") for _ in parent_pool]

    if not parent_pool:
        return []

    next_population: list[Chromosome] = []
    allow_same = len(parent_pool) < 2
    while len(next_population) < population_size:
        parent_a, parent_b = nsga2_parent_pair(
            parent_pool,
            rng,
            ranks=ranks,
            crowding_distances=crowding,
            allow_same=allow_same,
        )
        child_a, child_b = make_offspring_pair(
            parent_a,
            parent_b,
            rng,
            crossover_prob=1.0,
        )
        for child in (child_a, child_b):
            mutated, mutation_info = mutate_chromosome_with_info(
                child,
                rng,
                probabilities=mutation_probabilities,
                min_h_spacing=max(1, int(aisle_width)),
                min_v_spacing=max(1, int(aisle_width)),
            )
            _tag_chromosome(mutated, initialization_mode="offspring")
            mutated.mutation_operator = mutation_info["operator"]
            next_population.append(mutated)
            if len(next_population) >= population_size:
                break

    return next_population


def _candidate_status(candidate: LayoutCandidate) -> str:
    if not candidate.is_feasible:
        return "infeasible"
    return "selected" if candidate.selected else "evaluated"


def _candidate_decode_value(candidate: LayoutCandidate, key: str, default: Any = "") -> Any:
    return candidate.decode_metadata.get(key, default)


def candidate_to_csv_row(
    candidate: LayoutCandidate,
    run_id: str,
    instance_name: str,
    seed: int,
    generation: int,
) -> dict[str, Any]:
    """Convert a layout candidate to one CSV row."""
    h_count, v_count = candidate.chromosome.active_count()
    h_starts = chromosome_active_text(candidate.chromosome.active_h_indices())
    v_starts = chromosome_active_text(candidate.chromosome.active_v_indices())
    return {
        "run_id": run_id,
        "instance": instance_name,
        "seed": int(seed),
        "generation": int(generation),
        "candidate_id": int(candidate.candidate_id),
        "parent_chromosome_id": int(candidate.parent_chromosome_id),
        "depth": int(candidate.node.depth),
        "trace": " > ".join(str(item) for item in candidate.node.trace),
        "status": _candidate_status(candidate),
        "rank": "" if candidate.rank is None else int(candidate.rank),
        "crowding_distance": ""
        if candidate.crowding_distance is None
        else float(candidate.crowding_distance),
        "storage_total": candidate.metrics.get("storage_total", ""),
        "pick_faces": candidate.metrics.get("pick_faces", ""),
        "interior_storage": candidate.metrics.get("interior_storage", ""),
        "retrieval_penalty": candidate.metrics.get("retrieval_penalty", ""),
        "door_connectivity_index": candidate.metrics.get("door_connectivity_index", ""),
        "access_anchor_connectivity_index": candidate.metrics.get(
            "access_anchor_connectivity_index",
            "",
        ),
        "has_door_connected_aisle": candidate.metrics.get("has_door_connected_aisle", ""),
        "has_access_anchor_connected_aisle": candidate.metrics.get(
            "has_access_anchor_connected_aisle",
            "",
        ),
        "aisle_components": candidate.metrics.get("aisle_components", ""),
        "anchor_connected_components": candidate.metrics.get(
            "anchor_connected_components",
            "",
        ),
        "unanchored_aisle_components": candidate.metrics.get(
            "unanchored_aisle_components",
            "",
        ),
        "single_aisle_component": candidate.metrics.get("single_aisle_component", ""),
        "access_network_components": candidate.metrics.get(
            "access_network_components",
            "",
        ),
        "aisle_access_network_components": candidate.metrics.get(
            "aisle_access_network_components",
            "",
        ),
        "unreachable_aisle_components": candidate.metrics.get(
            "unreachable_aisle_components",
            "",
        ),
        "unreachable_aisle_cells": candidate.metrics.get(
            "unreachable_aisle_cells",
            "",
        ),
        "has_access_anchor_reachable_aisle_network": candidate.metrics.get(
            "has_access_anchor_reachable_aisle_network",
            "",
        ),
        "exact_width_ok": not candidate.exact_width_violations,
        "exact_width_violation_count": len(candidate.exact_width_violations),
        "chromosome_h_active_count": h_count,
        "chromosome_v_active_count": v_count,
        "active_h_count": h_count,
        "active_v_count": v_count,
        "chromosome_index": _candidate_decode_value(candidate, "chromosome_index", ""),
        "sorting_rule": _candidate_decode_value(candidate, "sorting_rule", ""),
        "sorting_rule_mode": _candidate_decode_value(candidate, "sorting_rule_mode", ""),
        "uses_scalar_score": _candidate_decode_value(candidate, "uses_scalar_score", ""),
        "rho": _candidate_decode_value(candidate, "rho", ""),
        "beam_w1": _candidate_decode_value(candidate, "beam_w1", ""),
        "beam_w2": _candidate_decode_value(candidate, "beam_w2", ""),
        "beam_lambda": _candidate_decode_value(candidate, "beam_lambda", ""),
        "adaptive_weight_mode": _candidate_decode_value(
            candidate,
            "adaptive_weight_mode",
            "",
        ),
        "mutation_operator": _candidate_decode_value(candidate, "mutation_operator", ""),
        "initialization_mode": _candidate_decode_value(
            candidate,
            "initialization_mode",
            _chromosome_initialization_mode(candidate.chromosome),
        ),
        "initialization_spacing_mode": _candidate_decode_value(
            candidate,
            "initialization_spacing_mode",
            getattr(candidate.chromosome, "initialization_spacing_mode", ""),
        ),
        "adaptive_spacing_used": _candidate_decode_value(
            candidate,
            "adaptive_spacing_used",
            getattr(candidate.chromosome, "adaptive_spacing_used", ""),
        ),
        "feasible_h_start_count": _candidate_decode_value(
            candidate,
            "feasible_h_start_count",
            getattr(candidate.chromosome, "feasible_h_start_count", ""),
        ),
        "feasible_v_start_count": _candidate_decode_value(
            candidate,
            "feasible_v_start_count",
            getattr(candidate.chromosome, "feasible_v_start_count", ""),
        ),
        "h_active_starts": _candidate_decode_value(candidate, "h_active_starts", h_starts),
        "v_active_starts": _candidate_decode_value(candidate, "v_active_starts", v_starts),
        "chromosome_signature": chromosome_signature_text(candidate.chromosome),
        "layout_signature": _signature_digest(layout_signature(candidate.node.layout)),
    }


def _empty_generation_row(
    run_id: str,
    instance_name: str,
    seed: int,
    generation: int,
    status: str,
) -> dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "instance": instance_name,
            "seed": int(seed),
            "generation": int(generation),
            "candidate_id": -1,
            "status": status,
        }
    )
    return row


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return output_csv
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_csv.with_name(
            f"{output_csv.stem}_{timestamp}{output_csv.suffix}"
        )
        with fallback.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"CSV target locked: {output_csv}. "
            f"Saved this run to fallback CSV: {fallback}"
        )
        return fallback


def _save_selected_figures(
    selected: list[LayoutCandidate],
    figure_dir: Path,
    instance_name: str,
    seed: int,
    generation: int,
) -> None:
    if not selected:
        return
    from whl_experiments.beam_decode_preview import _save_candidate_figure

    for index, candidate in enumerate(selected, start=1):
        output_path = figure_dir / (
            f"{instance_name}_seed_{seed}_gen_{generation:03d}_selected_{index:03d}.png"
        )
        title = f"{instance_name} | seed={seed} | gen={generation} | selected={index}"
        _save_candidate_figure(candidate.node, output_path, title)


@dataclass
class Nsga2BsResult:
    """Summary returned by ``run_nsga2_bs``."""

    csv_path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    generation_summaries: list[dict[str, Any]] = field(default_factory=list)


def run_nsga2_bs(
    instance: str | None = None,
    limit_instances: int | None = 1,
    seed: int = 1,
    population_size: int = 6,
    generations: int = 3,
    beam_width: int = 3,
    max_depth: int = 8,
    output_csv: Path | str = DEFAULT_NSGA2_BS_CSV,
    save_figures: bool = True,
    figure_dir: Path | str = DEFAULT_NSGA2_FIGURE_DIR,
) -> Nsga2BsResult:
    """Run a small chromosome-level NSGA-II validation loop around Beam Search."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    if generations <= 0:
        raise ValueError("generations must be positive.")

    output_path = Path(output_csv)
    image_dir = Path(figure_dir)
    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []

    for mask_path in resolve_instance_paths(instance, limit_instances):
        masks = load_mask(mask_path)
        grid = mask_to_grid(masks)
        aisle_width = _scalar_int(masks.get("aisle_width"), default=1)
        instance_name = _scalar_text(masks.get("name"), default=mask_path.stem)
        run_id = f"{mask_path.stem}_seed_{int(seed)}"

        print(f"instance={instance_name} file={mask_path.name}")
        population = create_initial_population_for_grid(
            grid,
            aisle_width,
            int(population_size),
            seed=int(seed),
        )

        for generation in range(int(generations)):
            started = time.perf_counter()
            candidates, decoded_count = build_layout_candidates(
                population,
                grid,
                masks,
                aisle_width,
                int(beam_width),
                int(max_depth),
                seed=int(seed) + generation * 1000,
            )
            selected = select_nsga2_survivors(candidates, int(population_size))
            feasible_count = sum(1 for candidate in candidates if candidate.is_feasible)
            non_dominated_count = sum(
                1 for candidate in candidates if candidate.is_feasible and candidate.rank == 0
            )

            if candidates:
                for candidate in candidates:
                    all_rows.append(
                        candidate_to_csv_row(
                            candidate,
                            run_id,
                            instance_name,
                            int(seed),
                            generation,
                        )
                    )
            else:
                all_rows.append(
                    _empty_generation_row(
                        run_id,
                        instance_name,
                        int(seed),
                        generation,
                        "no_candidates",
                    )
                )

            if save_figures:
                _save_selected_figures(
                    selected,
                    image_dir,
                    mask_path.stem,
                    int(seed),
                    generation,
                )

            elapsed = time.perf_counter() - started
            summary = {
                "instance": instance_name,
                "generation": generation,
                "chromosome_count": len(population),
                "decoded_candidate_count": decoded_count,
                "feasible_candidate_count": feasible_count,
                "non_dominated_count": non_dominated_count,
                "selected_survivor_count": len(selected),
                "runtime_seconds": elapsed,
            }
            generation_summaries.append(summary)
            print(
                "generation={generation} chromosome_count={chromosome_count} "
                "decoded_candidate_count={decoded_candidate_count} "
                "feasible_candidate_count={feasible_candidate_count} "
                "non_dominated_count={non_dominated_count} "
                "selected_survivor_count={selected_survivor_count} "
                "runtime_seconds={runtime_seconds:.3f}".format(**summary)
            )

            population = make_next_generation(
                selected,
                population,
                int(population_size),
                rng,
                aisle_width,
            )

    csv_path = _write_csv(all_rows, output_path)
    return Nsga2BsResult(
        csv_path=csv_path,
        rows=all_rows,
        generation_summaries=generation_summaries,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the NSGA-II runner."""
    parser = argparse.ArgumentParser(
        description="Run a minimal NSGA-II plus Beam Search NSGA-II plus Beam Search pipeline.",
    )
    parser.add_argument("--instance", default=None, help="Mask filename, path, or registry ID.")
    parser.add_argument(
        "--limit-instances",
        type=int,
        default=1,
        help="Maximum discovered masks to process when --instance is omitted.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--population-size", type=int, default=6, help="Population size.")
    parser.add_argument("--generations", type=int, default=3, help="Generation count.")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam Search width.")
    parser.add_argument("--max-depth", type=int, default=8, help="Beam Search max depth.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_NSGA2_BS_CSV,
        help="CSV output path.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Do not save figures.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_nsga2_bs(
        instance=args.instance,
        limit_instances=args.limit_instances,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        output_csv=args.output_csv,
        save_figures=not args.no_figures,
    )
    print(f"CSV saved: {result.csv_path}")
    if args.no_figures:
        print("Figure saving disabled.")


if __name__ == "__main__":
    main()


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_ADAPTIVE_WEIGHT_MODE",
    "DEFAULT_INITIALIZATION_MODE",
    "DEFAULT_MUTATION_MODE",
    "DEFAULT_NSGA2_BS_CSV",
    "DEFAULT_SORTING_RULE",
    "DEFAULT_SORTING_RULE_MODE",
    "INITIALIZATION_SPACING_MODES",
    "LayoutCandidate",
    "MUTATION_MODES",
    "Nsga2BsResult",
    "OBJECTIVE_DIRECTIONS",
    "OBJECTIVE_KEYS",
    "RANDOM_FEASIBLE_INITIALIZATION_MODE",
    "REFERENCE_ONLY_LAYOUT_FILENAMES",
    "build_layout_candidates",
    "build_parser",
    "candidate_to_csv_row",
    "chromosome_signature_text",
    "create_initial_population_for_grid",
    "decode_chromosome_with_beam_search",
    "discover_instance_masks",
    "is_reference_only_layout",
    "extract_objective_metrics",
    "make_next_generation",
    "mutation_probabilities_for_mode",
    "objective_array",
    "resolve_instance_paths",
    "run_nsga2_bs",
    "select_nsga2_survivors",
    "symmetry_breaking_enabled_for_mode",
]
