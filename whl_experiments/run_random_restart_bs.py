"""Random chromosome restarts plus Beam Search baseline."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from whl_algorithms.beam_node import BeamNode, layout_signature
from whl_algorithms.beam_scoring import DEFAULT_BEAM_WEIGHTS, sample_beam_weights
from whl_algorithms.beam_sorting import load_sorting_rules
from whl_algorithms.nsga2 import (
    assign_ranks,
    crowding_distances_for_fronts,
    non_dominated_sort,
)
from whl_core.blocks import detect_storage_blocks
from whl_core.connectivity import access_anchor_connectivity_report
from whl_core.feasibility import (
    access_anchor_mask_from_grid_and_masks,
    check_layout_feasible,
    oriented_aisle_thickness_violations,
)
from whl_core.layout_io import fixed_aisle_mask_from_masks
from whl_core.scoring import score_layout
from whl_experiments import run_nsga2_bs as nsga2_bs

METHOD_NAME = "random_restart_bs"
DEFAULT_POPULATION_SIZE = 8
DEFAULT_GENERATIONS = 10


def resolve_decode_budget(
    population_size: int,
    generations: int,
    decode_budget: int | None = None,
) -> int:
    """Resolve restart count from explicit budget or population x generations."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    if generations <= 0:
        raise ValueError("generations must be positive.")
    if decode_budget is not None:
        if int(decode_budget) <= 0:
            raise ValueError("decode_budget must be positive.")
        return int(decode_budget)
    return int(population_size) * int(generations)


def restart_beam_weights(
    *,
    restart_index: int,
    decode_budget: int,
    seed: int,
    adaptive_weight_mode: str,
) -> tuple[dict[str, float], float]:
    """Return Step 9C-compatible scalar-score weights for one restart."""
    if adaptive_weight_mode not in nsga2_bs.ADAPTIVE_WEIGHT_MODES:
        raise ValueError(
            f"adaptive_weight_mode must be one of {nsga2_bs.ADAPTIVE_WEIGHT_MODES}."
        )
    if restart_index <= 0:
        raise ValueError("restart_index must be one-based and positive.")
    if decode_budget <= 0:
        raise ValueError("decode_budget must be positive.")
    rho = float(int(restart_index) / int(decode_budget))
    if adaptive_weight_mode == "fixed":
        return dict(DEFAULT_BEAM_WEIGHTS), rho
    rng_seed = int(seed) * 1_000_003 + int(restart_index) * 101
    rng = np.random.default_rng(rng_seed)
    return sample_beam_weights(rho, rng), rho


def _evaluate_node(
    node: BeamNode,
    *,
    masks: dict[str, Any],
    aisle_width: int,
    fixed_aisle_mask: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
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
    return metrics, feasibility, exact_width_violations


def objective_array(candidates: list[nsga2_bs.LayoutCandidate]) -> np.ndarray:
    return np.asarray(
        [[candidate.metrics[key] for key in nsga2_bs.OBJECTIVE_KEYS] for candidate in candidates],
        dtype=float,
    )


def assign_posthoc_ranks(
    candidates: list[nsga2_bs.LayoutCandidate],
) -> list[nsga2_bs.LayoutCandidate]:
    """Rank feasible restart candidates post hoc without survivor selection."""
    feasible = [candidate for candidate in candidates if candidate.is_feasible]
    if not feasible:
        return []
    objectives = objective_array(feasible)
    fronts = non_dominated_sort(objectives, nsga2_bs.OBJECTIVE_DIRECTIONS)
    ranks = assign_ranks(fronts, len(feasible))
    crowding = crowding_distances_for_fronts(
        objectives,
        fronts,
        nsga2_bs.OBJECTIVE_DIRECTIONS,
    )
    for index, candidate in enumerate(feasible):
        candidate.rank = ranks[index]
        candidate.crowding_distance = crowding[index]
        candidate.selected = candidate.rank == 0
    return [candidate for candidate in feasible if candidate.rank == 0]


def _batch_index(restart_index: int, population_size: int) -> int:
    return int((int(restart_index) - 1) // int(population_size))


def _within_batch_index(restart_index: int, population_size: int) -> int:
    return int((int(restart_index) - 1) % int(population_size) + 1)


def build_random_restart_candidates(
    grid: np.ndarray,
    masks: dict[str, Any],
    *,
    aisle_width: int,
    population_size: int,
    generations: int,
    decode_budget: int | None,
    beam_width: int,
    max_depth: int,
    seed: int,
    sorting_rule_mode: str = nsga2_bs.DEFAULT_SORTING_RULE_MODE,
    sorting_rule: str = nsga2_bs.DEFAULT_SORTING_RULE,
    adaptive_weight_mode: str = nsga2_bs.DEFAULT_ADAPTIVE_WEIGHT_MODE,
    sorting_rules: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[list[nsga2_bs.LayoutCandidate], list[dict[str, Any]], dict[str, Any]]:
    """Sample restart chromosomes, decode each once, deduplicate, then rank."""
    restart_count = resolve_decode_budget(population_size, generations, decode_budget)
    selected_sorting_rules = load_sorting_rules() if sorting_rules is None else sorting_rules
    fixed_aisles = fixed_aisle_mask_from_masks(masks)
    population = nsga2_bs.create_initial_population_for_grid(
        grid,
        int(aisle_width),
        restart_count,
        seed=int(seed),
    )

    candidates: list[nsga2_bs.LayoutCandidate] = []
    seen_layouts: set[bytes] = set()
    decoded_node_count = 0
    duplicate_layout_count = 0
    batch_rows: dict[int, dict[str, Any]] = {}
    sorting_rule_counts: Counter[str] = Counter()
    scalar_restart_count = 0
    initialization_mode_counts: Counter[str] = Counter()

    for restart_index, chromosome in enumerate(population, start=1):
        batch = _batch_index(restart_index, population_size)
        within_batch = _within_batch_index(restart_index, population_size)
        row = batch_rows.setdefault(
            batch,
            {
                "generation": batch,
                "chromosome_count": 0,
                "decoded_candidate_count": 0,
                "runtime_seconds": 0.0,
            },
        )
        row["chromosome_count"] += 1
        restart_started = time.perf_counter()
        decode_rng = np.random.default_rng(int(seed) * 1_000_003 + restart_index)
        selected_sorting_rule = nsga2_bs.choose_sorting_rule_for_decode(
            sorting_rule_mode=sorting_rule_mode,
            fixed_sorting_rule=sorting_rule,
            sorting_rules=selected_sorting_rules,
            rng=decode_rng,
        )
        uses_scalar = nsga2_bs.sorting_rule_uses_scalar_score(
            selected_sorting_rule,
            selected_sorting_rules,
        )
        beam_weights, rho = restart_beam_weights(
            restart_index=restart_index,
            decode_budget=restart_count,
            seed=seed,
            adaptive_weight_mode=adaptive_weight_mode,
        )
        weights_for_sorting = beam_weights if uses_scalar else None
        init_mode = nsga2_bs._chromosome_initialization_mode(chromosome)
        h_starts = nsga2_bs.chromosome_active_text(chromosome.active_h_indices())
        v_starts = nsga2_bs.chromosome_active_text(chromosome.active_v_indices())
        decode_metadata = {
            "restart_index": restart_index,
            "batch_index": batch,
            "within_batch_index": within_batch,
            "decode_budget": restart_count,
            "chromosome_index": restart_index,
            "sorting_rule": selected_sorting_rule,
            "sorting_rule_mode": sorting_rule_mode,
            "uses_scalar_score": bool(uses_scalar),
            "rho": rho,
            "beam_w1": beam_weights["w1"],
            "beam_w2": beam_weights["w2"],
            "beam_lambda": beam_weights["lambda"],
            "adaptive_weight_mode": adaptive_weight_mode,
            "initialization_mode": init_mode,
            "initialization_spacing_mode": getattr(
                chromosome,
                "initialization_spacing_mode",
                init_mode,
            ),
            "adaptive_spacing_used": getattr(chromosome, "adaptive_spacing_used", ""),
            "feasible_h_start_count": getattr(chromosome, "feasible_h_start_count", ""),
            "feasible_v_start_count": getattr(chromosome, "feasible_v_start_count", ""),
            "h_active_starts": h_starts,
            "v_active_starts": v_starts,
            "terminal": True,
        }
        nodes = nsga2_bs.decode_chromosome_with_beam_search(
            chromosome,
            grid,
            int(aisle_width),
            int(beam_width),
            int(max_depth),
            seed=int(seed) + restart_index,
            sorting_rule=selected_sorting_rule,
            sorting_rules=selected_sorting_rules,
            weights=weights_for_sorting,
        )
        decoded_node_count += len(nodes)
        row["decoded_candidate_count"] += len(nodes)
        sorting_rule_counts[selected_sorting_rule] += 1
        scalar_restart_count += int(bool(uses_scalar))
        initialization_mode_counts[init_mode] += 1

        for node in nodes:
            signature = layout_signature(node.layout)
            if signature in seen_layouts:
                duplicate_layout_count += 1
                continue
            seen_layouts.add(signature)
            metrics, feasibility, exact_width_violations = _evaluate_node(
                node,
                masks=masks,
                aisle_width=int(aisle_width),
                fixed_aisle_mask=fixed_aisles,
            )
            candidates.append(
                nsga2_bs.LayoutCandidate(
                    node=node,
                    chromosome=nsga2_bs._node_chromosome_state(node, chromosome),
                    parent_chromosome_id=restart_index,
                    metrics=metrics,
                    feasibility=feasibility,
                    exact_width_violations=exact_width_violations,
                    decode_metadata=dict(decode_metadata),
                )
            )
        row["runtime_seconds"] += time.perf_counter() - restart_started

    for candidate_id, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = candidate_id
    assign_posthoc_ranks(candidates)

    batch_summaries: list[dict[str, Any]] = []
    for batch in sorted(batch_rows):
        row = dict(batch_rows[batch])
        batch_candidates = [
            candidate
            for candidate in candidates
            if int(candidate.decode_metadata.get("batch_index", -1)) == batch
        ]
        feasible_count = sum(1 for candidate in batch_candidates if candidate.is_feasible)
        rank0_count = sum(
            1
            for candidate in batch_candidates
            if candidate.is_feasible and candidate.rank == 0
        )
        row.update(
            {
                "feasible_candidate_count": feasible_count,
                "non_dominated_count": rank0_count,
                "selected_survivor_count": rank0_count,
            }
        )
        batch_summaries.append(row)

    metadata = {
        "decode_budget": restart_count,
        "population_size_equivalent": int(population_size),
        "generation_count_equivalent": int(generations),
        "decoded_node_count": int(decoded_node_count),
        "unique_candidate_count": len(candidates),
        "duplicate_layout_count": int(duplicate_layout_count),
        "feasible_candidate_count": sum(1 for candidate in candidates if candidate.is_feasible),
        "rank0_candidate_count": sum(
            1 for candidate in candidates if candidate.is_feasible and candidate.rank == 0
        ),
        "sorting_rule_counts": dict(sorted(sorting_rule_counts.items())),
        "scalar_score_restart_count": int(scalar_restart_count),
        "initialization_mode_counts": dict(sorted(initialization_mode_counts.items())),
    }
    return candidates, batch_summaries, metadata


def candidate_status(candidate: nsga2_bs.LayoutCandidate) -> str:
    if not candidate.is_feasible:
        return "infeasible"
    return "selected" if candidate.selected else "evaluated"


def candidate_to_csv_row(
    candidate: nsga2_bs.LayoutCandidate,
    *,
    run_id: str,
    method: str,
    instance_name: str,
    seed: int,
    generation: int | None = None,
) -> dict[str, Any]:
    metadata = candidate.decode_metadata
    h_count, v_count = candidate.chromosome.active_count()
    blocks = detect_storage_blocks(candidate.node.layout)
    violations = candidate.feasibility.get("violations", [])
    feasibility_reason = "|".join(str(item) for item in violations)
    selected_generation = metadata.get("batch_index", 0) if generation is None else generation
    return {
        "run_id": run_id,
        "method": method,
        "instance": instance_name,
        "seed": int(seed),
        "generation": int(selected_generation),
        "restart_index": metadata.get("restart_index", ""),
        "batch_index": metadata.get("batch_index", ""),
        "within_batch_index": metadata.get("within_batch_index", ""),
        "decode_budget": metadata.get("decode_budget", ""),
        "candidate_id": int(candidate.candidate_id),
        "parent_chromosome_id": int(candidate.parent_chromosome_id),
        "depth": int(candidate.node.depth),
        "trace": " > ".join(str(item) for item in candidate.node.trace),
        "status": candidate_status(candidate),
        "rank": "" if candidate.rank is None else int(candidate.rank),
        "crowding_distance": ""
        if candidate.crowding_distance is None
        else float(candidate.crowding_distance),
        "selected": bool(candidate.selected),
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
        "access_network_components": candidate.metrics.get("access_network_components", ""),
        "aisle_access_network_components": candidate.metrics.get(
            "aisle_access_network_components",
            "",
        ),
        "unreachable_aisle_components": candidate.metrics.get(
            "unreachable_aisle_components",
            "",
        ),
        "unreachable_aisle_cells": candidate.metrics.get("unreachable_aisle_cells", ""),
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
        "chromosome_index": metadata.get("chromosome_index", ""),
        "sorting_rule": metadata.get("sorting_rule", ""),
        "sorting_rule_mode": metadata.get("sorting_rule_mode", ""),
        "uses_scalar_score": metadata.get("uses_scalar_score", ""),
        "rho": metadata.get("rho", ""),
        "beam_w1": metadata.get("beam_w1", ""),
        "beam_w2": metadata.get("beam_w2", ""),
        "beam_lambda": metadata.get("beam_lambda", ""),
        "adaptive_weight_mode": metadata.get("adaptive_weight_mode", ""),
        "initialization_mode": metadata.get("initialization_mode", ""),
        "initialization_spacing_mode": metadata.get("initialization_spacing_mode", ""),
        "adaptive_spacing_used": metadata.get("adaptive_spacing_used", ""),
        "feasible_h_start_count": metadata.get("feasible_h_start_count", ""),
        "feasible_v_start_count": metadata.get("feasible_v_start_count", ""),
        "h_active_starts": metadata.get("h_active_starts", ""),
        "v_active_starts": metadata.get("v_active_starts", ""),
        "terminal": metadata.get("terminal", True),
        "feasible": bool(candidate.is_feasible),
        "feasibility_reason": feasibility_reason,
        "storage_block_count": len(blocks),
        "chromosome_signature": nsga2_bs.chromosome_signature_text(candidate.chromosome),
        "layout_signature": nsga2_bs._signature_digest(layout_signature(candidate.node.layout)),
    }


__all__ = [
    "DEFAULT_GENERATIONS",
    "DEFAULT_POPULATION_SIZE",
    "METHOD_NAME",
    "assign_posthoc_ranks",
    "build_random_restart_candidates",
    "candidate_to_csv_row",
    "restart_beam_weights",
    "resolve_decode_budget",
]
