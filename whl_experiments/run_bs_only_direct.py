"""Direct Beam Search-only baseline for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from whl_algorithms.beam_node import BeamNode, layout_signature
from whl_algorithms.beam_search import BeamSearchConfig, run_direct_beam_search
from whl_algorithms.beam_sorting import load_sorting_rules
from whl_algorithms.nsga2 import (
    assign_ranks,
    crowding_distances_for_fronts,
    non_dominated_sort,
)
from whl_algorithms.parameter_policy import min_fragment_size
from whl_core.blocks import detect_storage_blocks
from whl_core.connectivity import access_anchor_connectivity_report
from whl_core.feasibility import (
    access_anchor_mask_from_grid_and_masks,
    check_layout_feasible,
    oriented_aisle_thickness_violations,
)
from whl_core.layout_io import fixed_aisle_mask_from_masks
from whl_core.scoring import score_layout
from whl_experiments.run_nsga2_bs import (
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_KEYS,
    _signature_digest,
    sorting_rule_uses_scalar_score,
)

METHOD_NAME = "bs_only"
BS_RULE_POLICIES = ("all_rules", "fixed")
BS_WEIGHT_POLICIES = ("fixed",)
DEFAULT_BS_RULE_POLICY = "all_rules"
DEFAULT_BS_WEIGHT_POLICY = "fixed"
FIXED_BS_WEIGHTS = {
    "w1": 0.5,
    "w2": 0.5,
    "lambda": 0.1,
}


@dataclass
class DirectLayoutCandidate:
    """Layout-level record produced by direct Beam Search."""

    node: BeamNode
    parent_chromosome_id: int | str
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


def fixed_bs_weights() -> dict[str, float]:
    return dict(FIXED_BS_WEIGHTS)


def sorting_rules_for_policy(
    *,
    bs_rule_policy: str,
    sorting_rule: str,
    sorting_rules: dict[str, list[tuple[str, str]]],
) -> list[str]:
    if bs_rule_policy not in BS_RULE_POLICIES:
        raise ValueError(f"bs_rule_policy must be one of {BS_RULE_POLICIES}.")
    if bs_rule_policy == "fixed":
        if sorting_rule not in sorting_rules:
            raise KeyError(f"unknown sorting rule: {sorting_rule}")
        return [sorting_rule]
    return sorted(sorting_rules)


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


def objective_array(candidates: list[DirectLayoutCandidate]) -> np.ndarray:
    return np.asarray(
        [[candidate.metrics[key] for key in OBJECTIVE_KEYS] for candidate in candidates],
        dtype=float,
    )


def assign_posthoc_ranks(candidates: list[DirectLayoutCandidate]) -> list[DirectLayoutCandidate]:
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
        candidate.selected = candidate.rank == 0
    return [candidate for candidate in feasible if candidate.rank == 0]


def build_direct_bs_candidates(
    grid: np.ndarray,
    masks: dict[str, Any],
    *,
    aisle_width: int,
    beam_width: int,
    safety_max_depth: int,
    sorting_rule: str,
    bs_rule_policy: str = DEFAULT_BS_RULE_POLICY,
    bs_weight_policy: str = DEFAULT_BS_WEIGHT_POLICY,
    sorting_rules: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[list[DirectLayoutCandidate], list[dict[str, Any]], dict[str, Any]]:
    if bs_weight_policy not in BS_WEIGHT_POLICIES:
        raise ValueError(f"bs_weight_policy must be one of {BS_WEIGHT_POLICIES}.")

    selected_sorting_rules = load_sorting_rules() if sorting_rules is None else sorting_rules
    rule_names = sorting_rules_for_policy(
        bs_rule_policy=bs_rule_policy,
        sorting_rule=sorting_rule,
        sorting_rules=selected_sorting_rules,
    )
    fixed_aisles = fixed_aisle_mask_from_masks(masks)
    weights = fixed_bs_weights()
    seen_layouts: set[bytes] = set()
    candidates: list[DirectLayoutCandidate] = []
    depth_rows: list[dict[str, Any]] = []
    decoded_count = 0
    safety_reached = False
    max_depth_reached = 0

    for rule_name in rule_names:
        uses_scalar = sorting_rule_uses_scalar_score(rule_name, selected_sorting_rules)
        weights_for_sorting = weights if uses_scalar else None
        config = BeamSearchConfig(
            aisle_width=int(aisle_width),
            beam_width=int(beam_width),
            max_depth=int(safety_max_depth),
            min_fragment_size=min_fragment_size(int(aisle_width)),
            sorting_rule=rule_name,
        )
        result = run_direct_beam_search(
            grid,
            config,
            sorting_rules=selected_sorting_rules,
            weights=weights_for_sorting,
        )
        decoded_count += result.decoded_count
        safety_reached = safety_reached or result.safety_max_depth_reached
        max_depth_reached = max(max_depth_reached, result.max_depth_reached)
        for summary in result.depth_summaries:
            depth_rows.append(
                {
                    "sorting_rule": rule_name,
                    "depth": int(summary.depth),
                    "input_node_count": int(summary.input_node_count),
                    "generated_child_count": int(summary.generated_child_count),
                    "unique_child_count": int(summary.unique_child_count),
                    "retained_node_count": int(summary.retained_node_count),
                    "terminal_node_count": int(summary.terminal_node_count),
                    "safety_max_depth_reached": bool(summary.safety_max_depth_reached),
                }
            )

        for node in result.terminal_nodes:
            signature = layout_signature(node.layout)
            if signature in seen_layouts:
                continue
            seen_layouts.add(signature)
            metrics, feasibility, exact_width_violations = _evaluate_node(
                node,
                masks=masks,
                aisle_width=int(aisle_width),
                fixed_aisle_mask=fixed_aisles,
            )
            candidates.append(
                DirectLayoutCandidate(
                    node=node,
                    parent_chromosome_id="",
                    metrics=metrics,
                    feasibility=feasibility,
                    exact_width_violations=exact_width_violations,
                    decode_metadata={
                        "sorting_rule": rule_name,
                        "sorting_rule_mode": "",
                        "bs_rule_policy": bs_rule_policy,
                        "bs_weight_policy": bs_weight_policy,
                        "uses_scalar_score": bool(uses_scalar),
                        "rho": "",
                        "beam_w1": weights["w1"],
                        "beam_w2": weights["w2"],
                        "beam_lambda": weights["lambda"],
                        "adaptive_weight_mode": "",
                        "initialization_mode": "direct_root",
                        "h_active_starts": "",
                        "v_active_starts": "",
                        "terminal": True,
                        "safety_max_depth_reached": bool(result.safety_max_depth_reached),
                    },
                )
            )

    for candidate_id, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = candidate_id
    assign_posthoc_ranks(candidates)
    metadata = {
        "decoded_count": int(decoded_count),
        "rule_names": rule_names,
        "bs_rule_policy": bs_rule_policy,
        "bs_weight_policy": bs_weight_policy,
        "beam_w1": weights["w1"],
        "beam_w2": weights["w2"],
        "beam_lambda": weights["lambda"],
        "safety_max_depth_reached": bool(safety_reached),
        "max_depth_reached": int(max_depth_reached),
        "terminal_candidate_count": len(candidates),
        "feasible_candidate_count": sum(1 for candidate in candidates if candidate.is_feasible),
    }
    return candidates, depth_rows, metadata


def candidate_status(candidate: DirectLayoutCandidate) -> str:
    if not candidate.is_feasible:
        return "infeasible"
    return "selected" if candidate.selected else "evaluated"


def candidate_to_csv_row(
    candidate: DirectLayoutCandidate,
    *,
    run_id: str,
    method: str,
    instance_name: str,
    seed: int,
    generation: int = 0,
) -> dict[str, Any]:
    metadata = candidate.decode_metadata
    blocks = detect_storage_blocks(candidate.node.layout)
    violations = candidate.feasibility.get("violations", [])
    feasibility_reason = "|".join(str(item) for item in violations)
    return {
        "run_id": run_id,
        "method": method,
        "instance": instance_name,
        "seed": int(seed),
        "generation": int(generation),
        "candidate_id": int(candidate.candidate_id),
        "parent_chromosome_id": "",
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
        "chromosome_h_active_count": "",
        "chromosome_v_active_count": "",
        "chromosome_index": "",
        "sorting_rule": metadata.get("sorting_rule", ""),
        "sorting_rule_mode": metadata.get("sorting_rule_mode", ""),
        "bs_rule_policy": metadata.get("bs_rule_policy", ""),
        "bs_weight_policy": metadata.get("bs_weight_policy", ""),
        "uses_scalar_score": metadata.get("uses_scalar_score", ""),
        "rho": "",
        "beam_w1": metadata.get("beam_w1", ""),
        "beam_w2": metadata.get("beam_w2", ""),
        "beam_lambda": metadata.get("beam_lambda", ""),
        "adaptive_weight_mode": "",
        "initialization_mode": "direct_root",
        "h_active_starts": "",
        "v_active_starts": "",
        "terminal": metadata.get("terminal", True),
        "safety_max_depth_reached": metadata.get("safety_max_depth_reached", False),
        "feasible": bool(candidate.is_feasible),
        "feasibility_reason": feasibility_reason,
        "storage_block_count": len(blocks),
        "chromosome_signature": "",
        "layout_signature": _signature_digest(layout_signature(candidate.node.layout)),
    }


__all__ = [
    "BS_RULE_POLICIES",
    "BS_WEIGHT_POLICIES",
    "DEFAULT_BS_RULE_POLICY",
    "DEFAULT_BS_WEIGHT_POLICY",
    "FIXED_BS_WEIGHTS",
    "METHOD_NAME",
    "DirectLayoutCandidate",
    "assign_posthoc_ranks",
    "build_direct_bs_candidates",
    "candidate_to_csv_row",
    "fixed_bs_weights",
    "sorting_rules_for_policy",
]
