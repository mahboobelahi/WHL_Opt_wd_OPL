"""Beam Search expansion and pruning skeleton.

This module coordinates root initialization, global carving, block-level
carving, local scoring, and sorting/pruning. It does not implement NSGA-II
integration, baselines, experiment runners, or the proposed NSGA-II + Beam
Search wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whl_algorithms.beam_node import (
    BeamNode,
    initialize_direct_root_node,
    initialize_root_node,
    layout_signature,
)
from whl_algorithms.beam_scoring import (
    add_beam_metrics,
    compute_available_floor_area,
    score_nodes,
)
from whl_algorithms.beam_sorting import load_sorting_rules, sort_nodes_by_rule
from whl_algorithms.carving import (
    generate_block_children,
    generate_direct_global_children,
    generate_global_children,
)


@dataclass
class BeamSearchConfig:
    """Configuration for the standalone Beam Search decoder skeleton."""

    aisle_width: int
    beam_width: int = 3
    max_depth: int = 10
    min_fragment_size: int = 2
    sorting_rule: str = "PF_LS_RP"
    allowed_sorting_rules: list[str] | None = None
    max_children_per_block: int | None = None
    require_final_door_connected: bool = False
    require_final_single_aisle_component: bool = False

    def __post_init__(self) -> None:
        if self.aisle_width <= 0:
            raise ValueError("aisle_width must be positive.")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if self.min_fragment_size <= 0:
            raise ValueError("min_fragment_size must be positive.")
        if self.max_children_per_block is not None and self.max_children_per_block <= 0:
            raise ValueError("max_children_per_block must be positive when provided.")
        if (
            self.allowed_sorting_rules is not None
            and self.sorting_rule not in self.allowed_sorting_rules
        ):
            raise ValueError("sorting_rule must be listed in allowed_sorting_rules.")


@dataclass
class BeamDepthSummary:
    """One direct Beam Search expansion-depth summary."""

    sorting_rule: str
    depth: int
    input_node_count: int
    generated_child_count: int
    unique_child_count: int
    retained_node_count: int
    terminal_node_count: int
    safety_max_depth_reached: bool = False


@dataclass
class DirectBeamSearchResult:
    """Direct Beam Search output for one sorting rule."""

    sorting_rule: str
    terminal_nodes: list[BeamNode]
    retained_by_depth: list[tuple[int, list[BeamNode]]] = field(default_factory=list)
    depth_summaries: list[BeamDepthSummary] = field(default_factory=list)
    decoded_count: int = 0
    safety_max_depth_reached: bool = False
    max_depth_reached: int = 0


def expand_beam_node(
    node: BeamNode,
    chromosome,
    config: BeamSearchConfig,
    rng: np.random.Generator,
) -> list[BeamNode]:
    """Expand a root node globally or a deeper node by block carving."""
    del rng
    if node.depth == 0:
        return generate_global_children(node, chromosome, config.aisle_width)
    return generate_block_children(
        node,
        aisle_width=config.aisle_width,
        min_fragment_size=config.min_fragment_size,
        max_children_per_block=config.max_children_per_block,
    )


def expand_direct_beam_node(
    node: BeamNode,
    config: BeamSearchConfig,
) -> list[BeamNode]:
    """Expand a direct BS node without chromosome state."""
    if node.depth == 0:
        return generate_direct_global_children(node, config.aisle_width)
    return generate_block_children(
        node,
        aisle_width=config.aisle_width,
        min_fragment_size=config.min_fragment_size,
        max_children_per_block=config.max_children_per_block,
    )


def score_and_prune_nodes(
    nodes: list[BeamNode],
    config: BeamSearchConfig,
    sorting_rules: dict,
    weights: dict | None = None,
    available_floor_area: int | float | None = None,
) -> list[BeamNode]:
    """Score candidate nodes locally and keep the best beam-width entries."""
    if not nodes:
        return []
    scored = score_nodes(
        nodes,
        weights=weights,
        available_floor_area=available_floor_area,
    )
    sorted_nodes = sort_nodes_by_rule(
        scored,
        config.sorting_rule,
        rules=sorting_rules,
    )
    return sorted_nodes[: config.beam_width]


def _deduplicate_nodes(nodes: list[BeamNode]) -> list[BeamNode]:
    deduplicated: list[BeamNode] = []
    seen: set[bytes] = set()
    for node in nodes:
        signature = layout_signature(node.layout)
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(node)
    return deduplicated


def _deduplicate_nodes_with_count(nodes: list[BeamNode]) -> tuple[list[BeamNode], int]:
    deduplicated = _deduplicate_nodes(nodes)
    return deduplicated, len(deduplicated)


def beam_search_step(
    current_beam: list[BeamNode],
    chromosome,
    config: BeamSearchConfig,
    rng: np.random.Generator,
    sorting_rules: dict,
    weights: dict | None = None,
    available_floor_area: int | float | None = None,
) -> list[BeamNode]:
    """Expand, deduplicate, score, sort, and prune one Beam Search layer."""
    all_children: list[BeamNode] = []
    for node in current_beam:
        all_children.extend(expand_beam_node(node, chromosome, config, rng))
    if not all_children:
        return []

    unique_children = _deduplicate_nodes(all_children)
    return score_and_prune_nodes(
        unique_children,
        config,
        sorting_rules=sorting_rules,
        weights=weights,
        available_floor_area=available_floor_area,
    )


def direct_beam_search_step(
    current_beam: list[BeamNode],
    config: BeamSearchConfig,
    sorting_rules: dict,
    weights: dict | None = None,
    available_floor_area: int | float | None = None,
) -> tuple[list[BeamNode], list[BeamNode], int, int]:
    """Expand one direct BS layer and return nodes plus generation counts."""
    all_children: list[BeamNode] = []
    terminal_nodes: list[BeamNode] = []
    for node in current_beam:
        children = expand_direct_beam_node(node, config)
        if children:
            all_children.extend(children)
        else:
            terminal_nodes.append(
                _ensure_metrics(node, available_floor_area=available_floor_area)
            )

    if not all_children:
        return [], terminal_nodes, 0, 0

    unique_children, unique_count = _deduplicate_nodes_with_count(all_children)
    retained = score_and_prune_nodes(
        unique_children,
        config,
        sorting_rules=sorting_rules,
        weights=weights,
        available_floor_area=available_floor_area,
    )
    return retained, terminal_nodes, len(all_children), unique_count


def _ensure_metrics(
    node: BeamNode,
    available_floor_area: int | float | None = None,
) -> BeamNode:
    if (
        "has_door_connected_aisle" in node.metrics
        and "aisle_components" in node.metrics
    ):
        return node
    return add_beam_metrics(node, available_floor_area=available_floor_area)


def final_candidate_filter(
    nodes: list[BeamNode],
    config: BeamSearchConfig,
    available_floor_area: int | float | None = None,
) -> list[BeamNode]:
    """Apply optional final connectivity filters to candidate nodes."""
    filtered: list[BeamNode] = []
    for node in nodes:
        candidate = _ensure_metrics(
            node,
            available_floor_area=available_floor_area,
        )
        if (
            config.require_final_door_connected
            and not candidate.metrics["has_door_connected_aisle"]
        ):
            continue
        if (
            config.require_final_single_aisle_component
            and candidate.metrics["aisle_components"] != 1
        ):
            continue
        filtered.append(candidate)
    return filtered


def run_beam_search(
    chromosome,
    base_grid: np.ndarray,
    config: BeamSearchConfig,
    rng: np.random.Generator | None = None,
    sorting_rules: dict | None = None,
    weights: dict | None = None,
) -> list[BeamNode]:
    """Run the standalone Beam Search decoder and return mature beam nodes.

    Earlier smoke versions accumulated every intermediate beam node and returned
    them all. That lets depth-1/global-only layouts enter NSGA-II selection even
    if deeper Beam Search steps would repair or refine those layouts. Returning
    the final non-empty beam keeps candidate evaluation aligned with Beam Search:
    expand, prune, continue, then evaluate the mature beam.
    """
    selected_rng = np.random.default_rng() if rng is None else rng
    selected_rules = load_sorting_rules() if sorting_rules is None else sorting_rules
    available_floor_area = compute_available_floor_area(base_grid)

    root = initialize_root_node(chromosome, base_grid)
    current_beam = [root]
    last_nonempty_beam: list[BeamNode] = []

    for _depth in range(config.max_depth):
        next_beam = beam_search_step(
            current_beam,
            chromosome,
            config,
            selected_rng,
            selected_rules,
            weights=weights,
            available_floor_area=available_floor_area,
        )
        if not next_beam:
            break
        last_nonempty_beam = next_beam
        current_beam = next_beam

    candidates = _deduplicate_nodes(last_nonempty_beam)
    filtered = final_candidate_filter(
        candidates,
        config,
        available_floor_area=available_floor_area,
    )
    return filtered if filtered else candidates


def run_direct_beam_search(
    base_grid: np.ndarray,
    config: BeamSearchConfig,
    sorting_rules: dict | None = None,
    weights: dict | None = None,
) -> DirectBeamSearchResult:
    """Run direct Beam Search from the root layout without chromosomes."""
    selected_rules = load_sorting_rules() if sorting_rules is None else sorting_rules
    available_floor_area = compute_available_floor_area(base_grid)
    root = initialize_direct_root_node(base_grid)
    current_beam = [root]
    retained_by_depth: list[tuple[int, list[BeamNode]]] = []
    terminal_nodes: list[BeamNode] = []
    depth_summaries: list[BeamDepthSummary] = []
    decoded_count = 0
    safety_reached = False

    for _ in range(config.max_depth):
        depth = max(node.depth for node in current_beam) if current_beam else 0
        next_beam, newly_terminal, generated_count, unique_count = (
            direct_beam_search_step(
                current_beam,
                config,
                sorting_rules=selected_rules,
                weights=weights,
                available_floor_area=available_floor_area,
            )
        )
        terminal_nodes.extend(newly_terminal)
        decoded_count += unique_count
        retained_by_depth.append((depth + 1, next_beam))
        depth_summaries.append(
            BeamDepthSummary(
                sorting_rule=config.sorting_rule,
                depth=depth + 1,
                input_node_count=len(current_beam),
                generated_child_count=generated_count,
                unique_child_count=unique_count,
                retained_node_count=len(next_beam),
                terminal_node_count=len(newly_terminal),
                safety_max_depth_reached=False,
            )
        )
        if not next_beam:
            break
        current_beam = next_beam
    else:
        safety_reached = True
        terminal_nodes.extend(
            _ensure_metrics(node, available_floor_area=available_floor_area)
            for node in current_beam
        )
        if depth_summaries:
            depth_summaries[-1].safety_max_depth_reached = True

    terminal_nodes = _deduplicate_nodes(terminal_nodes)
    max_depth_reached = max(
        [node.depth for node in terminal_nodes]
        + [depth for depth, _ in retained_by_depth],
        default=0,
    )
    return DirectBeamSearchResult(
        sorting_rule=config.sorting_rule,
        terminal_nodes=terminal_nodes,
        retained_by_depth=retained_by_depth,
        depth_summaries=depth_summaries,
        decoded_count=decoded_count,
        safety_max_depth_reached=safety_reached,
        max_depth_reached=max_depth_reached,
    )


__all__ = [
    "BeamSearchConfig",
    "BeamDepthSummary",
    "DirectBeamSearchResult",
    "beam_search_step",
    "direct_beam_search_step",
    "expand_beam_node",
    "expand_direct_beam_node",
    "final_candidate_filter",
    "run_direct_beam_search",
    "run_beam_search",
    "score_and_prune_nodes",
]
