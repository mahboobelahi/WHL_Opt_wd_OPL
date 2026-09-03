"""Local Beam Search scoring utilities.

These helpers compute local ranking metrics for future Beam Search pruning.
They do not replace the final NSGA-II objective vector and do not implement a
Beam Search loop.
"""

from __future__ import annotations

import numpy as np
from whl_core.constants import ALL_CELL_CODES, STRUCTURAL_CODES
from whl_core.scoring import score_layout

from whl_algorithms.beam_node import BeamNode

DEFAULT_BEAM_WEIGHTS = {
    "w1": 1.0,
    "w2": 1.0,
    "lambda": 0.1,
}


def sample_beam_weights(
    rho: float,
    rng: np.random.Generator,
    sigma: float = 0.05,
    lower: float = 0.1,
    upper: float = 1.0,
    lambda_: float = 0.1,
) -> dict[str, float]:
    """Sample local Beam Search scalar-score weights."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1].")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if lower > upper:
        raise ValueError("lower must not exceed upper.")

    w1 = float(np.clip(rng.normal(rho, sigma), lower, upper))
    w2 = float(np.clip(1.0 - rho, lower, upper))
    return {
        "w1": w1,
        "w2": w2,
        "lambda": float(lambda_),
    }


def compute_beam_scalar_score(
    metrics: dict,
    available_floor_area: int | float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the submitted local Beam scalar score for pruning only.

    available_floor_area is the instance constant A_f. For the possible
    zero-storage edge case, N_pf / SC is defined locally as zero; every
    positive-storage candidate follows the submitted equation exactly.
    """
    selected_weights = DEFAULT_BEAM_WEIGHTS if weights is None else weights
    w1 = float(selected_weights.get("w1", DEFAULT_BEAM_WEIGHTS["w1"]))
    w2 = float(selected_weights.get("w2", DEFAULT_BEAM_WEIGHTS["w2"]))
    lambda_value = float(selected_weights.get("lambda", DEFAULT_BEAM_WEIGHTS["lambda"]))

    floor_area = float(available_floor_area)
    if floor_area <= 0.0:
        raise ValueError("available_floor_area must be positive.")

    storage_total = float(metrics.get("storage_total", 0.0))
    if storage_total < 0.0:
        raise ValueError("storage_total must be non-negative.")
    pick_faces = float(metrics.get("pick_faces", 0.0))
    retrieval_penalty = float(metrics.get("retrieval_penalty", 0.0))

    if storage_total == 0.0:
        if pick_faces != 0.0:
            raise ValueError("pick_faces must be zero when storage_total is zero.")
        pick_face_ratio = 0.0
    else:
        pick_face_ratio = pick_faces / storage_total

    return float(
        (storage_total - w1 * lambda_value * retrieval_penalty) / floor_area
        + w2 * pick_face_ratio
    )


def compute_available_floor_area(base_grid: np.ndarray) -> int:
    """Return instance-constant A_f from the original grid geometry.

    Storage, explicit pick-face, empty, and all aisle subtype cells are
    included. Walls, doors/access anchors, reserved cells, restricted cells,
    and pillars are excluded.
    """
    layout = np.asarray(base_grid)
    if layout.ndim != 2:
        raise ValueError("base_grid must be a 2D array.")
    unknown_codes = set(np.unique(layout).tolist()) - ALL_CELL_CODES
    if unknown_codes:
        raise ValueError(
            f"base_grid contains unknown cell codes: {sorted(unknown_codes)}"
        )
    available_floor_area = int(
        np.count_nonzero(~np.isin(layout, list(STRUCTURAL_CODES)))
    )
    if available_floor_area <= 0:
        raise ValueError("base_grid must contain at least one available floor cell.")
    return available_floor_area


def add_beam_metrics(
    node: BeamNode,
    weights: dict[str, float] | None = None,
    available_floor_area: int | float | None = None,
) -> BeamNode:
    """Return a new node with structural and local scalar Beam metrics."""
    metrics = dict(score_layout(node.layout))
    floor_area = (
        compute_available_floor_area(node.layout)
        if available_floor_area is None
        else available_floor_area
    )
    metrics["scalar_score"] = compute_beam_scalar_score(
        metrics,
        floor_area,
        weights,
    )
    return node.copy_with(metrics=metrics)


def score_nodes(
    nodes: list[BeamNode],
    weights: dict[str, float] | None = None,
    available_floor_area: int | float | None = None,
) -> list[BeamNode]:
    """Return scored copies of Beam Search nodes."""
    return [
        add_beam_metrics(
            node,
            weights=weights,
            available_floor_area=available_floor_area,
        )
        for node in nodes
    ]


__all__ = [
    "DEFAULT_BEAM_WEIGHTS",
    "add_beam_metrics",
    "compute_available_floor_area",
    "compute_beam_scalar_score",
    "sample_beam_weights",
    "score_nodes",
]
