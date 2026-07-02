"""Beam Search scoring utilities."""

from __future__ import annotations

import numpy as np
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
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a local scalar score for Beam Search pruning only."""
    selected_weights = DEFAULT_BEAM_WEIGHTS if weights is None else weights
    w1 = float(selected_weights.get("w1", DEFAULT_BEAM_WEIGHTS["w1"]))
    w2 = float(selected_weights.get("w2", DEFAULT_BEAM_WEIGHTS["w2"]))
    lambda_value = float(
        selected_weights.get("lambda", DEFAULT_BEAM_WEIGHTS["lambda"])
    )

    storage_total = max(float(metrics.get("storage_total", 0.0)), 1.0)
    pick_faces = float(metrics.get("pick_faces", 0.0))
    interior_storage = float(metrics.get("interior_storage", 0.0))
    retrieval_penalty = float(metrics.get("retrieval_penalty", 0.0))

    pick_face_ratio = pick_faces / storage_total
    interior_ratio = interior_storage / storage_total
    retrieval_penalty_normalized = retrieval_penalty / storage_total

    return float(
        w1 * pick_face_ratio
        - w2 * interior_ratio
        - lambda_value * retrieval_penalty_normalized
    )


def add_beam_metrics(
    node: BeamNode,
    weights: dict[str, float] | None = None,
) -> BeamNode:
    """Return a new node with structural and local scalar Beam metrics."""
    metrics = dict(score_layout(node.layout))
    metrics["scalar_score"] = compute_beam_scalar_score(metrics, weights)
    return node.copy_with(metrics=metrics)


def score_nodes(
    nodes: list[BeamNode],
    weights: dict[str, float] | None = None,
) -> list[BeamNode]:
    """Return scored copies of Beam Search nodes."""
    return [add_beam_metrics(node, weights=weights) for node in nodes]


__all__ = [
    "DEFAULT_BEAM_WEIGHTS",
    "add_beam_metrics",
    "compute_beam_scalar_score",
    "sample_beam_weights",
    "score_nodes",
]
