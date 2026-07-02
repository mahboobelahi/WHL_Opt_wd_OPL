"""Beam Search node state and root initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _copy_2d_layout(layout: np.ndarray) -> np.ndarray:
    array = np.asarray(layout)
    if array.ndim != 2:
        raise ValueError("layout must be a 2D array.")
    return array.copy()


def _copy_optional_vector(vector: np.ndarray | None, name: str) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    return array.copy()


def layout_signature(grid: np.ndarray) -> bytes:
    """Return a stable signature based on shape, dtype, and raw bytes."""
    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    header = f"shape={array.shape};dtype={array.dtype};".encode("ascii")
    return header + array.tobytes()


@dataclass
class BeamNode:
    """Beam Search node state."""

    layout: np.ndarray
    depth: int = 0
    action: str = "root"
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    chromosome_signature: tuple | None = None
    remaining_h: np.ndarray | None = None
    remaining_v: np.ndarray | None = None
    parent_id: int | None = None
    trace: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.layout = _copy_2d_layout(self.layout)
        self.remaining_h = _copy_optional_vector(self.remaining_h, "remaining_h")
        self.remaining_v = _copy_optional_vector(self.remaining_v, "remaining_v")
        self.metrics = dict(self.metrics)
        self.trace = tuple(self.trace)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the layout shape."""
        return self.layout.shape

    @property
    def signature(self) -> bytes:
        """Return the stable layout signature for this node."""
        return layout_signature(self.layout)

    def copy_with(
        self,
        layout: np.ndarray | None = None,
        depth: int | None = None,
        action: str | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        remaining_h: np.ndarray | None = None,
        remaining_v: np.ndarray | None = None,
        parent_id: int | None = None,
        trace_append: str | None = None,
    ) -> BeamNode:
        """Return a copied node with selected fields replaced."""
        next_trace = self.trace
        if trace_append is not None:
            next_trace = (*next_trace, trace_append)

        return BeamNode(
            layout=self.layout if layout is None else layout,
            depth=self.depth if depth is None else depth,
            action=self.action if action is None else action,
            metrics=self.metrics if metrics is None else metrics,
            chromosome_signature=self.chromosome_signature,
            remaining_h=self.remaining_h if remaining_h is None else remaining_h,
            remaining_v=self.remaining_v if remaining_v is None else remaining_v,
            parent_id=self.parent_id if parent_id is None else parent_id,
            trace=next_trace,
        )


def initialize_root_node(
    chromosome: Any,
    base_grid: np.ndarray,
) -> BeamNode:
    """Create a root BeamNode from a chromosome and base grid."""
    return BeamNode(
        layout=base_grid,
        depth=0,
        action="root",
        metrics={},
        chromosome_signature=chromosome.as_tuple(),
        remaining_h=chromosome.h,
        remaining_v=chromosome.v,
        parent_id=None,
        trace=("root",),
    )


def initialize_direct_root_node(base_grid: np.ndarray) -> BeamNode:
    """Create a chromosome-free root BeamNode for direct Beam Search."""
    return BeamNode(
        layout=base_grid,
        depth=0,
        action="root",
        metrics={},
        chromosome_signature=None,
        remaining_h=None,
        remaining_v=None,
        parent_id=None,
        trace=("root",),
    )


__all__ = [
    "BeamNode",
    "initialize_direct_root_node",
    "initialize_root_node",
    "layout_signature",
]
