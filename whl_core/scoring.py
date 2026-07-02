"""Scoring metrics for completed warehouse layout grids."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from whl_core.blocks import StorageBlock, detect_storage_blocks
from whl_core.connectivity import door_connectivity_report
from whl_core.constants import AISLE_CODES, CELL_PICK, CELL_STORAGE, SERVICE_CODES

Cell = tuple[int, int]
SideName = Literal["top", "bottom", "left", "right"]
AccessAxis = Literal["TB", "LR"]

SIDE_NAMES: tuple[SideName, ...] = ("top", "bottom", "left", "right")
PICK_ADJACENCY_CODES = {*AISLE_CODES, SERVICE_CODES["door"]}
STORAGE_LIKE_CODES = {CELL_STORAGE, CELL_PICK}

def _neighbor_for_side(row: int, col: int, side: SideName) -> Cell:
    if side == "top":
        return (row - 1, col)
    if side == "bottom":
        return (row + 1, col)
    if side == "left":
        return (row, col - 1)
    if side == "right":
        return (row, col + 1)
    raise ValueError("side must be one of: top, bottom, left, right.")

def _inside(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols

def _pick_faces_on_side(layout: np.ndarray, block: StorageBlock, side: SideName) -> list[Cell]:
    """Return block cells exposed to access from one specific side."""
    rows, cols = layout.shape
    exposed: list[Cell] = []

    for row, col in block.cells:
        if int(layout[row, col]) not in STORAGE_LIKE_CODES:
            continue

        next_row, next_col = _neighbor_for_side(row, col, side)
        if not _inside(next_row, next_col, rows, cols):
            continue
        if int(layout[next_row, next_col]) in PICK_ADJACENCY_CODES:
            exposed.append((row, col))

    return exposed

def _side_faces_by_block(layout: np.ndarray, block: StorageBlock) -> dict[SideName, list[Cell]]:
    return {side: _pick_faces_on_side(layout, block, side) for side in SIDE_NAMES}

def _choose_access_axis_from_counts(
    block: StorageBlock,
    top_bottom_count: int,
    left_right_count: int,
) -> AccessAxis | None:
    """Choose the axis used for pick-face and depth evaluation."""
    if top_bottom_count <= 0 and left_right_count <= 0:
        return None

    if block.orientation == "H":
        return "TB" if top_bottom_count > 0 else "LR"

    if block.orientation == "V":
        return "LR" if left_right_count > 0 else "TB"

    if top_bottom_count >= left_right_count:
        return "TB"
    return "LR"

def _choose_access_axis(
    block: StorageBlock,
    side_faces: dict[SideName, list[Cell]],
) -> AccessAxis | None:
    top_bottom_count = len(side_faces["top"]) + len(side_faces["bottom"])
    left_right_count = len(side_faces["left"]) + len(side_faces["right"])
    return _choose_access_axis_from_counts(block, top_bottom_count, left_right_count)

def _choose_access_axis_from_block(block: StorageBlock) -> AccessAxis | None:
    """Fallback axis selection using only block.access_side_names."""
    access_side_names = set(getattr(block, "access_side_names", frozenset()))
    top_bottom_count = int("top" in access_side_names) + int("bottom" in access_side_names)
    left_right_count = int("left" in access_side_names) + int("right" in access_side_names)
    return _choose_access_axis_from_counts(block, top_bottom_count, left_right_count)

def selected_access_sides_for_block(
    grid,
    block: StorageBlock,
) -> frozenset[SideName]:
    """Return the selected access side names used for scoring one block."""
    layout = np.asarray(grid)
    assign_pick_face_access_sides(layout, [block])
    return frozenset(block.pick_face_side_names)

def _choose_pick_face_group(
    block: StorageBlock,
    side_faces: dict[SideName, list[Cell]],
) -> list[Cell]:
    """Choose pick faces along one selected access axis."""
    axis = _choose_access_axis(block, side_faces)

    if axis == "TB":
        selected = side_faces["top"] + side_faces["bottom"]
    elif axis == "LR":
        selected = side_faces["left"] + side_faces["right"]
    else:
        selected = []

    return sorted(set(selected))

def _pick_face_sides_from_axis(
    axis: AccessAxis | None,
    side_faces: dict[SideName, list[Cell]],
) -> frozenset[SideName]:
    if axis == "TB":
        return frozenset(side for side in ("top", "bottom") if side_faces[side])
    if axis == "LR":
        return frozenset(side for side in ("left", "right") if side_faces[side])
    return frozenset()

def assign_pick_face_access_sides(
    grid,
    blocks: list[StorageBlock],
) -> list[StorageBlock]:
    """Assign pick-face-derived effective access sides to storage blocks."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    for block in blocks:
        side_faces = _side_faces_by_block(layout, block)
        axis = _choose_access_axis(block, side_faces)
        pick_faces = _choose_pick_face_group(block, side_faces)
        pick_face_sides = _pick_face_sides_from_axis(axis, side_faces)
        block.pick_faces = pick_faces
        block.pick_face_side_names = pick_face_sides
        block.access_side_names = pick_face_sides
    return blocks

def detect_pick_faces(grid, blocks: list[StorageBlock] | None = None) -> set[Cell]:
    """Detect storage cells acting as pick faces under one access axis per block."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    detected: set[Cell] = set(zip(*np.where(layout == CELL_PICK), strict=True))

    if blocks is None:
        blocks = detect_storage_blocks(layout)

    assign_pick_face_access_sides(layout, blocks)
    for block in blocks:
        detected.update(block.pick_faces)

    return detected

def count_storage_total(grid) -> int:
    """Count total storage-capable cells, including explicit pick-face cells."""
    layout = np.asarray(grid)
    return int(np.isin(layout, list(STORAGE_LIKE_CODES)).sum())

def count_pick_faces(grid) -> int:
    """Count detected and explicitly marked pick-face cells."""
    return len(detect_pick_faces(grid))

def compute_pick_face_mask(layout: np.ndarray) -> np.ndarray:
    """Return a boolean mask of official pick-face cells."""
    grid = np.asarray(layout)
    if grid.ndim != 2:
        raise ValueError("layout must be a 2D array.")
    mask = np.zeros(grid.shape, dtype=bool)
    for row, col in detect_pick_faces(grid):
        mask[row, col] = True
    return mask

def compute_storage_mask(layout: np.ndarray) -> np.ndarray:
    """Return a boolean mask of storage-capable cells."""
    grid = np.asarray(layout)
    if grid.ndim != 2:
        raise ValueError("layout must be a 2D array.")
    return np.isin(grid, list(STORAGE_LIKE_CODES))

def compute_aisle_mask(layout: np.ndarray) -> np.ndarray:
    """Return a boolean mask of aisle cells, including H/V/CROSS codes."""
    grid = np.asarray(layout)
    if grid.ndim != 2:
        raise ValueError("layout must be a 2D array.")
    return np.isin(grid, list(AISLE_CODES))

def count_interior_storage(grid) -> int:
    """Count storage-capable cells that are not exposed as pick faces."""
    layout = np.asarray(grid)
    pick_faces = detect_pick_faces(layout)
    storage_cells = set(zip(*np.where(np.isin(layout, list(STORAGE_LIKE_CODES))), strict=True))
    return len(storage_cells - pick_faces)

def compute_block_depth(block: StorageBlock, grid=None) -> int:
    """Return effective retrieval depth for a storage block."""
    if grid is None:
        axis = _choose_access_axis_from_block(block)
        selected_sides = set(getattr(block, "access_side_names", frozenset()))
    else:
        layout = np.asarray(grid)
        assign_pick_face_access_sides(layout, [block])
        selected_sides = set(block.pick_face_side_names)
        axis = _choose_access_axis(block, _side_faces_by_block(layout, block))

    if axis == "TB":
        raw_depth = block.height
        two_sided = {"top", "bottom"} <= selected_sides
    elif axis == "LR":
        raw_depth = block.width
        two_sided = {"left", "right"} <= selected_sides
    else:
        raw_depth = max(block.height, block.width)
        two_sided = False

    depth = math.ceil(raw_depth / 2) if two_sided and raw_depth > 1 else raw_depth
    return max(1, int(depth))

def _block_lane_width(block: StorageBlock, grid=None) -> int:
    """Return the number of storage lanes parallel to the selected pick face."""
    if grid is None:
        axis = _choose_access_axis_from_block(block)
    else:
        layout = np.asarray(grid)
        assign_pick_face_access_sides(layout, [block])
        axis = _choose_access_axis(block, _side_faces_by_block(layout, block))

    if axis == "TB":
        return max(1, int(block.width))
    if axis == "LR":
        return max(1, int(block.height))

    return max(1, int(min(block.height, block.width)))

def compute_retrieval_penalty(
    grid,
    blocks: list[StorageBlock] | None = None,
) -> float:
    """Compute block-depth retrieval penalty for completed layout grids."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    if blocks is None:
        blocks = detect_storage_blocks(layout)
    assign_pick_face_access_sides(layout, blocks)

    penalty = 0.0
    for block in blocks:
        depth = compute_block_depth(block, layout)
        lane_width = _block_lane_width(block, layout)
        penalty += lane_width * ((depth - 1) ** 2)

    return float(penalty)

def score_layout(grid) -> dict[str, bool | int | float]:
    """Return manuscript-compatible metrics for one completed layout grid."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    blocks = detect_storage_blocks(layout)
    pick_faces = detect_pick_faces(layout, blocks)
    connectivity = door_connectivity_report(layout)
    storage_total = count_storage_total(layout)
    interior_storage = storage_total - len(pick_faces)
    retrieval_penalty = compute_retrieval_penalty(layout, blocks)

    metrics: dict[str, bool | int | float] = {
        "storage_total": storage_total,
        "pick_faces": len(pick_faces),
        "interior_storage": int(interior_storage),
        "retrieval_penalty": retrieval_penalty,
        "door_connectivity_index": connectivity["door_connectivity_index"],
        "access_anchor_connectivity_index": connectivity[
            "access_anchor_connectivity_index"
        ],
        "aisle_components": connectivity["aisle_components"],
        "has_door_connected_aisle": connectivity["has_door_connected_aisle"],
        "has_access_anchor_connected_aisle": connectivity[
            "has_access_anchor_connected_aisle"
        ],
        "anchor_connected_components": connectivity["anchor_connected_components"],
        "unanchored_aisle_components": connectivity["unanchored_aisle_components"],
        "single_aisle_component": connectivity["single_aisle_component"],
        "access_network_components": connectivity["access_network_components"],
        "aisle_access_network_components": connectivity[
            "aisle_access_network_components"
        ],
        "unreachable_aisle_components": connectivity["unreachable_aisle_components"],
        "unreachable_aisle_cells": connectivity["unreachable_aisle_cells"],
        "has_access_anchor_reachable_aisle_network": connectivity[
            "has_access_anchor_reachable_aisle_network"
        ],
    }

    metrics["storage_locked"] = metrics["interior_storage"]
    metrics["RetrievalPenalty"] = metrics["retrieval_penalty"]
    return metrics

__all__ = [
    "assign_pick_face_access_sides",
    "compute_block_depth",
    "compute_aisle_mask",
    "compute_pick_face_mask",
    "compute_retrieval_penalty",
    "compute_storage_mask",
    "count_interior_storage",
    "count_pick_faces",
    "count_storage_total",
    "detect_pick_faces",
    "score_layout",
    "selected_access_sides_for_block",
]
