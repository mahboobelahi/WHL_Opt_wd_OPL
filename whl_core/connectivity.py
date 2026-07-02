"""Connectivity checks for aisle networks and door access."""

from __future__ import annotations

from collections import deque

import numpy as np

from whl_core.constants import AISLE_CODES, SERVICE_CODES

Cell = tuple[int, int]

def _neighbors(row: int, col: int, rows: int, cols: int) -> tuple[Cell, ...]:
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return tuple((r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols)

def _is_aisle(value: int) -> bool:
    return int(value) in AISLE_CODES

def _network_mask(layout: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    return np.isin(layout, list(AISLE_CODES)) | anchor_mask

def find_aisle_components(grid) -> list[list[Cell]]:
    """Return 4-neighbor connected components of all aisle codes."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    rows, cols = layout.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components: list[list[Cell]] = []

    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not _is_aisle(int(layout[start_row, start_col])):
                continue

            queue: deque[Cell] = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            component: list[Cell] = []

            while queue:
                row, col = queue.popleft()
                component.append((row, col))

                for next_row, next_col in _neighbors(row, col, rows, cols):
                    if visited[next_row, next_col]:
                        continue
                    if not _is_aisle(int(layout[next_row, next_col])):
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            components.append(sorted(component))

    return components

def component_adjacent_to_door(grid, component_cells) -> bool:
    """Return whether any component cell is 4-neighbor adjacent to a door."""
    layout = np.asarray(grid)
    door_mask = layout == SERVICE_CODES["door"]
    return component_adjacent_to_anchor_mask(layout, component_cells, door_mask)

def component_adjacent_to_anchor_mask(grid, component_cells, access_anchor_mask) -> bool:
    """Return whether any component cell is 4-neighbor adjacent to an access anchor."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    anchor_mask = np.asarray(access_anchor_mask, dtype=bool)
    if anchor_mask.shape != layout.shape:
        raise ValueError("access_anchor_mask must have the same shape as grid.")

    rows, cols = layout.shape

    for row, col in component_cells:
        for next_row, next_col in _neighbors(row, col, rows, cols):
            if anchor_mask[next_row, next_col]:
                return True
    return False

def access_anchor_connectivity_report(
    grid,
    access_anchor_mask=None,
) -> dict[str, bool | int | float]:
    """Return access-network connectivity diagnostics for a completed grid."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    if access_anchor_mask is None:
        anchor_mask = layout == SERVICE_CODES["door"]
    else:
        anchor_mask = np.asarray(access_anchor_mask, dtype=bool)
        if anchor_mask.shape != layout.shape:
            raise ValueError("access_anchor_mask must have the same shape as grid.")

    aisle_components = find_aisle_components(layout)
    rows, cols = layout.shape
    network = _network_mask(layout, anchor_mask)
    visited = np.zeros((rows, cols), dtype=bool)
    network_component_ids = np.full((rows, cols), -1, dtype=int)
    access_network_components = 0
    aisle_access_network_components = 0
    anchor_reachable_network_ids: set[int] = set()

    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not network[start_row, start_col]:
                continue

            component_id = access_network_components
            access_network_components += 1
            queue: deque[Cell] = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            has_aisle = False
            has_anchor = False

            while queue:
                row, col = queue.popleft()
                network_component_ids[row, col] = component_id
                has_aisle = has_aisle or _is_aisle(int(layout[row, col]))
                has_anchor = has_anchor or bool(anchor_mask[row, col])

                for next_row, next_col in _neighbors(row, col, rows, cols):
                    if visited[next_row, next_col] or not network[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            if has_aisle:
                aisle_access_network_components += 1
            if has_aisle and has_anchor:
                anchor_reachable_network_ids.add(component_id)

    reachable_raw_components = 0
    unreachable_aisle_cells = 0
    for component in aisle_components:
        ids = {
            int(network_component_ids[row, col])
            for row, col in component
            if network_component_ids[row, col] >= 0
        }
        if ids and ids <= anchor_reachable_network_ids:
            reachable_raw_components += 1
        else:
            unreachable_aisle_cells += len(component)

    total = len(aisle_components)
    unreachable_aisle_components = total - reachable_raw_components
    index = reachable_raw_components / total if total else 0.0

    return {
        "has_access_anchor_connected_aisle": reachable_raw_components > 0,
        "has_access_anchor_reachable_aisle_network": total > 0
        and unreachable_aisle_components == 0,
        "aisle_components": total,
        "anchor_connected_components": reachable_raw_components,
        "unanchored_aisle_components": unreachable_aisle_components,
        "access_network_components": access_network_components,
        "aisle_access_network_components": aisle_access_network_components,
        "unreachable_aisle_components": unreachable_aisle_components,
        "unreachable_aisle_cells": unreachable_aisle_cells,
        "single_aisle_component": total == 1,
        "access_anchor_connectivity_index": float(index),
    }

def door_connectivity_report(grid) -> dict[str, bool | int | float]:
    """Return aisle-to-door connectivity diagnostics for a completed grid."""
    report = access_anchor_connectivity_report(grid)
    return {
        "has_door_connected_aisle": report["has_access_anchor_connected_aisle"],
        "has_access_anchor_connected_aisle": report["has_access_anchor_connected_aisle"],
        "aisle_components": report["aisle_components"],
        "door_connected_components": report["anchor_connected_components"],
        "anchor_connected_components": report["anchor_connected_components"],
        "unanchored_aisle_components": report["unanchored_aisle_components"],
        "access_network_components": report["access_network_components"],
        "aisle_access_network_components": report["aisle_access_network_components"],
        "unreachable_aisle_components": report["unreachable_aisle_components"],
        "unreachable_aisle_cells": report["unreachable_aisle_cells"],
        "has_access_anchor_reachable_aisle_network": report[
            "has_access_anchor_reachable_aisle_network"
        ],
        "single_aisle_component": report["single_aisle_component"],
        "door_connectivity_index": report["access_anchor_connectivity_index"],
        "access_anchor_connectivity_index": report["access_anchor_connectivity_index"],
    }

__all__ = [
    "access_anchor_connectivity_report",
    "component_adjacent_to_anchor_mask",
    "component_adjacent_to_door",
    "door_connectivity_report",
    "find_aisle_components",
]
