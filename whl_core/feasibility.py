"""Hard feasibility checks for completed and intermediate warehouse layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from whl_core.blocks import detect_storage_blocks
from whl_core.connectivity import access_anchor_connectivity_report, door_connectivity_report
from whl_core.constants import (
    AISLE_CODES,
    ALL_CELL_CODES,
    CELL_AISLE_CROSS,
    CELL_AISLE_H,
    CELL_AISLE_V,
    CELL_PICK,
    CELL_STORAGE,
    SERVICE_CODES,
    STORAGE_LIKE_CODES,
)
from whl_core.layout_io import fixed_aisle_mask_from_masks
from whl_core.scoring import assign_pick_face_access_sides

Cell = tuple[int, int]

VALID_GRID_CODES = set(ALL_CELL_CODES)
ASSIGNABLE_CODES = {CELL_STORAGE, CELL_PICK, *AISLE_CODES}
EXCLUDED_LAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "wall": ("wall", "walls"),
    "door": ("door", "doors"),
    "reserved": ("reserved", "reserved_zones"),
    "restricted": ("restricted", "restricted_zones"),
    "pillar": ("pillar", "pillars"),
}
ACCESS_ANCHOR_LAYER_ALIASES: tuple[str, ...] = (
    "access_anchor",
    "access_anchors",
    "staging",
    "staging_zones",
    "consolidation",
    "consolidation_zones",
    "circulation",
    "circulation_zones",
    "loading",
    "loading_zones",
    "unloading",
    "unloading_zones",
)
ALL_MASK_LAYER_ALIASES: tuple[str, ...] = (
    "walls",
    "wall",
    "doors",
    "door",
    "reserved",
    "reserved_zones",
    "restricted",
    "restricted_zones",
    "pillars",
    "pillar",
    "storage",
    "aisle",
    "aisles",
    "aisle_h",
    "aisle_v",
    "aisle_cross",
    "pick",
    "picks",
)

def _as_2d_array(grid: Any) -> np.ndarray:
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    return layout

def _mask_layer(masks: Mapping[str, Any], aliases: tuple[str, ...]) -> np.ndarray | None:
    for key in aliases:
        if key in masks:
            layer = np.asarray(masks[key])
            if layer.ndim == 2:
                return layer
    return None

def _mask_shape(masks: Mapping[str, Any]) -> tuple[int, int] | None:
    if "rows" in masks and "cols" in masks:
        try:
            return int(masks["rows"]), int(masks["cols"])
        except (TypeError, ValueError):
            return None

    for value in masks.values():
        array = np.asarray(value)
        if array.ndim == 2:
            return int(array.shape[0]), int(array.shape[1])
    return None

def _combined_mask(masks: Mapping[str, Any], aliases_by_name: Mapping[str, tuple[str, ...]]) -> np.ndarray | None:
    shape = _mask_shape(masks)
    if shape is None:
        return None

    combined = np.zeros(shape, dtype=bool)
    found = False
    for aliases in aliases_by_name.values():
        layer = _mask_layer(masks, aliases)
        if layer is None:
            continue
        if layer.shape != shape:
            raise ValueError("mask layers must share the same 2D shape.")
        combined |= layer.astype(bool)
        found = True
    return combined if found else None

def _all_binary_mask_layers(masks: Mapping[str, Any]) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    shape = _mask_shape(masks)
    if shape is None:
        return layers

    for key in ALL_MASK_LAYER_ALIASES:
        if key not in masks:
            continue
        layer = np.asarray(masks[key])
        if layer.ndim != 2:
            continue
        if layer.shape != shape:
            raise ValueError("mask layers must share the same 2D shape.")
        layers.append(layer.astype(bool))
    return layers

def validate_grid_codes(grid: np.ndarray) -> list[str]:
    """Return violations for invalid grid shape or unknown cell codes."""
    violations: list[str] = []
    layout = np.asarray(grid)
    if layout.ndim != 2:
        return ["grid_not_2d"]

    unknown_codes = sorted(set(int(value) for value in np.unique(layout)) - VALID_GRID_CODES)
    if unknown_codes:
        violations.append(f"unknown_grid_codes:{unknown_codes}")
    return violations

def check_mask_layer_overlap(masks: Mapping[str, Any] | None) -> list[str]:
    """Return violations when two mask layers are active in the same cell."""
    if masks is None:
        return []

    try:
        layers = _all_binary_mask_layers(masks)
    except ValueError as exc:
        return [f"mask_shape_mismatch:{exc}"]

    if not layers:
        return []

    occupancy = np.zeros_like(layers[0], dtype=np.uint8)
    for layer in layers:
        occupancy += layer.astype(np.uint8)

    if np.any(occupancy > 1):
        return ["mask_layer_overlap"]
    return []

def check_no_storage_or_aisle_on_excluded_cells(grid, masks: Mapping[str, Any] | None = None) -> list[str]:
    """Return violations if assignable cells overlap excluded mask layers."""
    if masks is None:
        return []

    try:
        layout = _as_2d_array(grid)
    except ValueError as exc:
        return [str(exc)]

    try:
        excluded = _combined_mask(masks, EXCLUDED_LAYER_ALIASES)
    except ValueError as exc:
        return [f"mask_shape_mismatch:{exc}"]

    if excluded is None:
        return []
    if excluded.shape != layout.shape:
        return [f"mask_grid_shape_mismatch:{excluded.shape}!={layout.shape}"]

    assignable = np.isin(layout, list(ASSIGNABLE_CODES))
    overlap = np.argwhere(excluded & assignable)
    if overlap.size == 0:
        return []

    first = tuple(int(value) for value in overlap[0])
    return [f"assignable_cell_on_excluded_mask:{first}"]

def check_isolated_storage_blocks(grid) -> list[str]:
    """Return violations for true single-cell storage blocks."""
    layout = _as_2d_array(grid)
    violations: list[str] = []
    for block in detect_storage_blocks(layout):
        if block.cell_count == 1:
            violations.append(f"isolated_storage_block:id={block.id},cells=1")
    return violations

def check_two_sided_block_depth(grid, min_depth: int = 2) -> list[str]:
    """Return violations for opposite-access blocks that are too shallow."""
    if min_depth <= 0:
        raise ValueError("min_depth must be positive.")

    layout = _as_2d_array(grid)
    violations: list[str] = []
    blocks = detect_storage_blocks(layout)
    assign_pick_face_access_sides(layout, blocks)
    for block in blocks:
        if block.has_top_bottom_access and block.height < min_depth:
            violations.append(
                f"two_sided_block_too_shallow:id={block.id},axis=TB,height={block.height}"
            )
        if block.has_left_right_access and block.width < min_depth:
            violations.append(
                f"two_sided_block_too_shallow:id={block.id},axis=LR,width={block.width}"
            )
    return violations

def _run_length(
    layout: np.ndarray,
    row: int,
    col: int,
    allowed_codes: set[int],
    axis: str,
) -> int:
    """Return contiguous run length through one cell along row or column."""
    rows, cols = layout.shape
    if axis == "vertical":
        start = row
        while start - 1 >= 0 and int(layout[start - 1, col]) in allowed_codes:
            start -= 1
        end = row
        while end + 1 < rows and int(layout[end + 1, col]) in allowed_codes:
            end += 1
        return end - start + 1

    if axis == "horizontal":
        start = col
        while start - 1 >= 0 and int(layout[row, start - 1]) in allowed_codes:
            start -= 1
        end = col
        while end + 1 < cols and int(layout[row, end + 1]) in allowed_codes:
            end += 1
        return end - start + 1

    raise ValueError("axis must be 'vertical' or 'horizontal'.")

def _oriented_run_violations(
    layout: np.ndarray,
    aisle_width: int,
    codes: set[int],
    scan_axis: str,
    exact: bool,
) -> list[str]:
    rows, cols = layout.shape
    violations: list[str] = []

    if scan_axis == "vertical":
        for col in range(cols):
            row = 0
            while row < rows:
                if int(layout[row, col]) not in codes:
                    row += 1
                    continue

                start = row
                while row + 1 < rows and int(layout[row + 1, col]) in codes:
                    row += 1
                end = row
                width = end - start + 1
                violates = width != aisle_width if exact else width < aisle_width
                if violates:
                    violations.append(
                        "horizontal_aisle_width_not_exact:"
                        f"col={col},rows={start}-{end},width={width},expected={aisle_width}"
                    )
                row += 1
        return violations

    if scan_axis == "horizontal":
        for row in range(rows):
            col = 0
            while col < cols:
                if int(layout[row, col]) not in codes:
                    col += 1
                    continue

                start = col
                while col + 1 < cols and int(layout[row, col + 1]) in codes:
                    col += 1
                end = col
                width = end - start + 1
                violates = width != aisle_width if exact else width < aisle_width
                if violates:
                    violations.append(
                        "vertical_aisle_width_not_exact:"
                        f"row={row},cols={start}-{end},width={width},expected={aisle_width}"
                    )
                col += 1
        return violations

    raise ValueError("scan_axis must be 'vertical' or 'horizontal'.")

def _fixed_aisle_mask(
    layout: np.ndarray,
    fixed_aisle_mask: np.ndarray | None,
) -> np.ndarray | None:
    if fixed_aisle_mask is None:
        return None
    mask = np.asarray(fixed_aisle_mask, dtype=bool)
    if mask.shape != layout.shape:
        raise ValueError("fixed_aisle_mask must have the same shape as grid.")
    return mask

def _layout_excluding_fixed_aisles(
    layout: np.ndarray,
    fixed_aisle_mask: np.ndarray | None,
) -> np.ndarray:
    mask = _fixed_aisle_mask(layout, fixed_aisle_mask)
    if mask is None or not np.any(mask):
        return layout
    measured = layout.copy()
    measured[mask] = CELL_STORAGE
    return measured

def _fixed_aisle_mask_from_optional_inputs(
    layout: np.ndarray,
    fixed_aisle_mask: np.ndarray | None = None,
    masks: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    if fixed_aisle_mask is not None:
        return _fixed_aisle_mask(layout, fixed_aisle_mask)
    if masks is None:
        return None
    return _fixed_aisle_mask(layout, fixed_aisle_mask_from_masks(dict(masks)))

def access_anchor_mask_from_grid_and_masks(
    grid,
    masks: Mapping[str, Any] | None = None,
    access_anchor_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return access-anchor cells, defaulting to door cells for current masks."""
    layout = _as_2d_array(grid)
    if access_anchor_mask is not None:
        mask = np.asarray(access_anchor_mask, dtype=bool)
        if mask.shape != layout.shape:
            raise ValueError("access_anchor_mask must have the same shape as grid.")
        return mask

    anchor = layout == SERVICE_CODES["door"]
    if masks is None:
        return anchor

    shape = _mask_shape(masks)
    if shape is None:
        return anchor
    if shape != layout.shape:
        raise ValueError(f"mask_grid_shape_mismatch:{shape}!={layout.shape}")

    for key in ACCESS_ANCHOR_LAYER_ALIASES:
        if key not in masks:
            continue
        layer = np.asarray(masks[key])
        if layer.ndim != 2:
            continue
        if layer.shape != layout.shape:
            raise ValueError("mask layers must share the same 2D shape.")
        anchor |= layer.astype(bool)
    return anchor

def oriented_aisle_thickness_violations(
    grid,
    aisle_width: int,
    exact: bool = True,
    fixed_aisle_mask: np.ndarray | None = None,
) -> list[str]:
    """Return run-level oriented aisle thickness diagnostics."""
    if aisle_width <= 0:
        return ["invalid_aisle_width"]

    layout = _as_2d_array(grid)
    layout = _layout_excluding_fixed_aisles(layout, fixed_aisle_mask)
    h_codes = {CELL_AISLE_H, CELL_AISLE_CROSS}
    v_codes = {CELL_AISLE_V, CELL_AISLE_CROSS}

    violations: list[str] = []
    violations.extend(
        _oriented_run_violations(
            layout,
            int(aisle_width),
            h_codes,
            "vertical",
            exact,
        )
    )
    violations.extend(
        _oriented_run_violations(
            layout,
            int(aisle_width),
            v_codes,
            "horizontal",
            exact,
        )
    )
    return violations

def check_aisle_width(
    grid,
    aisle_width: int,
    exact: bool = False,
    fixed_aisle_mask: np.ndarray | None = None,
) -> list[str]:
    """Return violations for aisle cells thinner than the prescribed width."""
    layout = _as_2d_array(grid)
    measured_layout = _layout_excluding_fixed_aisles(layout, fixed_aisle_mask)
    if exact:
        return oriented_aisle_thickness_violations(
            measured_layout,
            aisle_width,
            exact=True,
        )

    if aisle_width <= 0:
        return ["invalid_aisle_width"]
    if aisle_width == 1:
        return []

    violations: list[str] = []
    h_codes = {CELL_AISLE_H, CELL_AISLE_CROSS}
    v_codes = {CELL_AISLE_V, CELL_AISLE_CROSS}

    h_positions = np.argwhere(np.isin(measured_layout, list(h_codes)))
    for row, col in h_positions:
        if _run_length(measured_layout, int(row), int(col), h_codes, "vertical") < aisle_width:
            violations.append(f"horizontal_aisle_too_thin:cell=({int(row)},{int(col)})")
            break

    v_positions = np.argwhere(np.isin(measured_layout, list(v_codes)))
    for row, col in v_positions:
        if _run_length(measured_layout, int(row), int(col), v_codes, "horizontal") < aisle_width:
            violations.append(f"vertical_aisle_too_thin:cell=({int(row)},{int(col)})")
            break

    return violations

def check_door_connectivity(grid, require_single_component: bool = False) -> list[str]:
    """Return violations for aisle-to-door connectivity requirements."""
    report = door_connectivity_report(grid)
    aisle_components = int(report["aisle_components"])
    if aisle_components == 0:
        return ["no_aisle_components"]
    if not bool(report["has_door_connected_aisle"]):
        return ["missing_door_connected_aisle"]
    if require_single_component and aisle_components > 1:
        return ["multiple_aisle_components"]
    return []

def check_access_anchor_connectivity(
    grid,
    access_anchor_mask: np.ndarray | None = None,
    require_single_component: bool = False,
) -> list[str]:
    """Return violations for access-network reachability requirements."""
    report = access_anchor_connectivity_report(
        grid,
        access_anchor_mask=access_anchor_mask,
    )
    aisle_components = int(report["aisle_components"])
    if aisle_components == 0:
        return ["no_aisle_components"]
    if not bool(report["has_access_anchor_reachable_aisle_network"]):
        if not bool(report["has_access_anchor_connected_aisle"]):
            return ["missing_access_anchor_connected_aisle"]
        return ["unreachable_aisle_components"]
    if not bool(report["has_access_anchor_connected_aisle"]):
        return ["missing_access_anchor_connected_aisle"]
    _ = require_single_component
    return []

def check_layout_feasible(
    grid,
    masks: Mapping[str, Any] | None = None,
    aisle_width: int = 1,
    require_door_connected: bool = True,
    require_access_anchor_connected: bool | None = None,
    require_single_aisle_component: bool = False,
    enforce_aisle_width: bool = True,
    enforce_exact_aisle_width: bool = False,
    fixed_aisle_mask: np.ndarray | None = None,
    access_anchor_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return a combined hard-feasibility report for a layout grid."""
    violations: list[str] = []
    layout = _as_2d_array(grid)
    fixed_aisles = _fixed_aisle_mask_from_optional_inputs(
        layout,
        fixed_aisle_mask=fixed_aisle_mask,
        masks=masks,
    )
    require_anchor_connected = (
        bool(require_door_connected)
        if require_access_anchor_connected is None
        else bool(require_access_anchor_connected)
    )

    violations.extend(validate_grid_codes(layout))
    if violations:
        return {"is_feasible": False, "violations": violations}

    violations.extend(check_mask_layer_overlap(masks))
    violations.extend(check_no_storage_or_aisle_on_excluded_cells(layout, masks))
    violations.extend(check_isolated_storage_blocks(layout))
    violations.extend(check_two_sided_block_depth(layout))

    if enforce_aisle_width:
        violations.extend(
            check_aisle_width(
                layout,
                int(aisle_width),
                exact=bool(enforce_exact_aisle_width),
                fixed_aisle_mask=fixed_aisles,
            )
        )

    if require_anchor_connected and require_access_anchor_connected is None:
        violations.extend(
            check_door_connectivity(
                layout,
                require_single_component=require_single_aisle_component,
            )
        )
    elif require_anchor_connected:
        try:
            anchors = access_anchor_mask_from_grid_and_masks(
                layout,
                masks=masks,
                access_anchor_mask=access_anchor_mask,
            )
        except ValueError as exc:
            violations.append(str(exc))
        else:
            violations.extend(
                check_access_anchor_connectivity(
                    layout,
                    access_anchor_mask=anchors,
                    require_single_component=require_single_aisle_component,
                )
            )

    return {
        "is_feasible": not violations,
        "violations": violations,
    }

def check_child_layout_hard_feasible(
    grid,
    masks: Mapping[str, Any] | None = None,
    aisle_width: int = 1,
    enforce_exact_aisle_width: bool = False,
    fixed_aisle_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return hard feasibility for intermediate Beam Search children."""
    return check_layout_feasible(
        grid,
        masks=masks,
        aisle_width=aisle_width,
        require_door_connected=False,
        require_single_aisle_component=False,
        enforce_aisle_width=True,
        enforce_exact_aisle_width=enforce_exact_aisle_width,
        fixed_aisle_mask=fixed_aisle_mask,
    )

def check_no_forbidden_overlap(grid, masks=None) -> bool:
    """Return whether grid codes and optional masks avoid forbidden overlap."""
    return not (
        validate_grid_codes(grid)
        or check_mask_layer_overlap(masks)
        or check_no_storage_or_aisle_on_excluded_cells(grid, masks)
    )

def check_has_door_connected_aisle(grid) -> bool:
    """Return whether at least one aisle component is connected to a door."""
    return bool(door_connectivity_report(grid)["has_door_connected_aisle"])

def check_storage_blocks_valid(grid, min_block_cells: int = 2) -> bool:
    """Legacy basic storage-block validity wrapper."""
    if min_block_cells <= 0:
        raise ValueError("min_block_cells must be positive.")
    blocks = detect_storage_blocks(grid)
    return all(block.cell_count >= min_block_cells for block in blocks)

__all__ = [
    "VALID_GRID_CODES",
    "access_anchor_mask_from_grid_and_masks",
    "check_aisle_width",
    "check_access_anchor_connectivity",
    "check_child_layout_hard_feasible",
    "check_door_connectivity",
    "check_has_door_connected_aisle",
    "check_isolated_storage_blocks",
    "check_layout_feasible",
    "check_mask_layer_overlap",
    "check_no_forbidden_overlap",
    "check_no_storage_or_aisle_on_excluded_cells",
    "check_storage_blocks_valid",
    "check_two_sided_block_depth",
    "oriented_aisle_thickness_violations",
    "validate_grid_codes",
]
