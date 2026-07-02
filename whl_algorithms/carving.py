"""Aisle-carving helpers for Beam Search decoding."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from whl_core.blocks import StorageBlock, detect_storage_blocks
from whl_core.constants import (
    AISLE_CODES,
    CELL_AISLE_CROSS,
    CELL_AISLE_H,
    CELL_AISLE_V,
    CELL_PICK,
    CELL_STORAGE,
    SERVICE_CODES,
    is_storage_like_code,
)
from whl_core.scoring import assign_pick_face_access_sides

from whl_algorithms.beam_node import BeamNode, layout_signature

PROTECTED_CODES = set(SERVICE_CODES.values())


def is_protected_cell(value: int) -> bool:
    """Return whether a cell is fixed infrastructure that cannot be carved."""
    return int(value) in PROTECTED_CODES


def is_carvable_cell(value: int) -> bool:
    """Return whether a cell can participate in aisle carving."""
    code = int(value)
    return code in {CELL_STORAGE, CELL_PICK, *AISLE_CODES}

def is_storage_eligible_cell(value: int) -> bool:
    """Return whether a cell belongs to the searchable layout area."""
    code = int(value)
    return code in {CELL_STORAGE, CELL_PICK, *AISLE_CODES}


def storage_eligible_mask_from_layout(grid: np.ndarray) -> np.ndarray:
    """Return a boolean mask for cells that belong to the searchable area."""
    layout = _validate_grid(grid)
    eligible_codes = {CELL_STORAGE, CELL_PICK, *AISLE_CODES}
    return np.isin(layout, list(eligible_codes))



def apply_aisle_code(existing_value: int, orientation: str) -> int:
    """Return the aisle code produced by carving over an existing cell."""
    code = int(existing_value)
    if is_protected_cell(code):
        raise ValueError("protected cells cannot be converted to aisle cells.")
    if orientation not in {"H", "V"}:
        raise ValueError("orientation must be 'H' or 'V'.")
    if not is_carvable_cell(code):
        raise ValueError(f"cell code {code} is not carvable.")

    if orientation == "H":
        if code in {CELL_AISLE_V, CELL_AISLE_CROSS}:
            return CELL_AISLE_CROSS
        return CELL_AISLE_H

    if code in {CELL_AISLE_H, CELL_AISLE_CROSS}:
        return CELL_AISLE_CROSS
    return CELL_AISLE_V


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    return layout


def _span_between_carvable_cells(codes: np.ndarray) -> tuple[int, int] | None:
    """Return the inclusive span between first and last carvable cell."""
    open_positions = [idx for idx, value in enumerate(codes) if is_carvable_cell(int(value))]
    if not open_positions:
        return None
    return min(open_positions), max(open_positions)


def _row_patch_span(layout: np.ndarray, row: int) -> tuple[int, int] | None:
    span = _span_between_carvable_cells(layout[row, :])
    if span is None:
        return None
    cmin, cmax = span
    values = layout[row, cmin : cmax + 1]
    if any(is_protected_cell(int(value)) for value in values):
        return None
    if any(not is_carvable_cell(int(value)) for value in values):
        return None
    return cmin, cmax


def _row_patch_cells(layout: np.ndarray, row: int) -> list[tuple[int, int]] | None:
    span = _row_patch_span(layout, row)
    if span is None:
        return None
    cmin, cmax = span
    return [(row, col) for col in range(cmin, cmax + 1)]


def _column_patch_span(layout: np.ndarray, col: int) -> tuple[int, int] | None:
    span = _span_between_carvable_cells(layout[:, col])
    if span is None:
        return None
    rmin, rmax = span
    values = layout[rmin : rmax + 1, col]
    if any(is_protected_cell(int(value)) for value in values):
        return None
    if any(not is_carvable_cell(int(value)) for value in values):
        return None
    return rmin, rmax


def _column_patch_cells(layout: np.ndarray, col: int) -> list[tuple[int, int]] | None:
    span = _column_patch_span(layout, col)
    if span is None:
        return None
    rmin, rmax = span
    return [(row, col) for row in range(rmin, rmax + 1)]


def global_horizontal_band_cells(
    grid: np.ndarray,
    start_row: int,
    aisle_width: int,
) -> list[tuple[int, int]] | None:
    """Return cells for a valid global horizontal aisle band."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if start_row < 0 or start_row + aisle_width > layout.shape[0]:
        return None

    spans: list[tuple[int, int]] = []
    for row in range(start_row, start_row + aisle_width):
        row_span = _row_patch_span(layout, row)
        if row_span is None:
            return None
        spans.append(row_span)
    if len(set(spans)) != 1:
        return None

    cmin, cmax = spans[0]
    values = layout[start_row : start_row + aisle_width, cmin : cmax + 1]
    if any(is_protected_cell(int(value)) for value in values.ravel()):
        return None
    if any(not is_carvable_cell(int(value)) for value in values.ravel()):
        return None
    return [
        (row, col)
        for row in range(start_row, start_row + aisle_width)
        for col in range(cmin, cmax + 1)
    ]


def global_vertical_band_cells(
    grid: np.ndarray,
    start_col: int,
    aisle_width: int,
) -> list[tuple[int, int]] | None:
    """Return cells for a valid global vertical aisle band."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if start_col < 0 or start_col + aisle_width > layout.shape[1]:
        return None

    spans: list[tuple[int, int]] = []
    for col in range(start_col, start_col + aisle_width):
        col_span = _column_patch_span(layout, col)
        if col_span is None:
            return None
        spans.append(col_span)
    if len(set(spans)) != 1:
        return None

    rmin, rmax = spans[0]
    values = layout[rmin : rmax + 1, start_col : start_col + aisle_width]
    if any(is_protected_cell(int(value)) for value in values.ravel()):
        return None
    if any(not is_carvable_cell(int(value)) for value in values.ravel()):
        return None
    return [
        (row, col)
        for row in range(rmin, rmax + 1)
        for col in range(start_col, start_col + aisle_width)
    ]


def _apply_band(
    grid: np.ndarray,
    cells: Iterable[tuple[int, int]],
    orientation: str,
) -> np.ndarray:
    carved = grid.copy()
    for row, col in cells:
        carved[row, col] = apply_aisle_code(int(carved[row, col]), orientation)
    return carved


def carve_global_horizontal(
    grid: np.ndarray,
    start_row: int,
    aisle_width: int,
) -> np.ndarray | None:
    """Carve one global horizontal aisle band without mutating ``grid``."""
    layout = _validate_grid(grid)
    cells = global_horizontal_band_cells(layout, start_row, aisle_width)
    if cells is None:
        return None
    return _apply_band(layout, cells, "H")


def carve_global_vertical(
    grid: np.ndarray,
    start_col: int,
    aisle_width: int,
) -> np.ndarray | None:
    """Carve one global vertical aisle band without mutating ``grid``."""
    layout = _validate_grid(grid)
    cells = global_vertical_band_cells(layout, start_col, aisle_width)
    if cells is None:
        return None
    return _apply_band(layout, cells, "V")


def find_feasible_global_horizontal_starts(
    grid: np.ndarray,
    aisle_width: int,
) -> list[int]:
    """Return all feasible global horizontal start rows for ``aisle_width``."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    return [
        row
        for row in range(0, layout.shape[0] - aisle_width + 1)
        if global_horizontal_band_cells(layout, row, aisle_width) is not None
    ]


def find_feasible_global_vertical_starts(
    grid: np.ndarray,
    aisle_width: int,
) -> list[int]:
    """Return all feasible global vertical start columns for ``aisle_width``."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    return [
        col
        for col in range(0, layout.shape[1] - aisle_width + 1)
        if global_vertical_band_cells(layout, col, aisle_width) is not None
    ]


def has_aisle_in_span(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    """Return whether any proposed carve cell already contains an aisle code."""
    layout = _validate_grid(grid)
    return any(int(layout[row, col]) in AISLE_CODES for row, col in cells)


def same_orientation_neighbor_span_has_aisle(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    orientation: str,
) -> bool:
    """Return whether a proposed band touches a same-orientation aisle."""
    layout = _validate_grid(grid)
    if not cells:
        return False
    if orientation not in {"H", "V"}:
        raise ValueError("orientation must be 'H' or 'V'.")

    row_values = [row for row, _ in cells]
    col_values = [col for _, col in cells]
    rmin = min(row_values)
    rmax = max(row_values)
    cmin = min(col_values)
    cmax = max(col_values)

    if orientation == "H":
        same_orientation_codes = {CELL_AISLE_H, CELL_AISLE_CROSS}
        for neighbor_row in (rmin - 1, rmax + 1):
            if 0 <= neighbor_row < layout.shape[0]:
                span = layout[neighbor_row, cmin : cmax + 1]
                if np.isin(span, list(same_orientation_codes)).any():
                    return True
        return False

    same_orientation_codes = {CELL_AISLE_V, CELL_AISLE_CROSS}
    for neighbor_col in (cmin - 1, cmax + 1):
        if 0 <= neighbor_col < layout.shape[1]:
            span = layout[rmin : rmax + 1, neighbor_col]
            if np.isin(span, list(same_orientation_codes)).any():
                return True
    return False


def is_span_protected(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    """Return whether any proposed carve cell is protected infrastructure."""
    layout = _validate_grid(grid)
    return any(is_protected_cell(int(layout[row, col])) for row, col in cells)


def block_fragment_valid_after_carve(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    min_fragment_size: int = 2,
) -> bool:
    """Return whether storage fragments remain valid after a block carve."""
    before = _validate_grid(grid_before)
    after = _validate_grid(grid_after)
    if before.shape != after.shape:
        raise ValueError("grid_before and grid_after must have the same shape.")
    if min_fragment_size <= 0:
        raise ValueError("min_fragment_size must be positive.")

    blocks = detect_storage_blocks(after)
    if not blocks:
        return False
    for block in blocks:
        if len(block.cells) < min_fragment_size:
            return False
        if block.height <= 1 or block.width <= 1:
            return False
    return True


def candidate_horizontal_band_cells(
    block: StorageBlock,
    start_row: int,
    aisle_width: int,
) -> list[tuple[int, int]]:
    """Return the rectangular horizontal band inside a block bounding box."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if start_row < block.rmin or start_row + aisle_width - 1 > block.rmax:
        return []
    return [
        (row, col)
        for row in range(start_row, start_row + aisle_width)
        for col in range(block.cmin, block.cmax + 1)
    ]


def candidate_vertical_band_cells(
    block: StorageBlock,
    start_col: int,
    aisle_width: int,
) -> list[tuple[int, int]]:
    """Return the rectangular vertical band inside a block bounding box."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if start_col < block.cmin or start_col + aisle_width - 1 > block.cmax:
        return []
    return [
        (row, col)
        for row in range(block.rmin, block.rmax + 1)
        for col in range(start_col, start_col + aisle_width)
    ]


def _is_good_fragment_split(
    block: StorageBlock,
    start: int,
    aisle_width: int,
    orientation: str,
    min_fragment_size: int,
) -> bool:
    """Pre-check fragment thickness caused by a proposed carve."""
    if orientation == "H":
        upper = start - block.rmin
        lower = block.rmax - (start + aisle_width - 1)
        return all(size == 0 or size >= min_fragment_size for size in (upper, lower))
    if orientation == "V":
        left = start - block.cmin
        right = block.cmax - (start + aisle_width - 1)
        return all(size == 0 or size >= min_fragment_size for size in (left, right))
    raise ValueError("orientation must be 'H' or 'V'.")


def try_carve_block_horizontal(
    grid: np.ndarray,
    block: StorageBlock,
    start_row: int,
    aisle_width: int,
    min_fragment_size: int = 2,
) -> np.ndarray | None:
    """Try to carve a horizontal aisle band inside one storage block."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if min_fragment_size <= 0:
        raise ValueError("min_fragment_size must be positive.")
    if not _is_good_fragment_split(block, start_row, aisle_width, "H", min_fragment_size):
        return None

    cells = candidate_horizontal_band_cells(block, start_row, aisle_width)
    if not cells:
        return None
    if is_span_protected(layout, cells) or has_aisle_in_span(layout, cells):
        return None
    if same_orientation_neighbor_span_has_aisle(layout, cells, "H"):
        return None
    if not all(is_storage_like_code(int(layout[row, col])) for row, col in cells):
        return None

    carved = _apply_band(layout, cells, "H")
    if not block_fragment_valid_after_carve(
        layout,
        carved,
        min_fragment_size=min_fragment_size,
    ):
        return None
    return carved


def try_carve_block_vertical(
    grid: np.ndarray,
    block: StorageBlock,
    start_col: int,
    aisle_width: int,
    min_fragment_size: int = 2,
) -> np.ndarray | None:
    """Try to carve a vertical aisle band inside one storage block."""
    layout = _validate_grid(grid)
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if min_fragment_size <= 0:
        raise ValueError("min_fragment_size must be positive.")
    if not _is_good_fragment_split(block, start_col, aisle_width, "V", min_fragment_size):
        return None

    cells = candidate_vertical_band_cells(block, start_col, aisle_width)
    if not cells:
        return None
    if is_span_protected(layout, cells) or has_aisle_in_span(layout, cells):
        return None
    if same_orientation_neighbor_span_has_aisle(layout, cells, "V"):
        return None
    if not all(is_storage_like_code(int(layout[row, col])) for row, col in cells):
        return None

    carved = _apply_band(layout, cells, "V")
    if not block_fragment_valid_after_carve(
        layout,
        carved,
        min_fragment_size=min_fragment_size,
    ):
        return None
    return carved


def candidate_horizontal_start_rows(
    block: StorageBlock,
    aisle_width: int,
    step: int | None = None,
    spacing_offset: int = 0,
) -> list[int]:
    """Return deterministic horizontal band start rows inside a block."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if spacing_offset < 0:
        raise ValueError("spacing_offset must be non-negative.")
    if block.height <= aisle_width + 1:
        return []
    selected_step = max(1, aisle_width // 2) if step is None else step
    if selected_step <= 0:
        raise ValueError("step must be positive.")

    start_min = block.rmin + 2 if block.height >= aisle_width + 4 else block.rmin + 1
    start_max = block.rmax - aisle_width + 1
    if start_min > start_max:
        return []
    if spacing_offset:
        start_min = min(start_min + (spacing_offset % selected_step), start_max)
    return list(range(start_min, start_max + 1, selected_step))


def candidate_vertical_start_cols(
    block: StorageBlock,
    aisle_width: int,
    step: int | None = None,
    spacing_offset: int = 0,
) -> list[int]:
    """Return deterministic vertical band start columns inside a block."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if spacing_offset < 0:
        raise ValueError("spacing_offset must be non-negative.")
    if block.width <= aisle_width + 1:
        return []
    selected_step = max(1, aisle_width // 2) if step is None else step
    if selected_step <= 0:
        raise ValueError("step must be positive.")

    start_min = block.cmin + 2 if block.width >= aisle_width + 4 else block.cmin + 1
    start_max = block.cmax - aisle_width + 1
    if start_min > start_max:
        return []
    if spacing_offset:
        start_min = min(start_min + (spacing_offset % selected_step), start_max)
    return list(range(start_min, start_max + 1, selected_step))


def is_block_too_small_for_carving(block: StorageBlock, aisle_width: int) -> bool:
    """Return whether a block is too small to carve safely."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    return min(int(block.height), int(block.width)) <= int(aisle_width)


def _block_depth_dimension_for_access(block: StorageBlock) -> int:
    """Return the block depth dimension implied by its access axis."""
    if getattr(block, "has_top_bottom_access", False):
        return int(block.height)
    if getattr(block, "has_left_right_access", False):
        return int(block.width)
    return int(min(block.height, block.width))


def is_shallow_block(block: StorageBlock, aisle_width: int) -> bool:
    """Return whether a block is too shallow for non-repair deep carving."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if not getattr(block, "has_access", True):
        return True
    return _block_depth_dimension_for_access(block) <= int(aisle_width) + 1


def block_spans_eligible_height(block: StorageBlock, grid: np.ndarray) -> bool:
    """Return whether ``block`` spans the eligible height in its column band."""
    eligible = storage_eligible_mask_from_layout(grid)
    if block.cmin < 0 or block.cmax >= eligible.shape[1]:
        return False
    band = eligible[:, block.cmin : block.cmax + 1]
    rows = np.where(np.any(band, axis=1))[0]
    if rows.size == 0:
        return False
    return bool(block.rmin == int(rows.min()) and block.rmax == int(rows.max()))


def block_spans_eligible_width(block: StorageBlock, grid: np.ndarray) -> bool:
    """Return whether ``block`` spans the eligible width in its row band."""
    eligible = storage_eligible_mask_from_layout(grid)
    if block.rmin < 0 or block.rmax >= eligible.shape[0]:
        return False
    band = eligible[block.rmin : block.rmax + 1, :]
    cols = np.where(np.any(band, axis=0))[0]
    if cols.size == 0:
        return False
    return bool(block.cmin == int(cols.min()) and block.cmax == int(cols.max()))


def requires_full_span_repair(block: StorageBlock, grid: np.ndarray) -> bool:
    """Return whether a two-sided block needs full-span repair carving."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("requires_full_span_repair expects a layout grid, not a shape tuple.")
    assign_pick_face_access_sides(layout, [block])
    if not bool(getattr(block, "is_two_sided", False)):
        return False
    return bool(
        block_spans_eligible_height(block, layout)
        or block_spans_eligible_width(block, layout)
    )


def _centered_start_order(start_min: int, start_max: int, center: int) -> list[int]:
    """Return bounded starts ordered by distance from ``center``."""
    if start_min > start_max:
        return []
    starts = list(range(start_min, start_max + 1))
    return sorted(starts, key=lambda value: (abs(value - center), value))


def candidate_horizontal_repair_start_rows(
    block: StorageBlock,
    aisle_width: int,
) -> list[int]:
    """Return midpoint-first horizontal repair starts inside a block."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    start_min = block.rmin
    start_max = block.rmax - aisle_width + 1
    center = block.rmin + max(0, (block.height - aisle_width) // 2)
    return _centered_start_order(start_min, start_max, center)


def candidate_vertical_repair_start_cols(
    block: StorageBlock,
    aisle_width: int,
) -> list[int]:
    """Return midpoint-first vertical repair starts inside a block."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    start_min = block.cmin
    start_max = block.cmax - aisle_width + 1
    center = block.cmin + max(0, (block.width - aisle_width) // 2)
    return _centered_start_order(start_min, start_max, center)


def choose_repair_carve_orientations(block: StorageBlock) -> list[str]:
    """Choose repair orientations opposite to the long block orientation."""
    if block.orientation == "H":
        return ["V"]
    if block.orientation == "V":
        return ["H"]
    return ["H", "V"]


def choose_block_carve_orientations(
    block: StorageBlock,
    repair_required: bool = False,
    carve_rule: dict | None = None,
) -> list[str]:
    """Choose block-level carve orientations while preserving repair behavior."""
    if repair_required:
        return choose_repair_carve_orientations(block)
    if bool(getattr(block, "is_two_sided", False)):
        if block.orientation == "H":
            return ["H"]
        if block.orientation == "V":
            return ["V"]
        return ["H", "V"]
    return choose_deep_carve_orientations(block, carve_rule=carve_rule)



def choose_deep_carve_orientations(
    block: StorageBlock,
    carve_rule: dict | None = None,
) -> list[str]:
    """Choose candidate deep-carve orientations for a storage block."""
    if carve_rule:
        access_count = int(getattr(block, "access_sides", 0))
        key = "two_sided" if access_count >= 2 else "one_sided"
        value = carve_rule.get(key, carve_rule.get("fallback"))
        if isinstance(value, str) and value in {"H", "V", "Both"}:
            return ["H", "V"] if value == "Both" else [value]
        if isinstance(value, (list, tuple)):
            orientations = [str(item) for item in value if item in {"H", "V"}]
            if orientations:
                return list(dict.fromkeys(orientations))

    if block.height > block.width:
        return ["H"]
    if block.width > block.height:
        return ["V"]
    return ["H", "V"]


def _mark_remaining_vector(
    vector: np.ndarray | None,
    start: int,
    aisle_width: int,
) -> np.ndarray | None:
    """Activate only the successful aisle start bit in a local vector copy."""
    del aisle_width
    if vector is None:
        return None
    updated = np.asarray(vector).copy()
    if 0 <= start < updated.shape[0]:
        updated[start] = 1
    return updated



def _append_block_child(
    children: list[BeamNode],
    seen_signatures: set[bytes],
    node: BeamNode,
    carved: np.ndarray,
    action: str,
    *,
    orientation: str,
    start: int,
    aisle_width: int,
) -> bool:
    signature = layout_signature(carved)
    if signature in seen_signatures:
        return False
    seen_signatures.add(signature)

    remaining_h = node.remaining_h
    remaining_v = node.remaining_v
    if orientation == "H":
        remaining_h = _mark_remaining_vector(remaining_h, start, aisle_width)
    elif orientation == "V":
        remaining_v = _mark_remaining_vector(remaining_v, start, aisle_width)

    children.append(
        node.copy_with(
            layout=carved,
            depth=node.depth + 1,
            action=action,
            remaining_h=remaining_h,
            remaining_v=remaining_v,
            trace_append=action,
        )
    )
    return True


def _try_append_repair_child_for_block(
    children: list[BeamNode],
    seen_signatures: set[bytes],
    node: BeamNode,
    block: StorageBlock,
    *,
    orientation: str,
    aisle_width: int,
    min_fragment_size: int,
) -> bool:
    """Try one midpoint-first repair carve for one full-span block."""
    if orientation == "H":
        starts = candidate_horizontal_repair_start_rows(block, aisle_width)
        for start_row in starts:
            carved = try_carve_block_horizontal(
                node.layout,
                block,
                start_row,
                aisle_width,
                min_fragment_size=min_fragment_size,
            )
            if carved is None:
                continue
            return _append_block_child(
                children,
                seen_signatures,
                node,
                carved,
                f"BH:{start_row}",
                orientation="H",
                start=start_row,
                aisle_width=aisle_width,
            )

    elif orientation == "V":
        starts = candidate_vertical_repair_start_cols(block, aisle_width)
        for start_col in starts:
            carved = try_carve_block_vertical(
                node.layout,
                block,
                start_col,
                aisle_width,
                min_fragment_size=min_fragment_size,
            )
            if carved is None:
                continue
            return _append_block_child(
                children,
                seen_signatures,
                node,
                carved,
                f"BV:{start_col}",
                orientation="V",
                start=start_col,
                aisle_width=aisle_width,
            )
    return False


def generate_block_children(
    node: BeamNode,
    aisle_width: int,
    min_fragment_size: int = 2,
    carve_rule: dict | None = None,
    max_children_per_block: int | None = None,
    secondary_spacing_offset: int = 0,
) -> list[BeamNode]:
    """Generate block-level deep-carve children from one node."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")
    if min_fragment_size <= 0:
        raise ValueError("min_fragment_size must be positive.")
    if max_children_per_block is not None and max_children_per_block <= 0:
        raise ValueError("max_children_per_block must be positive when provided.")
    if secondary_spacing_offset < 0:
        raise ValueError("secondary_spacing_offset must be non-negative.")

    blocks = detect_storage_blocks(node.layout)
    assign_pick_face_access_sides(node.layout, blocks)

    repair_children: list[BeamNode] = []
    repair_seen_signatures: set[bytes] = set()
    repair_block_seen = False

    for block in blocks:
        if not requires_full_span_repair(block, node.layout):
            continue

        repair_block_seen = True
        for orientation in choose_repair_carve_orientations(block):
            added = _try_append_repair_child_for_block(
                repair_children,
                repair_seen_signatures,
                node,
                block,
                orientation=orientation,
                aisle_width=aisle_width,
                min_fragment_size=min_fragment_size,
            )
            if added:
                break

    if repair_block_seen:
        return repair_children

    children: list[BeamNode] = []
    seen_signatures: set[bytes] = set()

    for block in blocks:
        if is_block_too_small_for_carving(block, aisle_width):
            continue
        if is_shallow_block(block, aisle_width):
            continue

        added_for_block = 0
        orientations = choose_block_carve_orientations(
            block,
            repair_required=False,
            carve_rule=carve_rule,
        )

        for orientation in orientations:
            if orientation == "H":
                starts = candidate_horizontal_start_rows(
                    block,
                    aisle_width,
                    spacing_offset=secondary_spacing_offset,
                )
                for start_row in starts:
                    carved = try_carve_block_horizontal(
                        node.layout,
                        block,
                        start_row,
                        aisle_width,
                        min_fragment_size=min_fragment_size,
                    )
                    if carved is None:
                        continue
                    if _append_block_child(
                        children,
                        seen_signatures,
                        node,
                        carved,
                        f"BH:{start_row}",
                        orientation="H",
                        start=start_row,
                        aisle_width=aisle_width,
                    ):
                        added_for_block += 1
                    if (
                        max_children_per_block is not None
                        and added_for_block >= max_children_per_block
                    ):
                        break

            elif orientation == "V":
                starts = candidate_vertical_start_cols(
                    block,
                    aisle_width,
                    spacing_offset=secondary_spacing_offset,
                )
                for start_col in starts:
                    carved = try_carve_block_vertical(
                        node.layout,
                        block,
                        start_col,
                        aisle_width,
                        min_fragment_size=min_fragment_size,
                    )
                    if carved is None:
                        continue
                    if _append_block_child(
                        children,
                        seen_signatures,
                        node,
                        carved,
                        f"BV:{start_col}",
                        orientation="V",
                        start=start_col,
                        aisle_width=aisle_width,
                    ):
                        added_for_block += 1
                    if (
                        max_children_per_block is not None
                        and added_for_block >= max_children_per_block
                    ):
                        break

            if (
                max_children_per_block is not None
                and added_for_block >= max_children_per_block
            ):
                break

    return children



def _remaining_h(root_node: BeamNode, chromosome) -> np.ndarray:
    source = root_node.remaining_h if root_node.remaining_h is not None else chromosome.h
    return np.asarray(source).copy()


def _remaining_v(root_node: BeamNode, chromosome) -> np.ndarray:
    source = root_node.remaining_v if root_node.remaining_v is not None else chromosome.v
    return np.asarray(source).copy()


def generate_global_children(
    root_node: BeamNode,
    chromosome,
    aisle_width: int,
) -> list[BeamNode]:
    """Generate first-level global H/V children from active chromosome bits."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")

    children: list[BeamNode] = []
    seen_signatures: set[bytes] = set()
    remaining_h = _remaining_h(root_node, chromosome)
    remaining_v = _remaining_v(root_node, chromosome)

    for row in chromosome.active_h_indices():
        carved = carve_global_horizontal(root_node.layout, row, aisle_width)
        if carved is None:
            continue
        signature = layout_signature(carved)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        action = f"H:{row}"
        children.append(
            root_node.copy_with(
                layout=carved,
                depth=root_node.depth + 1,
                action=action,
                remaining_h=remaining_h,
                remaining_v=remaining_v,
                trace_append=action,
            )
        )

    for col in chromosome.active_v_indices():
        carved = carve_global_vertical(root_node.layout, col, aisle_width)
        if carved is None:
            continue
        signature = layout_signature(carved)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        action = f"V:{col}"
        children.append(
            root_node.copy_with(
                layout=carved,
                depth=root_node.depth + 1,
                action=action,
                remaining_h=remaining_h,
                remaining_v=remaining_v,
                trace_append=action,
            )
        )

    return children


def generate_direct_global_children(
    root_node: BeamNode,
    aisle_width: int,
) -> list[BeamNode]:
    """Generate first-level global H/V children from all feasible starts."""
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")

    children: list[BeamNode] = []
    seen_signatures: set[bytes] = set()

    for row in find_feasible_global_horizontal_starts(root_node.layout, aisle_width):
        carved = carve_global_horizontal(root_node.layout, row, aisle_width)
        if carved is None:
            continue
        signature = layout_signature(carved)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        action = f"H:{row}"
        children.append(
            root_node.copy_with(
                layout=carved,
                depth=root_node.depth + 1,
                action=action,
                trace_append=action,
            )
        )

    for col in find_feasible_global_vertical_starts(root_node.layout, aisle_width):
        carved = carve_global_vertical(root_node.layout, col, aisle_width)
        if carved is None:
            continue
        signature = layout_signature(carved)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        action = f"V:{col}"
        children.append(
            root_node.copy_with(
                layout=carved,
                depth=root_node.depth + 1,
                action=action,
                trace_append=action,
            )
        )

    return children


__all__ = [
    "apply_aisle_code",
    "block_fragment_valid_after_carve",
    "carve_global_horizontal",
    "carve_global_vertical",
    "candidate_horizontal_band_cells",
    "candidate_horizontal_start_rows",
    "candidate_vertical_band_cells",
    "candidate_vertical_start_cols",
    "choose_deep_carve_orientations",
    "find_feasible_global_horizontal_starts",
    "find_feasible_global_vertical_starts",
    "generate_block_children",
    "generate_direct_global_children",
    "generate_global_children",
    "global_horizontal_band_cells",
    "global_vertical_band_cells",
    "has_aisle_in_span",
    "is_carvable_cell",
    "is_protected_cell",
    "is_span_protected",
    "same_orientation_neighbor_span_has_aisle",
    "try_carve_block_horizontal",
    "try_carve_block_vertical",
    "block_spans_eligible_height",
    "block_spans_eligible_width",
    "candidate_horizontal_repair_start_rows",
    "candidate_vertical_repair_start_cols",
    "choose_block_carve_orientations",
    "choose_repair_carve_orientations",
    "is_block_too_small_for_carving",
    "is_shallow_block",
    "is_storage_eligible_cell",
    "requires_full_span_repair",
    "storage_eligible_mask_from_layout",
]
