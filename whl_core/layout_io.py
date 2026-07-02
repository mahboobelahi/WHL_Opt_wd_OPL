"""Input and output helpers for warehouse layout mask bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from whl_core.constants import (
    AISLE_CODES,
    CELL_AISLE,
    CELL_AISLE_CROSS,
    CELL_AISLE_H,
    CELL_AISLE_V,
    CELL_EMPTY,
    CELL_PICK,
    CELL_STORAGE,
    SERVICE_CODES,
)

MASK_LAYER_KEYS = (
    "walls",
    "doors",
    "reserved",
    "restricted",
    "pillars",
    "storage",
    "pick_faces",
    "aisle",
)

GENERIC_AISLE_LAYER_KEYS = (
    "aisle",
    "aisles",
    "forced_aisle",
    "forced_aisles",
)

ORIENTED_AISLE_LAYER_KEYS = (
    "aisle_h",
    "aisle_v",
    "aisle_cross",
)

ALL_MASK_LAYER_KEYS = MASK_LAYER_KEYS + ORIENTED_AISLE_LAYER_KEYS

MASK_LAYER_PRIORITY = (
    ("pick_faces", CELL_PICK),
    ("aisle", CELL_AISLE),
    ("aisle_h", CELL_AISLE_H),
    ("aisle_v", CELL_AISLE_V),
    ("aisle_cross", CELL_AISLE_CROSS),
    ("pillars", SERVICE_CODES["pillar"]),
    ("restricted", SERVICE_CODES["restricted"]),
    ("reserved", SERVICE_CODES["reserved"]),
    ("doors", SERVICE_CODES["door"]),
    ("walls", SERVICE_CODES["wall"]),
)

def _validate_positive_int(value: int, name: str) -> int:
    """Return ``value`` as an int after validating that it is positive."""
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int_value

def _infer_shape(masks: dict[str, Any]) -> tuple[int, int]:
    """Infer mask shape from metadata or first available 2D layer."""
    if "rows" in masks and "cols" in masks:
        return _validate_positive_int(masks["rows"], "rows"), _validate_positive_int(
            masks["cols"], "cols"
        )

    for key in (*ALL_MASK_LAYER_KEYS, *GENERIC_AISLE_LAYER_KEYS):
        value = masks.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            rows, cols = value.shape
            return int(rows), int(cols)

    raise ValueError("Mask bundle must include rows/cols or at least one 2D layer.")

def _as_binary_layer(value: Any, shape: tuple[int, int], key: str) -> np.ndarray:
    """Convert a mask layer to a binary ``uint8`` array with ``shape``."""
    if value is None:
        return np.zeros(shape, dtype=np.uint8)

    array = np.asarray(value)
    if array.shape != shape:
        message = f"Mask layer {key!r} has shape {array.shape}, expected {shape}."
        raise ValueError(message)

    return (array > 0).astype(np.uint8)

def _combined_generic_aisle_layer(masks: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=np.uint8)
    for key in GENERIC_AISLE_LAYER_KEYS:
        combined |= _as_binary_layer(masks.get(key), shape, key)
    return combined

def _scalar_text(value: Any, default: str) -> str:
    """Convert npz scalar/object values to plain text."""
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)

def empty_mask_bundle(rows: int, cols: int, aisle_width: int, name: str) -> dict:
    """Return a new empty mask bundle with all legacy and oriented layers."""
    rows = _validate_positive_int(rows, "rows")
    cols = _validate_positive_int(cols, "cols")
    aisle_width = _validate_positive_int(aisle_width, "aisle_width")
    if not name:
        raise ValueError("name must not be empty.")

    shape = (rows, cols)
    masks = {key: np.zeros(shape, dtype=np.uint8) for key in ALL_MASK_LAYER_KEYS}
    masks.update(
        {
            "rows": rows,
            "cols": cols,
            "aisle_width": aisle_width,
            "name": name,
        }
    )
    return masks

def normalize_mask_layers(masks: dict) -> dict:
    """Return a complete, shape-validated mask bundle."""
    rows, cols = _infer_shape(masks)
    shape = (rows, cols)

    normalized = {
        "rows": rows,
        "cols": cols,
        "aisle_width": _validate_positive_int(
            masks.get("aisle_width", 1),
            "aisle_width",
        ),
        "name": _scalar_text(masks.get("name"), "layout"),
    }

    for key in MASK_LAYER_KEYS:
        if key == "aisle":
            normalized[key] = _combined_generic_aisle_layer(masks, shape)
        else:
            normalized[key] = _as_binary_layer(masks.get(key), shape, key)
    for key in ORIENTED_AISLE_LAYER_KEYS:
        normalized[key] = _as_binary_layer(masks.get(key), shape, key)

    return normalized

def fixed_aisle_mask_from_masks(masks: dict[str, Any]) -> np.ndarray:
    """Return cells that were fixed/user-painted aisles in the mask bundle."""
    normalized = normalize_mask_layers(masks)
    fixed = normalized["aisle"] > 0
    for key in ORIENTED_AISLE_LAYER_KEYS:
        fixed |= normalized[key] > 0
    return fixed

def _run_lengths_by_row(mask: np.ndarray) -> np.ndarray:
    rows, cols = mask.shape
    lengths = np.zeros(mask.shape, dtype=np.int32)
    for row in range(rows):
        col = 0
        while col < cols:
            if not mask[row, col]:
                col += 1
                continue
            start = col
            while col + 1 < cols and mask[row, col + 1]:
                col += 1
            end = col
            lengths[row, start : end + 1] = end - start + 1
            col += 1
    return lengths

def _run_lengths_by_col(mask: np.ndarray) -> np.ndarray:
    rows, cols = mask.shape
    lengths = np.zeros(mask.shape, dtype=np.int32)
    for col in range(cols):
        row = 0
        while row < rows:
            if not mask[row, col]:
                row += 1
                continue
            start = row
            while row + 1 < rows and mask[row + 1, col]:
                row += 1
            end = row
            lengths[start : end + 1, col] = end - start + 1
            row += 1
    return lengths

def _assign_tie_cells_by_component(
    aisle_mask: np.ndarray,
    tie_mask: np.ndarray,
    aisle_h_mask: np.ndarray,
    aisle_v_mask: np.ndarray,
) -> None:
    rows, cols = aisle_mask.shape
    visited = np.zeros(aisle_mask.shape, dtype=bool)

    for start_row, start_col in zip(*np.where(aisle_mask), strict=True):
        start_row = int(start_row)
        start_col = int(start_col)
        if visited[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if visited[next_row, next_col] or not aisle_mask[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))

        tie_cells = [(row, col) for row, col in cells if tie_mask[row, col]]
        if not tie_cells:
            continue

        component_rows = [row for row, _ in cells]
        component_cols = [col for _, col in cells]
        height = max(component_rows) - min(component_rows) + 1
        width = max(component_cols) - min(component_cols) + 1
        target = aisle_h_mask if width >= height else aisle_v_mask
        for row, col in tie_cells:
            target[row, col] = True

def infer_aisle_orientation_masks(
    aisle_mask: np.ndarray,
    aisle_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer H/V/CROSS masks from a generic aisle mask."""
    aisle_width = _validate_positive_int(aisle_width, "aisle_width")
    mask = np.asarray(aisle_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("aisle_mask must be a 2D array.")

    h_run = _run_lengths_by_row(mask)
    v_run = _run_lengths_by_col(mask)

    aisle_cross_mask = mask & (h_run > aisle_width) & (v_run > aisle_width)
    unresolved = mask & ~aisle_cross_mask
    aisle_h_mask = unresolved & (h_run > v_run)
    aisle_v_mask = unresolved & (v_run > h_run)
    tie_mask = unresolved & ~(aisle_h_mask | aisle_v_mask)

    _assign_tie_cells_by_component(mask, tie_mask, aisle_h_mask, aisle_v_mask)

    return aisle_h_mask.copy(), aisle_v_mask.copy(), aisle_cross_mask.copy()

def overlapping_mask_cells(masks: dict) -> list[tuple[int, int]]:
    """Return cells with invalid overlap among editable mask layers."""
    normalized = normalize_mask_layers(masks)

    protected_keys = (
        "walls",
        "doors",
        "reserved",
        "restricted",
        "pillars",
        "storage",
        "pick_faces",
        "aisle",
        "aisle_cross",
    )
    protected_occupancy = np.zeros(
        (normalized["rows"], normalized["cols"]),
        dtype=np.uint8,
    )
    for key in protected_keys:
        protected_occupancy += (normalized[key] > 0).astype(np.uint8)

    oriented_occupancy = (
        (normalized["aisle_h"] > 0) | (normalized["aisle_v"] > 0)
    ).astype(np.uint8)

    invalid = (protected_occupancy > 1) | (
        (protected_occupancy > 0) & (oriented_occupancy > 0)
    )

    rows, cols = np.where(invalid)
    return list(zip(rows.astype(int).tolist(), cols.astype(int).tolist(), strict=True))

def validate_mask_layer_exclusivity(masks: dict) -> None:
    """Raise ``ValueError`` if any cell is active in multiple mask layers."""
    overlaps = overlapping_mask_cells(masks)
    if overlaps:
        preview = ", ".join(f"({row},{col})" for row, col in overlaps[:10])
        suffix = "" if len(overlaps) <= 10 else f", ... +{len(overlaps) - 10} more"
        raise ValueError(
            "Mask layers must be mutually exclusive. "
            f"Overlapping cells: {preview}{suffix}"
        )

def load_mask(path: Path | str) -> dict:
    """Load a ``.npz`` mask bundle and fill missing optional layers safely."""
    mask_path = Path(path)
    with np.load(mask_path, allow_pickle=True) as archive:
        loaded = {key: archive[key] for key in archive.files}

    if "name" not in loaded:
        loaded["name"] = mask_path.stem

    return normalize_mask_layers(loaded)

def _fill_empty_cells_as_storage(masks: dict) -> dict:
    """Set cells with no active layer to storage before saving."""
    normalized = normalize_mask_layers(masks)
    validate_mask_layer_exclusivity(normalized)
    occupied = np.zeros((normalized["rows"], normalized["cols"]), dtype=bool)

    for key in ALL_MASK_LAYER_KEYS:
        if key != "storage":
            occupied |= normalized[key] > 0

    normalized["storage"][~occupied] = 1
    return normalized

def save_mask(masks: dict, path: Path | str) -> Path:
    """Save a mask bundle to a compressed ``.npz`` archive and return its path."""
    mask_path = Path(path)
    if mask_path.suffix.lower() != ".npz":
        mask_path = mask_path.with_suffix(".npz")

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _fill_empty_cells_as_storage(masks)
    np.savez_compressed(mask_path, **normalized)
    return mask_path

def mask_to_grid(masks: dict) -> np.ndarray:
    """Convert a mask bundle into a single display/analysis grid."""
    normalized = normalize_mask_layers(masks)
    grid = np.full(
        (normalized["rows"], normalized["cols"]),
        CELL_STORAGE,
        dtype=np.uint8,
    )

    grid[normalized["pick_faces"] > 0] = CELL_PICK

    explicit_h = normalized["aisle_h"] > 0
    explicit_v = normalized["aisle_v"] > 0
    explicit_cross = normalized["aisle_cross"] > 0
    explicit_any = explicit_h | explicit_v | explicit_cross

    inferred_h, inferred_v, inferred_cross = infer_aisle_orientation_masks(
        normalized["aisle"] > 0,
        int(normalized["aisle_width"]),
    )
    inferred_h &= ~explicit_any
    inferred_v &= ~explicit_any
    inferred_cross &= ~explicit_any

    aisle_h = explicit_h | inferred_h
    aisle_v = explicit_v | inferred_v
    aisle_cross = explicit_cross | inferred_cross | (aisle_h & aisle_v)

    grid[aisle_h] = CELL_AISLE_H
    grid[aisle_v] = CELL_AISLE_V
    grid[aisle_cross] = CELL_AISLE_CROSS

    for key, code in (
        ("pillars", SERVICE_CODES["pillar"]),
        ("restricted", SERVICE_CODES["restricted"]),
        ("reserved", SERVICE_CODES["reserved"]),
        ("doors", SERVICE_CODES["door"]),
        ("walls", SERVICE_CODES["wall"]),
    ):
        grid[normalized[key] > 0] = code

    return grid

def grid_to_display_codes(grid: np.ndarray) -> np.ndarray:
    """Return a uint8 grid suitable for display by visualization utilities."""
    array = np.asarray(grid, dtype=np.uint8)
    valid_codes = {
        CELL_EMPTY,
        CELL_STORAGE,
        CELL_PICK,
        *AISLE_CODES,
        *SERVICE_CODES.values(),
    }
    unknown = set(np.unique(array).tolist()) - valid_codes
    if unknown:
        raise ValueError(f"Grid contains unknown cell codes: {sorted(unknown)}")
    return array

__all__ = [
    "ALL_MASK_LAYER_KEYS",
    "GENERIC_AISLE_LAYER_KEYS",
    "MASK_LAYER_KEYS",
    "MASK_LAYER_PRIORITY",
    "ORIENTED_AISLE_LAYER_KEYS",
    "empty_mask_bundle",
    "fixed_aisle_mask_from_masks",
    "grid_to_display_codes",
    "infer_aisle_orientation_masks",
    "load_mask",
    "mask_to_grid",
    "normalize_mask_layers",
    "overlapping_mask_cells",
    "save_mask",
    "validate_mask_layer_exclusivity",
]
