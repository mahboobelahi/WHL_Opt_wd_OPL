"""Storage block detection and access-side classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from whl_core.constants import AISLE_CODES, CELL_STORAGE, SERVICE_CODES

Cell = tuple[int, int]
AccessSide = str

ACCESS_CODES: Final[frozenset[int]] = frozenset({*AISLE_CODES, SERVICE_CODES["door"]})
ACCESS_SIDE_NAMES: Final[frozenset[str]] = frozenset(
    {"top", "bottom", "left", "right"}
)

@dataclass(slots=True)
class StorageBlock:
    """A 4-neighbour connected component of storage cells."""

    id: int
    cells: list[Cell]
    rmin: int
    rmax: int
    cmin: int
    cmax: int
    height: int
    width: int
    access_side_names: frozenset[AccessSide] = field(default_factory=frozenset)
    raw_adjacent_access_side_names: frozenset[AccessSide] = field(default_factory=frozenset)
    pick_face_side_names: frozenset[AccessSide] = field(default_factory=frozenset)
    pick_faces: list[Cell] = field(default_factory=list)
    orientation: str = "S"

    @property
    def access_sides(self) -> int:
        """Number of effective access sides used by downstream logic."""
        return len(self.access_side_names)

    @property
    def raw_adjacent_access_sides(self) -> int:
        """Number of raw adjacent aisle/door/access-anchor sides."""
        return len(self.raw_adjacent_access_side_names)

    @property
    def cell_count(self) -> int:
        """Number of storage cells in the block."""
        return len(self.cells)

    @property
    def area(self) -> int:
        """Alias for the number of storage cells in the block."""
        return self.cell_count

    @property
    def has_top_access(self) -> bool:
        return "top" in self.access_side_names

    @property
    def has_bottom_access(self) -> bool:
        return "bottom" in self.access_side_names

    @property
    def has_left_access(self) -> bool:
        return "left" in self.access_side_names

    @property
    def has_right_access(self) -> bool:
        return "right" in self.access_side_names

    @property
    def has_top_bottom_access(self) -> bool:
        return set(self.access_side_names) == {"top", "bottom"}

    @property
    def has_left_right_access(self) -> bool:
        return set(self.access_side_names) == {"left", "right"}

    @property
    def has_access(self) -> bool:
        return self.access_sides > 0

    @property
    def is_one_sided(self) -> bool:
        return self.access_sides == 1

    @property
    def is_two_sided(self) -> bool:
        return self.has_top_bottom_access or self.has_left_right_access

    @property
    def two_sided_axis(self) -> str | None:
        """Return the opposite-side access axis, if present."""
        sides = set(self.access_side_names)
        if sides == {"top", "bottom"}:
            return "TB"
        if sides == {"left", "right"}:
            return "LR"
        return None

    @property
    def access_sides_T_B(self) -> bool:
        """Backward-compatible alias for top+bottom access."""
        return self.has_top_bottom_access

    @property
    def access_sides_L_R(self) -> bool:
        """Backward-compatible alias for left+right access."""
        return self.has_left_right_access

def neighbors4(row: int, col: int, rows: int, cols: int) -> tuple[Cell, ...]:
    """Return valid 4-neighbour coordinates for a grid cell."""
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return tuple((r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols)

_neighbors = neighbors4

def classify_block_orientation(height: int, width: int) -> str:
    """Classify a block by its bounding-box aspect."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    if width > height:
        return "H"
    if height > width:
        return "V"
    return "S"

_orientation = classify_block_orientation

def is_access_code(value: int) -> bool:
    """Return True if a grid value can provide storage-block access."""
    return int(value) in ACCESS_CODES

_is_access = is_access_code

def detect_block_access_sides(
    grid: np.ndarray,
    cells: list[Cell] | tuple[Cell, ...],
) -> frozenset[str]:
    """Detect actual access sides for a storage block."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    rows, cols = layout.shape
    sides: set[str] = set()

    for row, col in cells:
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"cell {(row, col)} is outside the grid.")

        if row > 0 and is_access_code(int(layout[row - 1, col])):
            sides.add("top")
        if row < rows - 1 and is_access_code(int(layout[row + 1, col])):
            sides.add("bottom")
        if col > 0 and is_access_code(int(layout[row, col - 1])):
            sides.add("left")
        if col < cols - 1 and is_access_code(int(layout[row, col + 1])):
            sides.add("right")

    return frozenset(sides)

def _access_sides(grid: np.ndarray, cells: list[Cell], orientation: str | None = None) -> int:
    """Return count of actual access sides."""
    _ = orientation
    return len(detect_block_access_sides(grid, cells))

def _make_block(block_id: int, cells: list[Cell], grid: np.ndarray) -> StorageBlock:
    block_rows = [row for row, _ in cells]
    block_cols = [col for _, col in cells]
    rmin = min(block_rows)
    rmax = max(block_rows)
    cmin = min(block_cols)
    cmax = max(block_cols)
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    raw_sides = detect_block_access_sides(grid, cells)
    return StorageBlock(
        id=block_id,
        cells=sorted(cells),
        rmin=rmin,
        rmax=rmax,
        cmin=cmin,
        cmax=cmax,
        height=height,
        width=width,
        access_side_names=raw_sides,
        raw_adjacent_access_side_names=raw_sides,
        orientation=classify_block_orientation(height, width),
    )

def detect_storage_blocks(grid: np.ndarray) -> list[StorageBlock]:
    """Detect maximal 4-neighbour connected CELL_STORAGE components."""
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")

    rows, cols = layout.shape
    visited = np.zeros((rows, cols), dtype=bool)
    blocks: list[StorageBlock] = []

    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or int(layout[start_row, start_col]) != CELL_STORAGE:
                continue

            queue: deque[Cell] = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            cells: list[Cell] = []

            while queue:
                row, col = queue.popleft()
                cells.append((row, col))

                for next_row, next_col in neighbors4(row, col, rows, cols):
                    if visited[next_row, next_col]:
                        continue
                    if int(layout[next_row, next_col]) != CELL_STORAGE:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            blocks.append(_make_block(len(blocks) + 1, cells, layout))

    return blocks

__all__ = [
    "ACCESS_CODES",
    "ACCESS_SIDE_NAMES",
    "AccessSide",
    "Cell",
    "StorageBlock",
    "classify_block_orientation",
    "detect_block_access_sides",
    "detect_storage_blocks",
    "is_access_code",
    "neighbors4",
]
