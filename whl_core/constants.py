"""Shared constants for warehouse layout models and experiments."""

from __future__ import annotations

CELL_EMPTY = 0

SERVICE_CODES = {
    "wall": 1,
    "door": 2,
    "reserved": 3,
    "restricted": 4,
    "pillar": 5,
}

CELL_STORAGE = 7
CELL_PICK = 8

CELL_AISLE_H = 9
CELL_AISLE_V = 10
CELL_AISLE_CROSS = 11
CELL_AISLE = CELL_AISLE_H

AISLE_CODES = {CELL_AISLE_H, CELL_AISLE_V, CELL_AISLE_CROSS}
STORAGE_LIKE_CODES = {CELL_STORAGE, CELL_PICK}
STRUCTURAL_CODES = set(SERVICE_CODES.values())
ALL_CELL_CODES = {
    CELL_EMPTY,
    *STRUCTURAL_CODES,
    CELL_STORAGE,
    CELL_PICK,
    *AISLE_CODES,
}

def is_aisle_code(value: int) -> bool:
    """Return whether a cell code represents any aisle orientation."""
    return int(value) in AISLE_CODES

def is_storage_like_code(value: int) -> bool:
    """Return whether a cell code represents storage or an explicit pick face."""
    return int(value) in STORAGE_LIKE_CODES

def is_door_code(value: int) -> bool:
    """Return whether a cell code represents a door/access anchor."""
    return int(value) == SERVICE_CODES["door"]

EDITOR_CELL_CODES = {
    CELL_EMPTY: "empty",
    SERVICE_CODES["wall"]: "wall",
    SERVICE_CODES["door"]: "door",
    SERVICE_CODES["reserved"]: "reserved",
    SERVICE_CODES["restricted"]: "restricted",
    SERVICE_CODES["pillar"]: "pillar",
    CELL_STORAGE: "storage",
    CELL_PICK: "pick",
    CELL_AISLE_H: "aisle_h",
    CELL_AISLE_V: "aisle_v",
    CELL_AISLE_CROSS: "aisle_cross",
}

__all__ = [
    "AISLE_CODES",
    "ALL_CELL_CODES",
    "CELL_AISLE",
    "CELL_AISLE_CROSS",
    "CELL_AISLE_H",
    "CELL_AISLE_V",
    "CELL_EMPTY",
    "CELL_PICK",
    "CELL_STORAGE",
    "EDITOR_CELL_CODES",
    "SERVICE_CODES",
    "STORAGE_LIKE_CODES",
    "STRUCTURAL_CODES",
    "is_aisle_code",
    "is_door_code",
    "is_storage_like_code",
]
