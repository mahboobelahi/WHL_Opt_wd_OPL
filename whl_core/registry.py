"""Layout filename registry utilities for editor-managed layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whl_core.paths import CONFIG_DIR

DEFAULT_LAYOUT_REGISTRY_PATH = CONFIG_DIR / "layouts.json"

def _registry_path(path: Path | None = None) -> Path:
    """Return the explicit registry path or the default layout registry path."""
    return path if path is not None else DEFAULT_LAYOUT_REGISTRY_PATH

def _coerce_layouts(data: Any) -> dict[int, str]:
    """Validate and convert JSON registry data to ``dict[int, str]``."""
    if not isinstance(data, dict):
        raise ValueError("Layout registry must be a JSON object.")

    layouts: dict[int, str] = {}
    for raw_key, raw_value in data.items():
        try:
            layout_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            message = f"Layout registry key is not an integer: {raw_key!r}"
            raise ValueError(message) from exc

        if layout_id < 1:
            raise ValueError(f"Layout registry ID must be positive: {layout_id}")
        if not isinstance(raw_value, str):
            raise ValueError(f"Layout filename must be a string for ID {layout_id}.")

        layouts[layout_id] = raw_value

    return dict(sorted(layouts.items()))

def load_layouts(path: Path | None = None) -> dict[int, str]:
    """Load the layout filename registry from JSON."""
    registry_path = _registry_path(path)
    if not registry_path.exists():
        return {}

    with registry_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return _coerce_layouts(data)

def save_layouts(layouts: dict[int, str], path: Path | None = None) -> None:
    """Save the layout filename registry to JSON."""
    registry_path = _registry_path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    clean_layouts = _coerce_layouts(layouts)
    serializable = {
        str(layout_id): filename for layout_id, filename in clean_layouts.items()
    }

    with registry_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)
        file.write("\n")

def next_layout_id(layouts: dict[int, str]) -> int:
    """Return the next positive integer layout ID for a registry mapping."""
    if not layouts:
        return 1
    return max(layouts) + 1

def add_layout(filename: str, path: Path | None = None) -> int:
    """Add a filename to the registry and return its assigned layout ID."""
    if not filename:
        raise ValueError("Layout filename must not be empty.")

    layouts = load_layouts(path)
    layout_id = next_layout_id(layouts)
    layouts[layout_id] = filename
    save_layouts(layouts, path)
    return layout_id

def delete_layout(
    layout_id: int,
    path: Path | None = None,
    reindex: bool = True,
) -> str | None:
    """Delete a layout ID and return the removed filename, if it existed."""
    layouts = load_layouts(path)
    removed = layouts.pop(layout_id, None)
    if removed is None:
        return None

    if reindex:
        layouts = {
            new_id: filename
            for new_id, filename in enumerate(layouts.values(), start=1)
        }

    save_layouts(layouts, path)
    return removed

def list_layouts(path: Path | None = None) -> dict[int, str]:
    """Return the current layout filename registry."""
    return load_layouts(path)

__all__ = [
    "DEFAULT_LAYOUT_REGISTRY_PATH",
    "add_layout",
    "delete_layout",
    "list_layouts",
    "load_layouts",
    "next_layout_id",
    "save_layouts",
]
