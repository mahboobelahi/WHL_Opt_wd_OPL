"""Matplotlib-based editor for warehouse layout mask bundles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from whl_core.constants import (
    CELL_AISLE,
    CELL_EMPTY,
    CELL_PICK,
    CELL_STORAGE,
    EDITOR_CELL_CODES,
    SERVICE_CODES,
)
from whl_core.layout_io import empty_mask_bundle, normalize_mask_layers, save_mask
from whl_core.paths import MASK_DIR

TOOL_TO_LAYER = {
    "wall": "walls",
    "door": "doors",
    "reserved": "reserved",
    "restricted": "restricted",
    "pillar": "pillars",
    "storage": "storage",
    "pick_faces": "pick_faces",
    "aisle": "aisle",
}

TOOLS = tuple(TOOL_TO_LAYER)

TOOL_SHORTCUTS = {
    "1": "wall",
    "2": "door",
    "3": "reserved",
    "4": "restricted",
    "5": "pillar",
    "6": "storage",
    "7": "pick_faces",
    "8": "aisle",
}

DISPLAY_PRIORITY = (
    ("storage", CELL_STORAGE),
    ("pick_faces", CELL_PICK),
    ("aisle", CELL_AISLE),
    ("pillars", SERVICE_CODES["pillar"]),
    ("restricted", SERVICE_CODES["restricted"]),
    ("reserved", SERVICE_CODES["reserved"]),
    ("doors", SERVICE_CODES["door"]),
    ("walls", SERVICE_CODES["wall"]),
)

PALETTE = {
    CELL_EMPTY: "#ffffff",
    SERVICE_CODES["wall"]: "#5a5a5a",
    SERVICE_CODES["door"]: "#4f83cc",
    SERVICE_CODES["reserved"]: "#f0b35f",
    SERVICE_CODES["restricted"]: "#2f2f2f",
    SERVICE_CODES["pillar"]: "#c95f5f",
    CELL_STORAGE: "#8fcf9f",
    CELL_PICK: "#5fbfbc",
    CELL_AISLE: "#f4f1e8",
}

HELP_TEXT = (
    "1 wall | 2 door | 3 reserved | 4 restricted\n"
    "5 pillar | 6 storage | 7 pick_faces | 8 aisle\n"
    "e erase | b/B brush -/+ | +/- aisle width\n"
    "s save | t coords | h help"
)


def _default_save_path(name: str) -> Path:
    """Return the default mask path for a layout name."""
    filename = name if name.lower().endswith(".npz") else f"{name}.npz"
    return MASK_DIR / filename


class GridEditor:
    """Small interactive Matplotlib editor for mask-layer painting."""

    def __init__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        aisle_width: int = 1,
        name: str = "layout",
        masks: dict | None = None,
        save_path: Path | str | None = None,
    ) -> None:
        if masks is None:
            if rows is None or cols is None:
                raise ValueError("rows and cols are required for a new layout.")
            self.masks = empty_mask_bundle(rows, cols, aisle_width, name)
        else:
            self.masks = normalize_mask_layers(masks)
            if name == "layout":
                name = str(self.masks["name"])

        self.rows = int(self.masks["rows"])
        self.cols = int(self.masks["cols"])
        self.aisle_width = int(self.masks["aisle_width"])
        self.name = name
        self.masks["name"] = name

        self.save_path = (
            Path(save_path) if save_path is not None else _default_save_path(name)
        )
        self.saved_path: Path | None = None
        self.current_tool = "wall"
        self.brush_size = 1
        self.dragging = False
        self.show_coords = True
        self.show_help = True

        self._cmap, self._norm = self._make_cmap()
        self.fig, self.ax = plt.subplots(
            figsize=(max(6, self.cols * 0.45), max(5, self.rows * 0.45))
        )
        self._bind_events()
        self._draw()

    def _make_cmap(self) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        max_code = max(PALETTE)
        colors = [PALETTE.get(code, "#ffffff") for code in range(max_code + 1)]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(-0.5, max_code + 1.5, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def _bind_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def compose_display(self) -> np.ndarray:
        """Compose the editable display grid from mask layers."""
        grid = np.full((self.rows, self.cols), CELL_EMPTY, dtype=np.uint8)
        for key, code in DISPLAY_PRIORITY:
            grid[self.masks[key] > 0] = code
        return grid

    def _draw(self) -> None:
        grid = self.compose_display()
        self.ax.clear()
        self.ax.imshow(grid, cmap=self._cmap, norm=self._norm, interpolation="nearest")
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"{self.name} | tool={self.current_tool} | "
            f"brush={self.brush_size} | aisle_width={self.aisle_width}"
        )

        self.ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        self.ax.grid(which="minor", color="black", linewidth=0.25, alpha=0.35)
        self.ax.tick_params(which="minor", bottom=False, left=False)

        if self.show_coords:
            self.ax.set_xticks(np.arange(self.cols))
            self.ax.set_yticks(np.arange(self.rows))
            self.ax.tick_params(labelsize=7)
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        handles = [
            mpatches.Patch(color=PALETTE[code], label=label.replace("_", " ").title())
            for code, label in sorted(EDITOR_CELL_CODES.items())
            if code in PALETTE
        ]
        self.ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

        if self.show_help:
            self.fig.text(
                0.78,
                0.45,
                HELP_TEXT,
                fontsize=8,
                family="monospace",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.65",
                    "alpha": 0.95,
                },
            )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _event_cell(self, event) -> tuple[int, int] | None:
        if event.xdata is None or event.ydata is None:
            return None
        row = int(round(event.ydata))
        col = int(round(event.xdata))
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        return row, col

    def _clear_cell(self, row: int, col: int) -> None:
        for key in TOOL_TO_LAYER.values():
            self.masks[key][row, col] = 0

    def _paint_cell(self, row: int, col: int, tool: str) -> None:
        self._clear_cell(row, col)
        if tool != "erase":
            self.masks[TOOL_TO_LAYER[tool]][row, col] = 1

    def _paint_brush(self, row: int, col: int, tool: str) -> None:
        radius = self.brush_size // 2
        for r in range(max(0, row - radius), min(self.rows, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.cols, col + radius + 1)):
                self._paint_cell(r, c, tool)

    def on_press(self, event) -> None:
        cell = self._event_cell(event)
        if cell is None:
            return
        self.dragging = True
        tool = "erase" if event.button == 3 else self.current_tool
        self._paint_brush(*cell, tool)
        self._draw()

    def on_motion(self, event) -> None:
        if not self.dragging:
            return
        cell = self._event_cell(event)
        if cell is None:
            return
        tool = "erase" if event.button == 3 else self.current_tool
        self._paint_brush(*cell, tool)
        self._draw()

    def on_release(self, event) -> None:
        self.dragging = False

    def on_key(self, event) -> None:
        key = event.key
        if key in TOOL_SHORTCUTS:
            self.current_tool = TOOL_SHORTCUTS[key]
        elif key in {"e", "E"}:
            self.current_tool = "erase"
        elif key == "b":
            self.brush_size = max(1, self.brush_size - 1)
        elif key == "B":
            self.brush_size += 1
        elif key in {"+", "="}:
            self.aisle_width += 1
        elif key in {"-", "_"}:
            self.aisle_width = max(1, self.aisle_width - 1)
        elif key in {"t", "T"}:
            self.show_coords = not self.show_coords
        elif key in {"h", "H"}:
            self.show_help = not self.show_help
        elif key in {"s", "S"}:
            self.save()
        self._draw()

    def save(self, path: Path | str | None = None) -> Path:
        """Save the current mask bundle and return the saved path."""
        target = Path(path) if path is not None else self.save_path
        self.masks["rows"] = self.rows
        self.masks["cols"] = self.cols
        self.masks["aisle_width"] = self.aisle_width
        self.masks["name"] = self.name
        self.saved_path = save_mask(self.masks, target)
        print(f"[layout_editor] saved {self.saved_path}")
        return self.saved_path

    def show(self) -> None:
        """Show the editor window."""
        plt.show()


__all__ = ["GridEditor", "TOOLS", "TOOL_TO_LAYER"]
