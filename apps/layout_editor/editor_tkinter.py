"""Tkinter-native canvas editor for warehouse layout mask bundles."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from whl_core.constants import (
    CELL_AISLE,
    CELL_EMPTY,
    CELL_PICK,
    CELL_STORAGE,
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
    SERVICE_CODES["wall"]: "#b4b4b499",
    SERVICE_CODES["door"]: "#9664c880",
    SERVICE_CODES["reserved"]: "#f0b46480",
    SERVICE_CODES["restricted"]: "#50505080",
    SERVICE_CODES["pillar"]: "#dc646480",
    CELL_STORAGE: "#93c57299",
    CELL_PICK: "#64aa8c80",
    CELL_AISLE: "#f5f5f097",
}

CONTROL_MASK = 0x0004


def _tk_color(color: str) -> str:
    """Return a Tk-compatible color string.

    Tkinter accepts ``#RRGGBB`` but not ``#RRGGBBAA`` alpha colors. If an alpha
    channel is present, it is ignored for canvas drawing.
    """
    if color.startswith("#") and len(color) == 9:
        return color[:7]
    return color


def is_ctrl_pressed(state: int) -> bool:
    """Return whether a Tk event state includes the Control modifier."""
    return bool(state & CONTROL_MASK)


def interpolate_cells(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return grid cells along a straight drag path from ``start`` to ``end``."""
    start_row, start_col = start
    end_row, end_col = end
    steps = max(abs(end_row - start_row), abs(end_col - start_col))
    if steps == 0:
        return [start]

    cells: list[tuple[int, int]] = []
    for step in range(steps + 1):
        ratio = step / steps
        row = round(start_row + (end_row - start_row) * ratio)
        col = round(start_col + (end_col - start_col) * ratio)
        cell = (row, col)
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return cells


def _default_save_path(name: str) -> Path:
    """Return the default mask path for a layout name."""
    filename = name if name.lower().endswith(".npz") else f"{name}.npz"
    return MASK_DIR / filename


class TkinterGridEditor:
    """Tkinter canvas editor for painting layout mask layers."""

    def __init__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        aisle_width: int = 1,
        name: str = "layout",
        masks: dict | None = None,
        save_path: Path | str | None = None,
        master: tk.Misc | None = None,
        cell_size: int = 28,
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
        self._drag_tool: str | None = None
        self._last_drag_cell: tuple[int, int] | None = None
        self._owns_root = master is None
        self.window = tk.Tk() if master is None else tk.Toplevel(master)
        self.window.title(f"WHL Layout Editor - {self.name}")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.current_tool = tk.StringVar(master=self.window, value="wall")
        self.brush_size = tk.IntVar(master=self.window, value=1)
        self.aisle_width_var = tk.IntVar(master=self.window, value=self.aisle_width)
        self.show_coords = tk.BooleanVar(master=self.window, value=True)
        self.status_text = tk.StringVar(master=self.window, value="")
        self.cell_size = max(12, int(cell_size))

        self._build()
        self._bind_events()
        self._draw()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.window, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Tool").pack(side="left")
        self.tool_menu = ttk.Combobox(
            toolbar,
            textvariable=self.current_tool,
            values=TOOLS,
            state="readonly",
            width=14,
        )
        self.tool_menu.pack(side="left", padx=(4, 12))

        ttk.Label(toolbar, text="Brush").pack(side="left")
        ttk.Spinbox(
            toolbar,
            from_=1,
            to=25,
            textvariable=self.brush_size,
            width=4,
            command=self._draw,
        ).pack(side="left", padx=(4, 12))

        ttk.Label(toolbar, text="Aisle width").pack(side="left")
        ttk.Button(toolbar, text="-", width=3, command=self.decrease_aisle_width).pack(
            side="left", padx=(4, 0)
        )
        ttk.Label(toolbar, textvariable=self.aisle_width_var, width=4).pack(side="left")
        ttk.Button(toolbar, text="+", width=3, command=self.increase_aisle_width).pack(
            side="left", padx=(0, 12)
        )

        ttk.Checkbutton(
            toolbar,
            text="Coords",
            variable=self.show_coords,
            command=self._draw,
        ).pack(side="left", padx=(0, 12))
        ttk.Button(toolbar, text="Save", command=self.save).pack(side="right")

        body = ttk.Frame(self.window)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            body,
            width=min(1000, max(400, self.cols * self.cell_size)),
            height=min(700, max(300, self.rows * self.cell_size)),
            background="#f7f7f7",
        )
        x_scroll = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self._update_status()
        ttk.Label(self.window, textvariable=self.status_text, padding=(8, 4)).pack(
            fill="x"
        )

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self.on_paint)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-3>", self.on_erase)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion_status)
        self.window.bind("<Key>", self.on_key)
        self.canvas.configure(
            scrollregion=(0, 0, self.cols * self.cell_size, self.rows * self.cell_size)
        )
        self.canvas.focus_set()

    def compose_display(self) -> np.ndarray:
        """Compose the editable display grid from mask layers."""
        grid = np.full((self.rows, self.cols), CELL_EMPTY, dtype=np.uint8)
        for key, code in DISPLAY_PRIORITY:
            grid[self.masks[key] > 0] = code
        return grid

    def _draw(self) -> None:
        self.canvas.delete("all")
        grid = self.compose_display()
        size = self.cell_size

        for row in range(self.rows):
            for col in range(self.cols):
                x0 = col * size
                y0 = row * size
                x1 = x0 + size
                y1 = y0 + size
                code = int(grid[row, col])
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=_tk_color(PALETTE.get(code, "#ffffff")),
                    outline="#303030",
                    width=1,
                )
                if self.show_coords.get() and size >= 20:
                    self.canvas.create_text(
                        x0 + size / 2,
                        y0 + size / 2,
                        text=f"{row},{col}",
                        font=("Segoe UI", max(6, size // 5)),
                        fill="#222222",
                    )

    def _update_status(self, cell: tuple[int, int] | None = None) -> None:
        row_col = "row/col: -" if cell is None else f"row/col: {cell[0]},{cell[1]}"
        self.status_text.set(
            f"Tool: {self.current_tool.get()} | "
            f"Brush: {self.brush_size.get()} | "
            f"Aisle width: {self.aisle_width_var.get()} | "
            f"{row_col} | Ctrl-drag paints continuously"
        )

    def _event_cell(self, event) -> tuple[int, int] | None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
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
        brush = max(1, int(self.brush_size.get()))
        radius = brush // 2
        for r in range(max(0, row - radius), min(self.rows, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.cols, col + radius + 1)):
                self._paint_cell(r, c, tool)

    def _paint_drag_path(
        self,
        cell: tuple[int, int],
        tool: str,
        interpolate: bool,
    ) -> None:
        if interpolate and self._last_drag_cell is not None:
            cells = interpolate_cells(self._last_drag_cell, cell)
        else:
            cells = [cell]

        for row, col in cells:
            self._paint_brush(row, col, tool)

        self._last_drag_cell = cell
        self._update_status(cell)
        self._draw()

    def on_paint(self, event) -> None:
        """Paint with the selected tool."""
        cell = self._event_cell(event)
        if cell is None:
            return
        self._drag_tool = self.current_tool.get()
        self._paint_drag_path(cell, self._drag_tool, is_ctrl_pressed(event.state))

    def on_erase(self, event) -> None:
        """Erase with right-click drag."""
        cell = self._event_cell(event)
        if cell is None:
            return
        self._drag_tool = "erase"
        self._paint_drag_path(cell, "erase", is_ctrl_pressed(event.state))

    def on_release(self, event) -> None:
        """Reset drag state after mouse release."""
        self._drag_tool = None
        self._last_drag_cell = None
        self._update_status(self._event_cell(event))

    def on_motion_status(self, event) -> None:
        """Update row/col status without painting."""
        self._update_status(self._event_cell(event))

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts."""
        key = event.keysym
        char = event.char
        if char in TOOL_SHORTCUTS:
            self.current_tool.set(TOOL_SHORTCUTS[char])
        elif char in {"e", "E"}:
            self.current_tool.set("erase")
        elif char == "b":
            self.brush_size.set(max(1, int(self.brush_size.get()) - 1))
        elif char == "B":
            self.brush_size.set(int(self.brush_size.get()) + 1)
        elif char in {"+", "="} or key == "plus":
            self.increase_aisle_width()
        elif char in {"-", "_"} or key == "minus":
            self.decrease_aisle_width()
        elif char in {"s", "S"}:
            self.save()
        self._update_status()
        self._draw()

    def increase_aisle_width(self) -> None:
        """Increase saved aisle-width metadata."""
        self.aisle_width_var.set(int(self.aisle_width_var.get()) + 1)
        self._update_status()

    def decrease_aisle_width(self) -> None:
        """Decrease saved aisle-width metadata."""
        self.aisle_width_var.set(max(1, int(self.aisle_width_var.get()) - 1))
        self._update_status()

    def save(self, path: Path | str | None = None) -> Path:
        """Save the current mask bundle and return the saved path."""
        target = Path(path) if path is not None else self.save_path
        self.aisle_width = int(self.aisle_width_var.get())
        self.masks["rows"] = self.rows
        self.masks["cols"] = self.cols
        self.masks["aisle_width"] = self.aisle_width
        self.masks["name"] = self.name
        try:
            self.saved_path = save_mask(self.masks, target)
        except ValueError as exc:
            messagebox.showwarning("Cannot save layout", str(exc))
            raise
        messagebox.showinfo("Saved", f"Saved {self.saved_path}")
        return self.saved_path

    def close(self) -> None:
        """Close without automatic save."""
        self.window.destroy()

    def show(self) -> None:
        """Show the editor window and wait until it is closed."""
        if self._owns_root:
            self.window.mainloop()
        else:
            self.window.wait_window()


GridEditor = TkinterGridEditor

__all__ = [
    "GridEditor",
    "TOOLS",
    "TOOL_TO_LAYER",
    "TkinterGridEditor",
    "interpolate_cells",
    "is_ctrl_pressed",
]
