"""Simple plotting utilities for warehouse layout grids."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from whl_core.constants import (
    CELL_AISLE_CROSS,
    CELL_AISLE_H,
    CELL_AISLE_V,
    CELL_EMPTY,
    CELL_PICK,
    CELL_STORAGE,
    EDITOR_CELL_CODES,
    SERVICE_CODES,
)
from whl_core.layout_io import grid_to_display_codes

PALETTE = {
    CELL_EMPTY: "#ffffff",
    SERVICE_CODES["wall"]: "#b4b4b499",
    SERVICE_CODES["door"]: "#9664c880",
    SERVICE_CODES["reserved"]: "#f0b46480",
    SERVICE_CODES["restricted"]: "#50505080",
    SERVICE_CODES["pillar"]: "#dc646480",
    CELL_STORAGE: "#64aa8c80",
    CELL_PICK: "#93c57299",
    CELL_AISLE_H: "#f5f5f0c1",
    CELL_AISLE_V: "#f5f5f0c1",
    CELL_AISLE_CROSS: "#f5f5f0c1",
}
PLOT_PALETTE = PALETTE
TITLE_WRAP_WIDTH = 88


def _plot_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Return a discrete colormap for known cell codes."""
    max_code = max(PALETTE)
    colors = [PALETTE.get(code, "#ffffff") for code in range(max_code + 1)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, max_code + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _wrapped_title(title: str) -> str:
    """Wrap long plot titles so they do not clip against the legend/right edge."""
    text = title or "Warehouse Layout"
    return "\n".join(
        textwrap.fill(line, width=TITLE_WRAP_WIDTH, break_long_words=False)
        for line in str(text).splitlines()
    )


def plot_layout_grid(
    grid,
    title: str = "",
    show_coords: bool = True,
    save_path: Path | str | None = None,
    pick_face_mask: np.ndarray | None = None,
):
    """Plot a warehouse layout grid and optionally save it to disk."""
    display_grid = grid_to_display_codes(np.asarray(grid))
    if pick_face_mask is not None:
        mask = np.asarray(pick_face_mask, dtype=bool)
        if mask.shape != display_grid.shape:
            raise ValueError("pick_face_mask must have the same shape as grid.")
        display_grid = display_grid.copy()
        display_grid[mask] = CELL_PICK
    rows, cols = display_grid.shape
    cmap, norm = _plot_cmap()

    fig, ax = plt.subplots(figsize=(max(6, cols * 0.45), max(5, rows * 0.45)))
    ax.imshow(display_grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(_wrapped_title(title), fontsize=10, pad=10)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.25, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_coords:
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.tick_params(labelsize=7)
        for row in range(rows):
            for col in range(cols):
                ax.text(
                    col,
                    row,
                    f"{row},{col}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    alpha=0.55,
                )
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        mpatches.Patch(color=PALETTE[code], label=label.replace("_", " ").title())
        for code, label in sorted(EDITOR_CELL_CODES.items())
        if code in PALETTE
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    plt.show()
    return None


__all__ = ["PALETTE", "PLOT_PALETTE", "plot_layout_grid"]
