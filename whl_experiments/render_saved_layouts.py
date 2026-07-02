"""Offline renderer for archived experiment layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from whl_core.constants import CELL_PICK, EDITOR_CELL_CODES
from whl_core.layout_io import grid_to_display_codes
from whl_core.scoring import compute_pick_face_mask
from whl_visualization.layout_plot import PLOT_PALETTE

ARCHIVE_FILTERS = ("all", "rank0", "rank0_to_rank3", "rank0_to_rank4", "selected")
TITLE_FORMATS = ("fields", "metrics_trace")

DEFAULT_TITLE_FIELDS = (
    "seed",
    "generation",
    "rank",
    "candidate_id",
    "trace",
    "storage_total",
    "pick_faces",
    "interior_storage",
    "retrieval_penalty",
)


def load_archive_index(index_path: Path | str) -> list[dict[str, Any]]:
    """Load a JSON archive index."""
    path = Path(index_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("archive index must contain a list of metadata objects.")
    return [dict(item) for item in data]


def _rank_value(item: dict[str, Any]) -> int | None:
    value = item.get("rank")
    if value in {"", None}:
        return None
    return int(value)


def filter_index(
    index: list[dict[str, Any]],
    filter_name: str,
) -> list[dict[str, Any]]:
    """Filter archive metadata entries."""
    if filter_name not in ARCHIVE_FILTERS:
        raise ValueError(f"filter must be one of {ARCHIVE_FILTERS}.")
    if filter_name == "all":
        return list(index)
    if filter_name == "selected":
        return [item for item in index if bool(item.get("selected"))]
    if filter_name == "rank0":
        return [item for item in index if _rank_value(item) == 0]
    rank_max = 4 if filter_name == "rank0_to_rank4" else 3
    return [
        item
        for item in index
        if _rank_value(item) is not None and int(_rank_value(item)) <= rank_max
    ]


def _field_value(item: dict[str, Any], field: str) -> Any:
    if field in item:
        return item.get(field)
    metrics = item.get("metrics", {})
    if isinstance(metrics, dict) and field in metrics:
        return metrics.get(field)
    return ""


def title_for_index_item(
    item: dict[str, Any],
    title_fields: list[str] | None = None,
    title_format: str = "fields",
) -> str:
    """Build a readable figure title from archive metadata."""
    if title_format not in TITLE_FORMATS:
        raise ValueError(f"title_format must be one of {TITLE_FORMATS}.")
    if title_format == "metrics_trace":
        interior_storage = _field_value(item, "interior_storage")
        pick_faces = _field_value(item, "pick_faces")
        retrieval_penalty = _field_value(item, "retrieval_penalty")
        trace = str(_field_value(item, "trace") or "root").replace(" > ", " -> ")
        return (
            f"Interior storage: {interior_storage} | "
            f"Pick faces: {pick_faces} | Retrieval penalty: {retrieval_penalty}\n"
            f"Trace: {trace}"
        )

    fields = title_fields or list(DEFAULT_TITLE_FIELDS)
    parts: list[str] = []
    for field in fields:
        value = _field_value(item, field)
        if value in {"", None}:
            continue
        parts.append(f"{field}={value}")
    if not parts:
        return str(item.get("archive_key", "layout"))

    first_line: list[str] = []
    second_line: list[str] = []
    for part in parts:
        if part.startswith("trace="):
            second_line.append(part)
        else:
            first_line.append(part)
    if second_line:
        return " | ".join(first_line) + "\n" + " | ".join(second_line)
    return " | ".join(first_line)


def _plot_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    max_code = max(PLOT_PALETTE)
    colors = [PLOT_PALETTE.get(code, "#ffffff") for code in range(max_code + 1)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, max_code + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def render_layout_png(
    grid: np.ndarray,
    output_path: Path | str,
    title: str,
    dpi: int = 150,
    pick_face_mask: np.ndarray | None = None,
    show_legend: bool = True,
    show_coords: bool = False,
) -> Path:
    """Render one layout grid to a PNG file."""
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    display_grid = grid_to_display_codes(np.asarray(grid))
    if pick_face_mask is not None:
        mask = np.asarray(pick_face_mask, dtype=bool)
        if mask.shape != display_grid.shape:
            raise ValueError("pick_face_mask must have the same shape as grid.")
        display_grid = display_grid.copy()
        display_grid[mask] = CELL_PICK
    rows, cols = display_grid.shape
    cmap, norm = _plot_cmap()
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.35), max(5, rows * 0.35)))
    ax.imshow(display_grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.2, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    if show_coords:
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.tick_params(axis="both", which="major", labelsize=6, length=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_legend:
        handles = [
            mpatches.Patch(
                color=PLOT_PALETTE[code],
                label=label.replace("_", " ").title(),
            )
            for code, label in sorted(EDITOR_CELL_CODES.items())
            if code in PLOT_PALETTE
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def default_output_dir_for_archive(
    archive: Path | str,
    filter_name: str,
) -> Path:
    """Infer the run-local figure directory for an archive path."""
    archive_path = Path(archive)
    archive_type_by_name = {
        "final_ranked_layouts.npz": "final_ranked",
        "generation_elites.npz": "generation_elites",
        "all_debug_layouts.npz": "all_debug",
        "all_candidates_debug_layouts.npz": "all_candidates_debug",
    }
    archive_type = archive_type_by_name.get(archive_path.name, archive_path.stem)
    return archive_path.parent / "figures" / archive_type / filter_name


def _stored_pick_face_mask(
    layouts: np.lib.npyio.NpzFile,
    item: dict[str, Any],
    layout: np.ndarray,
) -> np.ndarray:
    feature_keys = item.get("feature_keys", {})
    mask_key = ""
    if isinstance(feature_keys, dict):
        mask_key = str(feature_keys.get("pick_face_mask", "") or "")
    if mask_key and mask_key in layouts.files:
        return np.asarray(layouts[mask_key], dtype=bool)
    return compute_pick_face_mask(layout)


def _expected_pick_face_count(item: dict[str, Any]) -> int | None:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("pick_faces")
    if value in {"", None}:
        return None
    return int(value)


def render_saved_layouts(
    archive: Path | str,
    index: Path | str,
    output_dir: Path | str | None = None,
    filter_name: str = "all",
    max_layouts: int | None = None,
    dpi: int = 150,
    title_fields: list[str] | None = None,
    title_format: str = "fields",
    show_legend: bool = True,
    show_coords: bool = False,
) -> list[Path]:
    """Render archived layouts to PNG files without rerunning optimization."""
    if max_layouts is not None and max_layouts < 0:
        raise ValueError("max_layouts must be non-negative or None.")

    archive_path = Path(archive)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir_for_archive(archive_path, filter_name)
    )
    metadata = filter_index(load_archive_index(index), filter_name)
    if max_layouts is not None:
        metadata = metadata[: int(max_layouts)]

    rendered: list[Path] = []
    with np.load(archive_path, allow_pickle=False) as layouts:
        for item in metadata:
            archive_key = str(item.get("archive_key", ""))
            if not archive_key:
                continue
            if archive_key not in layouts.files:
                raise KeyError(f"archive key not found in npz: {archive_key}")
            layout = np.asarray(layouts[archive_key])
            pick_face_mask = _stored_pick_face_mask(layouts, item, layout)
            expected_pick_faces = _expected_pick_face_count(item)
            rendered_pick_faces = int(np.count_nonzero(pick_face_mask))
            if expected_pick_faces is not None and rendered_pick_faces != expected_pick_faces:
                print(
                    "WARNING: rendered pick-face mask count does not match metrics: "
                    f"archive_key={archive_key}, mask_count={rendered_pick_faces}, "
                    f"metrics_pick_faces={expected_pick_faces}"
                )
            filename = f"{_safe_filename(archive_key)}.png"
            title = title_for_index_item(
                item,
                title_fields=title_fields,
                title_format=title_format,
            )
            rendered.append(
                render_layout_png(
                    layout,
                    output_path / filename,
                    title,
                    dpi=dpi,
                    pick_face_mask=pick_face_mask,
                    show_legend=show_legend,
                    show_coords=show_coords,
                )
            )
    return rendered


def _parse_title_fields(value: str | None) -> list[str] | None:
    if not value:
        return None
    fields = [item.strip() for item in value.split(",") if item.strip()]
    return fields or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render archived experiment layouts to PNG files.",
    )
    parser.add_argument("--archive", type=Path, required=True, help="Path to archive .npz file.")
    parser.add_argument("--index", type=Path, required=True, help="Path to archive index JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output figure directory.")
    parser.add_argument("--filter", choices=ARCHIVE_FILTERS, default="all")
    parser.add_argument("--max-layouts", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--title-fields",
        default=None,
        help="Comma-separated metadata/metric fields to include in titles.",
    )
    parser.add_argument(
        "--title-format",
        choices=TITLE_FORMATS,
        default="fields",
        help="Title format. Use metrics_trace for Step 10B archive figures.",
    )
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--show-coords", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rendered = render_saved_layouts(
        archive=args.archive,
        index=args.index,
        output_dir=args.output_dir,
        filter_name=args.filter,
        max_layouts=args.max_layouts,
        dpi=args.dpi,
        title_fields=_parse_title_fields(args.title_fields),
        title_format=args.title_format,
        show_legend=not args.no_legend,
        show_coords=args.show_coords,
    )
    print(f"Rendered layouts: {len(rendered)}")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_FILTERS",
    "DEFAULT_TITLE_FIELDS",
    "TITLE_FORMATS",
    "build_parser",
    "default_output_dir_for_archive",
    "filter_index",
    "load_archive_index",
    "render_layout_png",
    "render_saved_layouts",
    "title_for_index_item",
]
