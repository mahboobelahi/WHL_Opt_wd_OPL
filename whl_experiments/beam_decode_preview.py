"""Beam Search decoder preview helpers for real WHL instance masks.

This is not the full NSGA-II + Beam Search optimizer. It only verifies that a
real ``.npz`` mask can be loaded, converted to a grid, decoded by the standalone
Beam Search decoder, summarized, and optionally visualized.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from whl_algorithms.beam_node import BeamNode, layout_signature
from whl_algorithms.beam_search import BeamSearchConfig, run_beam_search
from whl_algorithms.carving import (
    find_feasible_global_horizontal_starts,
    find_feasible_global_vertical_starts,
)
from whl_algorithms.population import initialize_population
from whl_core.chromosome import Chromosome
from whl_core.feasibility import oriented_aisle_thickness_violations
from whl_core.layout_io import fixed_aisle_mask_from_masks, load_mask, mask_to_grid
from whl_core.scoring import detect_pick_faces
from whl_core.constants import CELL_PICK, CELL_STORAGE

try:
    from whl_core.paths import FIGURES_DIR, MASK_DIR, PROCESSED_RESULTS_DIR, PROJECT_ROOT
except Exception:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MASK_DIR = PROJECT_ROOT / "data" / "instances" / "masks"
    PROCESSED_RESULTS_DIR = PROJECT_ROOT / "data" / "processed_results"
    FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DEFAULT_PREVIEW_CSV = PROCESSED_RESULTS_DIR / "beam_decode_preview_summary.csv"
DEFAULT_PREVIEW_FIGURE_DIR = FIGURES_DIR / "beam_decode_preview"


def _scalar_int(value: Any, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError("Cannot convert None to int without a default.")
        return int(default)
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(value)


def _scalar_text(value: Any, *, default: str = "layout") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def discover_instance_masks(limit: int | None = None, mask_dir: Path | None = None) -> list[Path]:
    """Discover generated ``.npz`` instance masks."""
    directory = Path(mask_dir) if mask_dir is not None else Path(MASK_DIR)
    if not directory.exists():
        return []

    masks = sorted(path for path in directory.glob("*.npz") if path.is_file())
    if limit is not None:
        if int(limit) < 0:
            raise ValueError("limit must be non-negative or None.")
        return masks[: int(limit)]
    return masks


def _fallback_chromosome(rows: int, cols: int) -> Chromosome:
    h_indices: list[int] = []
    v_indices: list[int] = []
    if rows > 2:
        h_indices.append(max(0, min(rows - 1, rows // 2)))
    if cols > 2:
        v_indices.append(max(0, min(cols - 1, cols // 2)))
    if not h_indices and not v_indices:
        h_indices.append(0)
    return Chromosome.from_indices(rows, cols, h_indices, v_indices)


def make_preview_chromosome(rows: int, cols: int, seed: int = 1) -> Chromosome:
    """Create a deterministic dimension-only chromosome for unit tests."""
    rows = _scalar_int(rows)
    cols = _scalar_int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")

    try:
        chromosome = initialize_population(rows, cols, population_size=1, seed=seed)[0]
    except Exception:
        chromosome = _fallback_chromosome(rows, cols)

    chromosome.validate(rows=rows, cols=cols)
    h_count, v_count = chromosome.active_count()
    if h_count + v_count == 0:
        chromosome = _fallback_chromosome(rows, cols)
        chromosome.validate(rows=rows, cols=cols)
    return chromosome


def _sample_feasible_starts(starts: list[int], rng: np.random.Generator, max_count: int) -> list[int]:
    if not starts:
        return []
    count = min(max_count, len(starts))
    if count == len(starts):
        return sorted(starts)
    selected = rng.choice(np.asarray(starts, dtype=int), size=count, replace=False)
    return sorted(int(value) for value in selected.tolist())


def make_preview_chromosome_for_grid(
    grid: np.ndarray,
    aisle_width: int,
    seed: int = 1,
    max_h: int = 2,
    max_v: int = 2,
) -> Chromosome:
    """Create a preview chromosome from feasible global start positions.

    This is deliberately grid-aware. A purely random chromosome may activate rows
    or columns blocked by pillars/restricted cells, producing zero children and
    zero figures. That is a bad preview check, unless the goal is to admire empty
    folders.
    """
    layout = np.asarray(grid)
    if layout.ndim != 2:
        raise ValueError("grid must be a 2D array.")
    if aisle_width <= 0:
        raise ValueError("aisle_width must be positive.")

    rows, cols = layout.shape
    rng = np.random.default_rng(seed)
    h_starts = find_feasible_global_horizontal_starts(layout, aisle_width)
    v_starts = find_feasible_global_vertical_starts(layout, aisle_width)

    h_indices = _sample_feasible_starts(h_starts, rng, max_h)
    v_indices = _sample_feasible_starts(v_starts, rng, max_v)

    if not h_indices and not v_indices:
        chromosome = make_preview_chromosome(rows, cols, seed=seed)
    else:
        chromosome = Chromosome.from_indices(rows, cols, h_indices, v_indices)
    chromosome.validate(rows=rows, cols=cols)
    return chromosome


def decode_mask_with_beam_search(
    mask_path: Path | str,
    seed: int = 1,
    beam_width: int = 3,
    max_depth: int = 3,
) -> list[BeamNode]:
    """Decode one real instance mask using the Beam Search skeleton."""
    path = Path(mask_path)
    masks = load_mask(path)
    grid = mask_to_grid(masks)

    aisle_width = _scalar_int(masks.get("aisle_width"), default=1)
    chromosome = make_preview_chromosome_for_grid(grid, aisle_width, seed=seed)
    config = BeamSearchConfig(
        aisle_width=aisle_width,
        beam_width=int(beam_width),
        max_depth=int(max_depth),
    )
    rng = np.random.default_rng(seed)
    return run_beam_search(chromosome, grid, config, rng=rng)


def _metrics_value(metrics: dict[str, Any], key: str, default: Any = "") -> Any:
    value = metrics.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_candidates(
    mask_path: Path | str,
    candidates: list[BeamNode],
    seed: int,
) -> list[dict[str, Any]]:
    """Summarize candidate BeamNodes into CSV-friendly dictionaries."""
    path = Path(mask_path)
    try:
        masks = load_mask(path)
        instance_name = _scalar_text(masks.get("name"), default=path.stem)
        aisle_width = _scalar_int(masks.get("aisle_width"), default=1)
        fixed_aisle_mask = fixed_aisle_mask_from_masks(masks)
    except Exception:
        instance_name = path.stem
        aisle_width = 1
        fixed_aisle_mask = None

    if not candidates:
        return [
            {
                "filename": path.name,
                "instance_name": instance_name,
                "seed": int(seed),
                "candidate_id": -1,
                "status": "no_candidates",
                "depth": "",
                "action": "",
                "trace": "",
                "storage_total": "",
                "pick_faces": "",
                "interior_storage": "",
                "retrieval_penalty": "",
                "door_connectivity_index": "",
                "aisle_components": "",
                "has_door_connected_aisle": "",
                "scalar_score": "",
                "exact_width_ok": "",
                "exact_width_violation_count": "",
                "exact_width_violations": "",
            }
        ]

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        metrics = dict(candidate.metrics or {})
        exact_width_violations = oriented_aisle_thickness_violations(
            candidate.layout,
            aisle_width,
            exact=True,
            fixed_aisle_mask=fixed_aisle_mask,
        )
        rows.append(
            {
                "filename": path.name,
                "instance_name": instance_name,
                "seed": int(seed),
                "candidate_id": idx,
                "status": "ok",
                "depth": candidate.depth,
                "action": candidate.action,
                "trace": " > ".join(str(item) for item in candidate.trace),
                "storage_total": _metrics_value(metrics, "storage_total"),
                "pick_faces": _metrics_value(metrics, "pick_faces"),
                "interior_storage": _metrics_value(metrics, "interior_storage"),
                "retrieval_penalty": _metrics_value(metrics, "retrieval_penalty"),
                "door_connectivity_index": _metrics_value(metrics, "door_connectivity_index"),
                "aisle_components": _metrics_value(metrics, "aisle_components"),
                "has_door_connected_aisle": _metrics_value(metrics, "has_door_connected_aisle"),
                "scalar_score": _metrics_value(metrics, "scalar_score"),
                "exact_width_ok": not exact_width_violations,
                "exact_width_violation_count": len(exact_width_violations),
                "exact_width_violations": " | ".join(exact_width_violations[:5]),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path



def layout_with_pick_faces_for_preview(grid: np.ndarray) -> np.ndarray:
    """Return a display-only layout copy with access-direction pick faces painted.

    Scoring remains side-effect free. This helper uses the same pick-face
    detector as ``score_layout`` and only paints a copy for figures.
    """
    preview = np.asarray(grid).copy()
    for row, col in detect_pick_faces(preview):
        if int(preview[row, col]) == CELL_STORAGE:
            preview[row, col] = CELL_PICK
    return preview

def _save_candidate_figure(candidate: BeamNode, output_path: Path, title: str) -> None:
    from whl_visualization.layout_plot import plot_layout_grid

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_layout = layout_with_pick_faces_for_preview(candidate.layout)

    try:
        plot_layout_grid(
            preview_layout,
            title=title,
            show_coords=True,
            save_path=output_path,
        )
    except TypeError:
        plot_layout_grid(preview_layout, title=title, save_path=output_path)

    try:  # pragma: no cover
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def run_beam_decode_preview(
    limit: int | None = 3,
    seed: int = 1,
    beam_width: int = 3,
    max_depth: int = 3,
    save_csv: bool = True,
    save_figures: bool = True,
    output_csv: Path | None = None,
    figure_dir: Path | None = None,
    mask_dir: Path | None = None,
) -> Path | None:
    """Run Beam Search decoding preview checks on real instance masks."""
    masks = discover_instance_masks(limit=limit, mask_dir=mask_dir)
    if not masks:
        raise FileNotFoundError(f"No .npz masks found in {Path(mask_dir) if mask_dir else MASK_DIR}")

    csv_path = Path(output_csv) if output_csv is not None else DEFAULT_PREVIEW_CSV
    image_dir = Path(figure_dir) if figure_dir is not None else DEFAULT_PREVIEW_FIGURE_DIR

    all_rows: list[dict[str, Any]] = []
    for mask_path in masks:
        try:
            candidates = decode_mask_with_beam_search(
                mask_path,
                seed=seed,
                beam_width=beam_width,
                max_depth=max_depth,
            )
            all_rows.extend(summarize_candidates(mask_path, candidates, seed=seed))

            if save_figures and candidates:
                for candidate_id, candidate in enumerate(candidates, start=1):
                    figure_path = image_dir / (
                        f"{mask_path.stem}_seed_{seed}_candidate_{candidate_id:03d}.png"
                    )
                    title = f"{mask_path.stem} | seed={seed} | candidate={candidate_id}"
                    _save_candidate_figure(candidate, figure_path, title)
        except Exception as exc:
            all_rows.append(
                {
                    "filename": mask_path.name,
                    "instance_name": mask_path.stem,
                    "seed": int(seed),
                    "candidate_id": -1,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "depth": "",
                    "action": "",
                    "trace": "",
                    "storage_total": "",
                    "pick_faces": "",
                    "interior_storage": "",
                    "retrieval_penalty": "",
                    "door_connectivity_index": "",
                    "aisle_components": "",
                    "has_door_connected_aisle": "",
                    "scalar_score": "",
                    "exact_width_ok": "",
                    "exact_width_violation_count": "",
                    "exact_width_violations": "",
                }
            )

    if save_csv:
        return _write_csv(all_rows, csv_path)
    return None


def candidate_layout_signatures(candidates: list[BeamNode]) -> list[bytes]:
    """Return stable layout signatures for candidate lists."""
    return [layout_signature(candidate.layout) for candidate in candidates]


__all__ = [
    "DEFAULT_PREVIEW_CSV",
    "DEFAULT_PREVIEW_FIGURE_DIR",
    "candidate_layout_signatures",
    "decode_mask_with_beam_search",
    "discover_instance_masks",
    "layout_with_pick_faces_for_preview",
    "make_preview_chromosome",
    "make_preview_chromosome_for_grid",
    "run_beam_decode_preview",
    "summarize_candidates",
]
