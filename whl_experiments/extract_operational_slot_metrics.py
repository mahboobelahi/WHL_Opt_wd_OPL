"""Lock the L1-L4 operational layout panel and extract static slot metrics."""

from __future__ import annotations

import csv
import heapq
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from whl_core.blocks import detect_storage_blocks, neighbors4
from whl_core.constants import AISLE_CODES, CELL_PICK, CELL_STORAGE, SERVICE_CODES, STORAGE_LIKE_CODES
from whl_core.feasibility import access_anchor_mask_from_grid_and_masks
from whl_core.layout_io import grid_to_display_codes
from whl_core.scoring import assign_pick_face_access_sides, compute_pick_face_mask
from whl_visualization.layout_plot import _plot_cmap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = PROJECT_ROOT / "data" / "operational_layer"
DATA_ROOT = OP_ROOT / "paper_inputs"
LOG_ROOT = OP_ROOT / "paper_outputs" / "logs"
FIG_ROOT = OP_ROOT / "layout_panel"
DOC_ROOT = PROJECT_ROOT / "docs"

SCREENING_CSV = DATA_ROOT / "candidate_layout_screening.csv"
REVIEW_CSV = DATA_ROOT / "unique38_layout_review_table.csv"
SELECTED_CSV = DATA_ROOT / "selected_layouts.csv"
CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"

FINAL_PANEL_PNG = FIG_ROOT / "selected_layouts_panel_final_L1_L4.png"
M3C_SUMMARY_JSON = LOG_ROOT / "m3C_final_selection_lock_summary.json"
M3C_REPORT_MD = DOC_ROOT / "098C_operational_layout_selection_final_L1_L4_report.md"

SLOT_METRICS_CSV = DATA_ROOT / "slot_metrics_by_layout.csv"
M4_SUMMARY_JSON = LOG_ROOT / "m4_slot_metrics_summary.json"
M4_REPORT_MD = DOC_ROOT / "094_operational_layer_slot_metrics.md"

INSTANCE_NAME = "Gyorgy-KOVACS_WH_Narrow_AW_4"
INSTANCE_SLUG = "KOV_WH_N_AW4"
METHOD_NAME = "proposed_nsga2_bs"
METHOD_SLUG = "nsga2"
PHASE = "p11"

SELECTED_COLUMNS = [
    "selection_label",
    "selection_type",
    "selection_reason_short",
    "visual_selection_note",
    "layout_signature_short",
    "layout_signature",
    "seed",
    "rank",
    "candidate_id",
    "layout_id",
    "archive_key",
    "storage_cells",
    "pick_faces",
    "interior_deep_storage_cells",
    "interior_deep_storage_share",
    "retrieval_penalty",
    "mean_depth",
    "max_depth",
    "largest_block_size",
    "pick_face_density",
    "pallet_slot_capacity",
    "objective_vector",
    "selection_score",
    "door_connectivity_index",
    "access_anchor_connectivity_index",
    "is_feasible",
    "visual_confirmation_status",
    "visual_confirmation_note",
    "phase",
    "method_slug",
    "method_name",
    "instance_slug",
    "instance_name",
    "archive_entry_index",
    "source_generation",
    "source_candidate_index",
    "run_folder",
    "archive_index_path",
    "archive_npz_path",
    "candidates_csv_path",
]

SLOT_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "candidate_id",
    "layout_id",
    "archive_key",
    "row",
    "col",
    "level",
    "block_id",
    "block_size",
    "access_type",
    "access_side_count",
    "access_sides",
    "effective_access_side",
    "effective_pick_face_row",
    "effective_pick_face_col",
    "effective_depth",
    "horizontal_access_distance",
    "vertical_level",
    "normalized_distance",
    "normalized_depth",
    "normalized_level",
    "slot_cost",
    "storage_cells_layout",
    "pick_faces_layout",
    "interior_deep_storage_share_layout",
    "retrieval_penalty_layout",
    "mean_depth_layout",
    "max_depth_layout",
    "largest_block_size_layout",
    "pallet_slot_capacity_layout",
    "slot_metric_status",
    "slot_metric_warning",
]

FINAL_SELECTIONS = [
    {
        "selection_label": "L1",
        "selection_type": "accessibility-oriented",
        "signature": "1fa9344c00a95c630e382533991ef575b2f5e6b5",
        "score_field": "score_L1_accessibility",
        "seed": 101,
        "rank": 0,
        "storage_cells": 560,
        "pick_faces": 270,
        "interior_deep_storage_share": 0.518,
        "retrieval_penalty": 341,
        "mean_depth": 1.84,
        "max_depth": 5,
        "pallet_slot_capacity": 4480,
        "reason": (
            "Accessibility-oriented rank-0 layout with the highest pick-face availability "
            "and lowest retrieval burden among the selected archetypes."
        ),
        "visual_note": "Accessibility-first structural archetype with broad pick-face exposure.",
    },
    {
        "selection_label": "L2",
        "selection_type": "deep-block / reserve-oriented",
        "signature": "76cdc821160b1fd8a5952575af463e70cb84ba4e",
        "score_field": "score_L2_dense_deep",
        "seed": 120,
        "rank": 3,
        "storage_cells": 632,
        "pick_faces": 237,
        "interior_deep_storage_share": 0.625,
        "retrieval_penalty": 517,
        "mean_depth": 2.09,
        "max_depth": 5,
        "pallet_slot_capacity": 5056,
        "reason": (
            "Deep-block / reserve-oriented diagnostic contrast selected for stronger "
            "interior/deep-storage structure, fewer pick faces, and higher retrieval "
            "burden; rank is used for traceability only."
        ),
        "visual_note": "Reserve-oriented contrast with stronger deep-block structure.",
    },
    {
        "selection_label": "L3",
        "selection_type": "high-capacity balanced",
        "signature": "fc4825140fa5e2560776b4b932f5ef46c1588f36",
        "score_field": "score_L3_high_capacity_balanced",
        "seed": 105,
        "rank": 0,
        "storage_cells": 680,
        "pick_faces": 262,
        "interior_deep_storage_share": 0.615,
        "retrieval_penalty": 472,
        "mean_depth": 1.85,
        "max_depth": 4,
        "pallet_slot_capacity": 5440,
        "reason": (
            "High-capacity balanced rank-0 layout with strong storage capacity and a "
            "regular access structure, while avoiding the stronger retrieval burden of L2."
        ),
        "visual_note": "High-capacity structural archetype with balanced access burden.",
    },
    {
        "selection_label": "L4",
        "selection_type": "intermediate accessibility-capacity compromise",
        "signature": "31618e0fa9e7ba38a349d70c4dff96d8a35cbd09",
        "score_field": "score_L4_intermediate",
        "seed": 105,
        "rank": 0,
        "storage_cells": 600,
        "pick_faces": 256,
        "interior_deep_storage_share": 0.573,
        "retrieval_penalty": 444,
        "mean_depth": 1.88,
        "max_depth": 5,
        "pallet_slot_capacity": 4800,
        "reason": (
            "Intermediate accessibility-capacity compromise selected as a bridge between "
            "L1 and the high-capacity/deep-storage archetypes."
        ),
        "visual_note": "Intermediate compromise between accessibility and capacity.",
    },
]

SIDE_OFFSETS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_LABELS = {"top": "N", "bottom": "S", "left": "W", "right": "E"}
SIDE_ORDER = {"N": 0, "S": 1, "E": 2, "W": 3}


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def rel_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def numeric(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in {"", None}:
        return math.nan
    return float(value)


def int_numeric(row: dict[str, str], field: str) -> int:
    return int(round(float(row[field])))


def load_config() -> dict[str, Any]:
    if not CONFIG_JSON.is_file():
        raise FileNotFoundError(f"Missing operational config: {as_posix(CONFIG_JSON)}")
    return json.loads(CONFIG_JSON.read_text(encoding="utf-8"))


def load_layout(row: dict[str, str]) -> np.ndarray:
    archive_path = Path(row["archive_npz_path"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing archive npz: {as_posix(archive_path)}")
    with np.load(archive_path) as archive:
        key = row["archive_key"]
        if key not in archive.files:
            raise KeyError(f"archive key {key!r} not in {as_posix(archive_path)}")
        return np.asarray(archive[key])


def normalize_for_blocks(layout: np.ndarray) -> np.ndarray:
    normalized = np.asarray(layout).copy()
    normalized[normalized == CELL_PICK] = CELL_STORAGE
    return normalized


def validate_review_row(row: dict[str, str], spec: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    checks = [
        ("seed", int_numeric(row, "seed"), spec["seed"]),
        ("rank", int_numeric(row, "rank"), spec["rank"]),
        ("storage_cells", int_numeric(row, "storage_cells"), spec["storage_cells"]),
        ("pick_faces", int_numeric(row, "pick_faces"), spec["pick_faces"]),
        ("retrieval_penalty", int_numeric(row, "retrieval_penalty"), spec["retrieval_penalty"]),
        ("max_depth", int_numeric(row, "max_depth"), spec["max_depth"]),
        ("pallet_slot_capacity", int_numeric(row, "pallet_slot_capacity"), spec["pallet_slot_capacity"]),
    ]
    for field, observed, expected in checks:
        if observed != expected:
            warnings.append(
                f"{spec['selection_label']} {field} mismatch: observed {observed}, expected {expected}"
            )
    float_checks = [
        ("interior_deep_storage_share", numeric(row, "interior_deep_storage_share"), spec["interior_deep_storage_share"], 0.002),
        ("mean_depth", numeric(row, "mean_depth"), spec["mean_depth"], 0.01),
    ]
    for field, observed, expected, tolerance in float_checks:
        if not math.isfinite(observed) or abs(observed - expected) > tolerance:
            warnings.append(
                f"{spec['selection_label']} {field} mismatch: observed {observed}, expected about {expected}"
            )
    return warnings


def lock_final_selection() -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not SCREENING_CSV.is_file():
        raise FileNotFoundError(f"Missing candidate screening CSV: {as_posix(SCREENING_CSV)}")
    if not REVIEW_CSV.is_file():
        raise FileNotFoundError(f"Missing unique38 review table: {as_posix(REVIEW_CSV)}")

    review_rows = read_rows(REVIEW_CSV)
    by_signature = {row["layout_signature"]: row for row in review_rows}

    selected: list[dict[str, str]] = []
    warnings: list[str] = []
    for spec in FINAL_SELECTIONS:
        signature = spec["signature"]
        if signature not in by_signature:
            raise RuntimeError(f"Final signature is missing from unique38 review table: {signature}")
        source = by_signature[signature]
        warnings.extend(validate_review_row(source, spec))
        selected_row = {column: source.get(column, "") for column in SELECTED_COLUMNS}
        selected_row.update(
            {
                "selection_label": spec["selection_label"],
                "selection_type": spec["selection_type"],
                "selection_reason_short": spec["reason"],
                "visual_selection_note": spec["visual_note"],
                "layout_signature_short": signature[:12],
                "layout_signature": signature,
                "selection_score": source.get(spec["score_field"], ""),
                "visual_confirmation_status": "confirmed",
                "visual_confirmation_note": (
                    "Final L1-L4 panel rendered with standard WHL grid colors, gridlines, and pick-face overlay."
                ),
            }
        )
        selected.append(selected_row)

    expected_labels = [spec["selection_label"] for spec in FINAL_SELECTIONS]
    expected_signatures = [spec["signature"] for spec in FINAL_SELECTIONS]
    if [row["selection_label"] for row in selected] != expected_labels:
        warnings.append("selected layout labels are not in exact L1-L4 order")
    if [row["layout_signature"] for row in selected] != expected_signatures:
        warnings.append("selected layout signatures do not match the final locked signature list")

    panel_note = render_panel(selected, FINAL_PANEL_PNG)
    for row in selected:
        row["visual_confirmation_note"] = panel_note
    write_csv(SELECTED_CSV, selected, SELECTED_COLUMNS)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Lock final operational-layer L1-L4 layout signatures for KOV Narrow AW4.",
        "input_screening_csv": as_posix(SCREENING_CSV),
        "input_unique38_review_csv": as_posix(REVIEW_CSV),
        "updated_selected_layouts_csv": as_posix(SELECTED_CSV),
        "visual_panel_path": as_posix(FINAL_PANEL_PNG),
        "summary_json_path": as_posix(M3C_SUMMARY_JSON),
        "markdown_report_path": as_posix(M3C_REPORT_MD),
        "final_selection_interpretation": (
            "Selected layouts are structural archetypes, not Pareto-rank representatives; "
            "archive ranks are used for traceability only."
        ),
        "selected_layouts": [
            {
                "selection_label": row["selection_label"],
                "selection_type": row["selection_type"],
                "layout_signature": row["layout_signature"],
                "seed": int_numeric(row, "seed"),
                "rank": int_numeric(row, "rank"),
                "storage_cells": int_numeric(row, "storage_cells"),
                "pick_faces": int_numeric(row, "pick_faces"),
                "interior_deep_storage_share": numeric(row, "interior_deep_storage_share"),
                "retrieval_penalty": numeric(row, "retrieval_penalty"),
                "mean_depth": numeric(row, "mean_depth"),
                "max_depth": int_numeric(row, "max_depth"),
                "pallet_slot_capacity": int_numeric(row, "pallet_slot_capacity"),
            }
            for row in selected
        ],
        "validation": {
            "selected_layout_count": len(selected),
            "expected_labels": expected_labels,
            "observed_labels": [row["selection_label"] for row in selected],
            "expected_signatures": expected_signatures,
            "observed_signatures": [row["layout_signature"] for row in selected],
            "max_rank": max(int_numeric(row, "rank") for row in selected),
            "rank_profile": {"rank_0_count": 3, "rank_3_count": 1},
            "panel_created": FINAL_PANEL_PNG.is_file(),
            "selected_csv_column_order": SELECTED_COLUMNS,
        },
        "warnings": warnings,
        "ready_for_slot_metric_extraction": not warnings,
        "what_was_not_implemented": [
            "SKU generation",
            "order generation",
            "representative or reserve pallet assignment",
            "Regime A/B metrics",
            "routing",
            "batching",
            "picker/forklift simulation",
            "replenishment",
            "honeycombing",
            "slotting optimization",
            "optimizer edits",
            "Phase 11 archive edits",
        ],
    }
    M3C_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    M3C_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_m3c_report(summary, selected)

    if warnings:
        raise RuntimeError("M3C final selection validation failed: " + "; ".join(warnings))
    return selected, summary


def add_gridlines(ax: Any, grid: np.ndarray) -> None:
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#d7d7d7", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def render_panel(selected: list[dict[str, str]], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap, norm_obj = _plot_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    for ax, row in zip(axes.ravel(), selected, strict=True):
        grid = load_layout(row)
        display = grid_to_display_codes(grid).copy()
        display[compute_pick_face_mask(grid)] = CELL_PICK
        ax.imshow(display, cmap=cmap, norm=norm_obj, interpolation="nearest")
        add_gridlines(ax, grid)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        title = (
            f"{row['selection_label']} {row['selection_type']}\n"
            f"seed {row['seed']} rank {row['rank']} | "
            f"S={row['storage_cells']} PF={row['pick_faces']} "
            f"deep={float(row['interior_deep_storage_share']):.3f}\n"
            f"RP={float(row['retrieval_penalty']):.0f} "
            f"meanD={float(row['mean_depth']):.2f} "
            f"maxD={row['max_depth']} cap={row['pallet_slot_capacity']}"
        )
        ax.set_title(title, fontsize=9)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return "Rendered with standard WHL grid colors, gridlines, and pick-face overlay."


def inside(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols


def access_mask_for_layout(layout: np.ndarray) -> np.ndarray:
    return np.isin(layout, list(AISLE_CODES)) | (layout == SERVICE_CODES["door"])


def anchor_distances(layout: np.ndarray) -> np.ndarray:
    access_mask = access_mask_for_layout(layout)
    anchors = access_anchor_mask_from_grid_and_masks(layout)
    if not np.any(anchors):
        raise RuntimeError("No door/access-anchor cells found for horizontal access distance BFS.")

    rows, cols = layout.shape
    distances = np.full(layout.shape, math.inf, dtype=float)
    queue: deque[tuple[int, int]] = deque()
    for row, col in zip(*np.where(anchors & access_mask), strict=True):
        distances[row, col] = 0.0
        queue.append((int(row), int(col)))

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in neighbors4(row, col, rows, cols):
            if not access_mask[next_row, next_col]:
                continue
            if math.isfinite(distances[next_row, next_col]):
                continue
            distances[next_row, next_col] = distances[row, col] + 1.0
            queue.append((next_row, next_col))
    return distances


def block_sources(
    layout: np.ndarray,
    block: Any,
    access_distances: np.ndarray,
) -> list[dict[str, Any]]:
    rows, cols = layout.shape
    access_mask = access_mask_for_layout(layout)
    selected_sides = sorted(
        getattr(block, "pick_face_side_names", frozenset()),
        key=lambda side: SIDE_ORDER[SIDE_LABELS[side]],
    )
    sources: list[dict[str, Any]] = []
    for row, col in sorted(getattr(block, "pick_faces", [])):
        for side in selected_sides:
            row_delta, col_delta = SIDE_OFFSETS[side]
            access_row = row + row_delta
            access_col = col + col_delta
            if not inside(access_row, access_col, rows, cols):
                continue
            if not access_mask[access_row, access_col]:
                continue
            side_label = SIDE_LABELS[side]
            sources.append(
                {
                    "cell": (int(row), int(col)),
                    "side": side_label,
                    "pick_face_row": int(row),
                    "pick_face_col": int(col),
                    "horizontal_access_distance": float(access_distances[access_row, access_col]),
                    "adjacent_access_cell": (int(access_row), int(access_col)),
                }
            )
    return sources


def assign_effective_access(block: Any, sources: list[dict[str, Any]], shape: tuple[int, int]) -> dict[tuple[int, int], dict[str, Any]]:
    rows, cols = shape
    block_cells = {tuple(cell) for cell in block.cells}
    assignments: dict[tuple[int, int], dict[str, Any]] = {}
    heap: list[tuple[float, float, int, int, int, int, int, int, dict[str, Any]]] = []

    for source_index, source in enumerate(sources):
        row, col = source["cell"]
        hdist = source["horizontal_access_distance"]
        hkey = hdist if math.isfinite(hdist) else math.inf
        heapq.heappush(
            heap,
            (
                1.0,
                hkey,
                SIDE_ORDER[source["side"]],
                source["pick_face_row"],
                source["pick_face_col"],
                row,
                col,
                source_index,
                source,
            ),
        )

    while heap:
        depth, hkey, side_key, pf_row, pf_col, row, col, source_index, source = heapq.heappop(heap)
        cell = (row, col)
        if cell in assignments:
            continue
        assignments[cell] = {
            "effective_access_side": source["side"],
            "effective_pick_face_row": source["pick_face_row"],
            "effective_pick_face_col": source["pick_face_col"],
            "effective_depth": int(depth),
            "horizontal_access_distance": source["horizontal_access_distance"],
        }
        for next_row, next_col in neighbors4(row, col, rows, cols):
            if (next_row, next_col) not in block_cells:
                continue
            if (next_row, next_col) in assignments:
                continue
            heapq.heappush(
                heap,
                (
                    depth + 1.0,
                    hkey,
                    side_key,
                    pf_row,
                    pf_col,
                    next_row,
                    next_col,
                    source_index,
                    source,
                ),
            )
    return assignments


def layout_slot_rows(selected_row: dict[str, str], vertical_levels: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layout = load_layout(selected_row)
    measured_layout = normalize_for_blocks(layout)
    storage_mask = np.isin(measured_layout, list(STORAGE_LIKE_CODES))
    storage_count = int(storage_mask.sum())
    expected_storage_count = int_numeric(selected_row, "storage_cells")
    if storage_count != expected_storage_count:
        raise RuntimeError(
            f"{selected_row['selection_label']} storage-cell mismatch: "
            f"archive has {storage_count}, selected CSV has {expected_storage_count}"
        )

    blocks = detect_storage_blocks(measured_layout)
    assign_pick_face_access_sides(measured_layout, blocks)
    access_distances = anchor_distances(measured_layout)

    cell_to_block: dict[tuple[int, int], Any] = {}
    for block in blocks:
        for cell in block.cells:
            cell_to_block[tuple(cell)] = block

    missing_block_cells = [
        (int(row), int(col))
        for row, col in zip(*np.where(storage_mask), strict=True)
        if (int(row), int(col)) not in cell_to_block
    ]
    if missing_block_cells:
        raise RuntimeError(
            f"{selected_row['selection_label']} has {len(missing_block_cells)} storage cells without block_id"
        )

    block_summaries: dict[int, dict[str, Any]] = {}
    for block in blocks:
        sources = block_sources(measured_layout, block, access_distances)
        access_sides = sorted({source["side"] for source in sources}, key=lambda side: SIDE_ORDER[side])
        assignments = assign_effective_access(block, sources, measured_layout.shape) if sources else {}
        access_type = "no_access"
        if len(access_sides) == 1:
            access_type = "one_sided"
        elif len(access_sides) >= 2:
            access_type = "two_sided"
        block_summaries[int(block.id)] = {
            "block": block,
            "sources": sources,
            "assignments": assignments,
            "access_sides": access_sides,
            "access_type": access_type,
        }

    rows: list[dict[str, Any]] = []
    warnings: set[str] = set()
    for storage_row, storage_col in sorted(zip(*np.where(storage_mask), strict=True)):
        cell = (int(storage_row), int(storage_col))
        block = cell_to_block[cell]
        block_info = block_summaries[int(block.id)]
        assignment = block_info["assignments"].get(cell)
        warning = ""
        status = "ok"
        if assignment is None:
            warning = "no official pick-face access source reached this storage cell"
            status = "warning"
            warnings.add(warning)
            assignment = {
                "effective_access_side": "",
                "effective_pick_face_row": "",
                "effective_pick_face_col": "",
                "effective_depth": "",
                "horizontal_access_distance": "",
            }
        elif not math.isfinite(float(assignment["horizontal_access_distance"])):
            warning = "effective pick-face access side is not reachable from door/access anchors"
            status = "warning"
            warnings.add(warning)

        for level in range(vertical_levels):
            rows.append(
                {
                    "selection_label": selected_row["selection_label"],
                    "selection_type": selected_row["selection_type"],
                    "layout_signature": selected_row["layout_signature"],
                    "seed": selected_row["seed"],
                    "rank": selected_row["rank"],
                    "candidate_id": selected_row["candidate_id"],
                    "layout_id": selected_row["layout_id"],
                    "archive_key": selected_row["archive_key"],
                    "row": cell[0],
                    "col": cell[1],
                    "level": level,
                    "block_id": int(block.id),
                    "block_size": int(block.cell_count),
                    "access_type": block_info["access_type"],
                    "access_side_count": len(block_info["access_sides"]),
                    "access_sides": ",".join(block_info["access_sides"]),
                    "effective_access_side": assignment["effective_access_side"],
                    "effective_pick_face_row": assignment["effective_pick_face_row"],
                    "effective_pick_face_col": assignment["effective_pick_face_col"],
                    "effective_depth": assignment["effective_depth"],
                    "horizontal_access_distance": assignment["horizontal_access_distance"],
                    "vertical_level": level,
                    "normalized_distance": "",
                    "normalized_depth": "",
                    "normalized_level": "",
                    "slot_cost": "",
                    "storage_cells_layout": selected_row["storage_cells"],
                    "pick_faces_layout": selected_row["pick_faces"],
                    "interior_deep_storage_share_layout": selected_row["interior_deep_storage_share"],
                    "retrieval_penalty_layout": selected_row["retrieval_penalty"],
                    "mean_depth_layout": selected_row["mean_depth"],
                    "max_depth_layout": selected_row["max_depth"],
                    "largest_block_size_layout": selected_row["largest_block_size"],
                    "pallet_slot_capacity_layout": selected_row["pallet_slot_capacity"],
                    "slot_metric_status": status,
                    "slot_metric_warning": warning,
                }
            )

    summary = {
        "selection_label": selected_row["selection_label"],
        "storage_cells": storage_count,
        "expected_slots": storage_count * vertical_levels,
        "observed_slots": len(rows),
        "block_count": len(blocks),
        "no_access_block_count": sum(
            1 for block_info in block_summaries.values() if block_info["access_type"] == "no_access"
        ),
        "warning_count": sum(1 for row in rows if row["slot_metric_status"] != "ok"),
        "warnings": sorted(warnings),
    }
    return rows, summary


def normalize_slot_rows(rows: list[dict[str, Any]], vertical_levels: int) -> dict[str, Any]:
    finite_distances = [
        float(row["horizontal_access_distance"])
        for row in rows
        if row["horizontal_access_distance"] != "" and math.isfinite(float(row["horizontal_access_distance"]))
    ]
    finite_depths = [
        float(row["effective_depth"])
        for row in rows
        if row["effective_depth"] != "" and math.isfinite(float(row["effective_depth"]))
    ]
    max_distance = max(finite_distances) if finite_distances else 0.0
    max_depth = max(finite_depths) if finite_depths else 1.0
    max_level = max(1, vertical_levels - 1)

    for row in rows:
        normalized_level = float(row["level"]) / max_level
        row["normalized_level"] = stringify(normalized_level)
        if row["horizontal_access_distance"] == "" or row["effective_depth"] == "":
            continue
        distance = float(row["horizontal_access_distance"])
        depth = float(row["effective_depth"])
        if not math.isfinite(distance) or not math.isfinite(depth):
            continue
        normalized_distance = 0.0 if max_distance <= 0 else distance / max_distance
        normalized_depth = 0.0 if max_depth <= 1 else (depth - 1.0) / (max_depth - 1.0)
        slot_cost = normalized_distance + normalized_depth + normalized_level
        row["normalized_distance"] = stringify(normalized_distance)
        row["normalized_depth"] = stringify(normalized_depth)
        row["slot_cost"] = stringify(slot_cost)

    return {
        "max_horizontal_access_distance": max_distance,
        "max_effective_depth": max_depth,
        "max_vertical_level": vertical_levels - 1,
    }


def extract_slot_metrics(selected: list[dict[str, str]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vertical_levels = int(config.get("vertical_levels", 0))
    if vertical_levels != 8:
        raise RuntimeError(f"Expected vertical_levels=8, observed {vertical_levels}")

    expected_labels = [spec["selection_label"] for spec in FINAL_SELECTIONS]
    expected_signatures = [spec["signature"] for spec in FINAL_SELECTIONS]
    if [row["selection_label"] for row in selected] != expected_labels:
        raise RuntimeError("selected_layouts.csv labels are not exactly L1-L4")
    if [row["layout_signature"] for row in selected] != expected_signatures:
        raise RuntimeError("selected_layouts.csv signatures do not match the final locked signatures")

    all_rows: list[dict[str, Any]] = []
    layout_summaries: list[dict[str, Any]] = []
    for selected_row in selected:
        layout_rows, layout_summary = layout_slot_rows(selected_row, vertical_levels)
        all_rows.extend(layout_rows)
        layout_summaries.append(layout_summary)

    normalization = normalize_slot_rows(all_rows, vertical_levels)
    write_csv(SLOT_METRICS_CSV, all_rows, SLOT_COLUMNS)

    warnings = validate_slot_output(all_rows, selected, vertical_levels, layout_summaries)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Extract static slot structural/access descriptors for final L1-L4 operational layouts.",
        "input_selected_layouts_csv": as_posix(SELECTED_CSV),
        "input_operational_config_json": as_posix(CONFIG_JSON),
        "slot_metrics_csv": as_posix(SLOT_METRICS_CSV),
        "summary_json_path": as_posix(M4_SUMMARY_JSON),
        "markdown_report_path": as_posix(M4_REPORT_MD),
        "vertical_levels": vertical_levels,
        "level_range": [0, vertical_levels - 1],
        "normalization": normalization,
        "slot_count_total": len(all_rows),
        "expected_slot_count_total": sum(int_numeric(row, "storage_cells") * vertical_levels for row in selected),
        "expected_slot_count_prompt": 19776,
        "layout_summaries": layout_summaries,
        "validation": {
            "labels_exact_L1_L4": [row["selection_label"] for row in selected] == expected_labels,
            "signatures_exact": [row["layout_signature"] for row in selected] == expected_signatures,
            "max_selected_rank": max(int_numeric(row, "rank") for row in selected),
            "levels_exact_0_to_7": sorted({int(row["level"]) for row in all_rows}) == list(range(vertical_levels)),
            "all_storage_cells_have_block_id": all(row["block_id"] != "" for row in all_rows),
            "all_accessible_slots_have_finite_depth_and_distance": all(
                row["slot_metric_status"] != "ok"
                or (
                    row["effective_depth"] != ""
                    and row["horizontal_access_distance"] != ""
                    and math.isfinite(float(row["effective_depth"]))
                    and math.isfinite(float(row["horizontal_access_distance"]))
                )
                for row in all_rows
            ),
            "normalized_values_in_unit_interval": normalized_values_in_unit_interval(all_rows),
            "slot_counts_by_layout": {
                row["selection_label"]: sum(1 for slot_row in all_rows if slot_row["selection_label"] == row["selection_label"])
                for row in selected
            },
            "expected_slot_counts_by_layout": {
                row["selection_label"]: int_numeric(row, "storage_cells") * vertical_levels for row in selected
            },
        },
        "ready_for_milestone_5": not warnings,
        "warnings": warnings,
        "what_was_not_implemented": [
            "SKU generation",
            "order generation",
            "representative or reserve pallet assignment",
            "Regime A/B metrics",
            "routing",
            "batching",
            "picker/forklift simulation",
            "replenishment",
            "honeycombing",
            "slotting optimization",
            "optimizer edits",
            "Phase 11 archive edits",
        ],
    }
    M4_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    M4_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_m4_report(summary, selected)

    if warnings:
        raise RuntimeError("M4 slot metric validation failed: " + "; ".join(warnings))
    return all_rows, summary


def normalized_values_in_unit_interval(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for field in ("normalized_distance", "normalized_depth", "normalized_level"):
            value = row[field]
            if value == "":
                continue
            number = float(value)
            if number < -1e-9 or number > 1.0 + 1e-9:
                return False
    return True


def validate_slot_output(
    rows: list[dict[str, Any]],
    selected: list[dict[str, str]],
    vertical_levels: int,
    layout_summaries: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if len(rows) != sum(int_numeric(row, "storage_cells") * vertical_levels for row in selected):
        warnings.append("total slot count does not equal storage_cells * vertical_levels")
    if len(rows) != 19776:
        warnings.append(f"total slot count is {len(rows)}, expected prompt value 19776")
    if max(int_numeric(row, "rank") for row in selected) != 3:
        warnings.append("max selected archive rank is not 3")
    if sorted({int(row["level"]) for row in rows}) != list(range(vertical_levels)):
        warnings.append("slot levels are not exactly 0-7")
    if any(row["block_id"] == "" for row in rows):
        warnings.append("one or more slot rows are missing block_id")
    if any(row["slot_metric_status"] != "ok" for row in rows):
        warnings.append("one or more slot rows have slot_metric_status warning")
    if not normalized_values_in_unit_interval(rows):
        warnings.append("one or more normalized values fall outside [0,1]")
    for layout_summary in layout_summaries:
        if layout_summary["observed_slots"] != layout_summary["expected_slots"]:
            warnings.append(f"{layout_summary['selection_label']} slot count mismatch")
        if layout_summary["no_access_block_count"]:
            warnings.append(
                f"{layout_summary['selection_label']} has {layout_summary['no_access_block_count']} no-access blocks"
            )
    return warnings


def selected_metric_table(selected: list[dict[str, str]]) -> str:
    lines = [
        "| Label | Type | Seed | Rank | Signature | S | PF | Deep share | RP | MeanD | MaxD | Cap |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {label} | {stype} | {seed} | {rank} | `{sig}` | {s} | {pf} | {deep:.3f} | {rp:.0f} | {mean:.2f} | {maxd} | {cap} |".format(
                label=row["selection_label"],
                stype=row["selection_type"],
                seed=row["seed"],
                rank=row["rank"],
                sig=row["layout_signature_short"],
                s=row["storage_cells"],
                pf=row["pick_faces"],
                deep=float(row["interior_deep_storage_share"]),
                rp=float(row["retrieval_penalty"]),
                mean=float(row["mean_depth"]),
                maxd=row["max_depth"],
                cap=row["pallet_slot_capacity"],
            )
        )
    return "\n".join(lines)


def slot_summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Label | Storage cells | Expected slots | Observed slots | Blocks | No-access blocks | Warning slots |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["layout_summaries"]:
        lines.append(
            f"| {item['selection_label']} | {item['storage_cells']} | {item['expected_slots']} | "
            f"{item['observed_slots']} | {item['block_count']} | {item['no_access_block_count']} | "
            f"{item['warning_count']} |"
        )
    return "\n".join(lines)


def write_m3c_report(summary: dict[str, Any], selected: list[dict[str, str]]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    report = f"""# Operational layout selection final L1-L4 report

## Final selected layouts

{selected_metric_table(selected)}

## Panel

Final panel: `{summary['visual_panel_path']}`.

## Output files

- Selected layouts: `{summary['updated_selected_layouts_csv']}`
- Layout panel: `{summary['visual_panel_path']}`
- Summary JSON: `{summary['summary_json_path']}`
- Report: `{summary['markdown_report_path']}`

## Warnings

{warnings}
"""
    M3C_REPORT_MD.write_text(report, encoding="utf-8")


def write_m4_report(summary: dict[str, Any], selected: list[dict[str, str]]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    report = f"""# Operational-layer slot metrics

## Inputs

- Selected layouts: `{summary['input_selected_layouts_csv']}`
- Configuration: `{summary['input_operational_config_json']}`

## Final selected layouts

{selected_metric_table(selected)}

## Per-layout slot summary

{slot_summary_table(summary)}

## Validation summary

- Total slot rows: `{summary['slot_count_total']}`
- Expected total slot rows: `{summary['expected_slot_count_total']}`
- Prompt expected total: `{summary['expected_slot_count_prompt']}`
- Levels exactly 0-7: `{summary['validation']['levels_exact_0_to_7']}`
- All storage cells have block IDs: `{summary['validation']['all_storage_cells_have_block_id']}`
- Accessible slots have finite depth and distance: `{summary['validation']['all_accessible_slots_have_finite_depth_and_distance']}`
- Normalized values in [0,1]: `{summary['validation']['normalized_values_in_unit_interval']}`
- Ready for Milestone 5: `{summary['ready_for_milestone_5']}`

## Output files

- Slot metrics: `{summary['slot_metrics_csv']}`
- Summary JSON: `{summary['summary_json_path']}`
- Report: `{summary['markdown_report_path']}`

## Warnings

{warnings}
"""
    M4_REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    config = load_config()
    selected, lock_summary = lock_final_selection()
    slot_rows, slot_summary = extract_slot_metrics(selected, config)

    counts_by_rank: dict[str, dict[str, int]] = {}
    for row in selected:
        rank_key = str(row["rank"])
        counts_by_rank.setdefault(rank_key, {"layouts": 0, "slots": 0})
        counts_by_rank[rank_key]["layouts"] += 1
        counts_by_rank[rank_key]["slots"] += int(row["pallet_slot_capacity"])

    print("Milestone 4 final lock and slot metric extraction complete.")
    print(f"selected layout rows: {len(selected)}")
    print(f"final signatures: {', '.join(row['layout_signature'] for row in selected)}")
    print(f"slot rows total: {len(slot_rows)}")
    print(f"slot rows by layout: {json.dumps(slot_summary['validation']['slot_counts_by_layout'], sort_keys=True)}")
    print(f"expected slot rows by layout: {json.dumps(slot_summary['validation']['expected_slot_counts_by_layout'], sort_keys=True)}")
    print(f"counts by selected archive rank: {json.dumps(counts_by_rank, sort_keys=True)}")
    print(f"selected_layouts.csv: {rel_posix(SELECTED_CSV)}")
    print(f"final panel: {rel_posix(FINAL_PANEL_PNG)}")
    print(f"m3C summary JSON: {rel_posix(M3C_SUMMARY_JSON)}")
    print(f"m3C report: {rel_posix(M3C_REPORT_MD)}")
    print(f"slot metrics CSV: {rel_posix(SLOT_METRICS_CSV)}")
    print(f"m4 summary JSON: {rel_posix(M4_SUMMARY_JSON)}")
    print(f"m4 report: {rel_posix(M4_REPORT_MD)}")
    print(f"ready_for_milestone_5: {slot_summary['ready_for_milestone_5']}")
    all_warnings = lock_summary["warnings"] + slot_summary["warnings"]
    print("warnings or blockers: " + ("; ".join(all_warnings) if all_warnings else "none"))


if __name__ == "__main__":
    main()
