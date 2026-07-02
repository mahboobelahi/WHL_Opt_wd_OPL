"""Compute fixed-assignment operational diagnostic metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = PROJECT_ROOT / "data" / "operational_layer"
DATA_ROOT = OP_ROOT / "paper_inputs"
LOG_ROOT = OP_ROOT / "paper_outputs" / "logs"
DOC_ROOT = PROJECT_ROOT / "docs"

SELECTED_LAYOUTS_CSV = DATA_ROOT / "selected_layouts.csv"
SLOT_METRICS_CSV = DATA_ROOT / "slot_metrics_by_layout.csv"
SKU_CATALOG_CSV = DATA_ROOT / "sku_catalog.csv"
REP_CSV = DATA_ROOT / "representative_access_assignment.csv"
RESERVE_CSV = DATA_ROOT / "reserve_pallet_assignment.csv"
CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"
M6_SUMMARY_JSON = LOG_ROOT / "m6_assignment_summary.json"

REGIME_A_CSV = DATA_ROOT / "regime_A_metrics.csv"
REGIME_B_CSV = DATA_ROOT / "regime_B_metrics.csv"
FRAGMENTATION_CSV = DATA_ROOT / "reserve_fragmentation_summary.csv"
SUMMARY_JSON = LOG_ROOT / "m7_regime_metrics_summary.json"
REPORT_MD = DOC_ROOT / "102_operational_layer_regime_metrics.md"

LAYOUTS = ["L1", "L2", "L3", "L4"]
EXPECTED_CAPACITIES = {"L1": 4480, "L2": 5056, "L3": 5440, "L4": 4800}
REGIME_A_WEIGHTS = {"lambda_depth": 1.0, "lambda_level": 1.0}
REGIME_B_WEIGHTS = {"lambda_depth": 0.1, "lambda_level": 0.1}
VERTICAL_LEVELS = 8

REGIME_A_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "sku_count_total",
    "A_sku_count",
    "B_sku_count",
    "C_sku_count",
    "A_high_access_share",
    "AB_high_access_share",
    "demand_weight_sum",
    "A_demand_weight_sum",
    "B_demand_weight_sum",
    "C_demand_weight_sum",
    "demand_weighted_horizontal_distance",
    "demand_weighted_normalized_distance",
    "demand_weighted_effective_depth",
    "demand_weighted_normalized_depth",
    "demand_weighted_level",
    "demand_weighted_normalized_level",
    "lambda_depth",
    "lambda_level",
    "weighted_access_cost",
    "A_weighted_access_cost",
    "B_weighted_access_cost",
    "C_weighted_access_cost",
    "A_mean_horizontal_distance",
    "B_mean_horizontal_distance",
    "C_mean_horizontal_distance",
    "A_mean_effective_depth",
    "B_mean_effective_depth",
    "C_mean_effective_depth",
    "A_mean_level",
    "B_mean_level",
    "C_mean_level",
    "representative_rows",
    "representative_validation_status",
    "regime_A_metric_status",
    "regime_A_metric_warning",
]

REGIME_B_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "total_pallet_slot_capacity",
    "storage_cells_layout",
    "assigned_pallets_total",
    "assigned_inventory_utilization",
    "interior_deep_slot_capacity",
    "interior_deep_slot_share",
    "upper_level_slot_capacity",
    "upper_level_slot_share",
    "reserve_pallets_assigned",
    "reserve_deep_pallets",
    "reserve_deep_pallet_share",
    "reserve_upper_level_pallets",
    "reserve_upper_level_pallet_share",
    "largest_contiguous_block_size",
    "block_count",
    "reserve_blocks_used",
    "reserve_block_side_groups_used",
    "reserve_fragmentation_proxy",
    "mean_reserve_groups_per_sku",
    "mean_reserve_blocks_per_sku",
    "mean_reserve_slots_per_group",
    "same_block_reserve_share",
    "same_side_reserve_share",
    "lambda_depth",
    "lambda_level",
    "low_depth_level_weight_access_cost",
    "A_reserve_mean_depth",
    "B_reserve_mean_depth",
    "C_reserve_mean_depth",
    "A_reserve_mean_level",
    "B_reserve_mean_level",
    "C_reserve_mean_level",
    "regime_B_metric_status",
    "regime_B_metric_warning",
]

FRAGMENTATION_COLUMNS = [
    "selection_label",
    "selection_type",
    "sku_class",
    "sku_count",
    "reserve_pallets_assigned",
    "reserve_blocks_used",
    "reserve_block_side_groups_used",
    "mean_groups_per_sku",
    "mean_blocks_per_sku",
    "mean_slots_per_group",
    "same_block_reserve_share",
    "same_side_reserve_share",
    "reserve_deep_pallet_share",
    "reserve_upper_level_pallet_share",
    "reserve_fragmentation_proxy",
    "fragmentation_status",
    "fragmentation_warning",
]


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def rel_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.nan
    return float(value)


def to_int(row: dict[str, Any], field: str) -> int:
    return int(round(float(row[field])))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def share(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def true_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def reserve_group(row: dict[str, str]) -> str:
    return f"{row['selection_label']}__block{row['block_id']}__side{row['effective_access_side']}"


def high_access(row: dict[str, str]) -> bool:
    return (
        to_float(row, "effective_depth") <= 1.0
        and to_int(row, "level") <= 1
        and math.isfinite(to_float(row, "horizontal_access_distance"))
        and row.get("access_type") != "no_access"
    )


def load_inputs() -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    missing = [
        path
        for path in (
            SELECTED_LAYOUTS_CSV,
            SLOT_METRICS_CSV,
            SKU_CATALOG_CSV,
            REP_CSV,
            RESERVE_CSV,
            CONFIG_JSON,
            M6_SUMMARY_JSON,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required M7 input(s): " + ", ".join(as_posix(path) for path in missing))

    config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    selected = read_csv(SELECTED_LAYOUTS_CSV)
    slots = read_csv(SLOT_METRICS_CSV)
    skus = read_csv(SKU_CATALOG_CSV)
    reps = read_csv(REP_CSV)
    reserves = read_csv(RESERVE_CSV)
    m6_summary = json.loads(M6_SUMMARY_JSON.read_text(encoding="utf-8"))

    if [row.get("selection_label") for row in selected] != LAYOUTS:
        raise RuntimeError("selected_layouts.csv labels are not exactly L1-L4")
    if sorted({row.get("selection_label") for row in slots}) != LAYOUTS:
        raise RuntimeError("slot_metrics_by_layout.csv does not contain exact L1-L4 labels")
    if len(skus) != 100:
        raise RuntimeError(f"sku_catalog.csv has {len(skus)} rows, expected 100")
    if any(m6_summary.get("representative_rows_by_layout", {}).get(label) != 100 for label in LAYOUTS):
        raise RuntimeError("M6 summary does not report 100 representative rows per layout")
    if any(m6_summary.get("reserve_rows_by_layout", {}).get(label) != 690 for label in LAYOUTS):
        raise RuntimeError("M6 summary does not report 690 reserve rows per layout")

    for label in LAYOUTS:
        rep_count = sum(1 for row in reps if row["selection_label"] == label)
        reserve_count = sum(1 for row in reserves if row["selection_label"] == label)
        if rep_count != 100:
            raise RuntimeError(f"{label}: representative rows are {rep_count}, expected 100")
        if reserve_count != 690:
            raise RuntimeError(f"{label}: reserve rows are {reserve_count}, expected 690")
        if rep_count + reserve_count != 790:
            raise RuntimeError(f"{label}: assigned pallets are {rep_count + reserve_count}, expected 790")

    return config, selected, slots, skus, reps, reserves


def compute_regime_a(reps: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lambda_depth = REGIME_A_WEIGHTS["lambda_depth"]
    lambda_level = REGIME_A_WEIGHTS["lambda_level"]
    for label in LAYOUTS:
        layout_reps = [row for row in reps if row["selection_label"] == label]
        first = layout_reps[0]
        class_rows = {sku_class: [row for row in layout_reps if row["sku_class"] == sku_class] for sku_class in ("A", "B", "C")}
        weights = [to_float(row, "demand_weight") for row in layout_reps]
        demand_sum = sum(weights)
        class_demand = {
            sku_class: sum(to_float(row, "demand_weight") for row in class_rows[sku_class])
            for sku_class in ("A", "B", "C")
        }

        def weighted(field: str) -> float:
            return sum(to_float(row, "demand_weight") * to_float(row, field) for row in layout_reps)

        def access_cost(row: dict[str, str]) -> float:
            return (
                to_float(row, "normalized_distance")
                + lambda_depth * to_float(row, "normalized_depth")
                + lambda_level * to_float(row, "normalized_level")
            )

        class_cost = {
            sku_class: sum(to_float(row, "demand_weight") * access_cost(row) for row in class_rows[sku_class])
            for sku_class in ("A", "B", "C")
        }
        weighted_access_cost = sum(class_cost.values())
        a_high = sum(1 for row in class_rows["A"] if high_access(row))
        ab_rows = class_rows["A"] + class_rows["B"]
        ab_high = sum(1 for row in ab_rows if high_access(row))
        warning = ""
        status = "ok"
        if not math.isclose(demand_sum, 1.0, rel_tol=0, abs_tol=1e-9):
            status = "warning"
            warning = f"demand weight sum is {demand_sum:.12g}"

        row = {
            "selection_label": label,
            "selection_type": first["selection_type"],
            "layout_signature": first["layout_signature"],
            "seed": first["seed"],
            "rank": first["rank"],
            "sku_count_total": len(layout_reps),
            "A_sku_count": len(class_rows["A"]),
            "B_sku_count": len(class_rows["B"]),
            "C_sku_count": len(class_rows["C"]),
            "A_high_access_share": share(a_high, len(class_rows["A"])),
            "AB_high_access_share": share(ab_high, len(ab_rows)),
            "demand_weight_sum": demand_sum,
            "A_demand_weight_sum": class_demand["A"],
            "B_demand_weight_sum": class_demand["B"],
            "C_demand_weight_sum": class_demand["C"],
            "demand_weighted_horizontal_distance": weighted("horizontal_access_distance"),
            "demand_weighted_normalized_distance": weighted("normalized_distance"),
            "demand_weighted_effective_depth": weighted("effective_depth"),
            "demand_weighted_normalized_depth": weighted("normalized_depth"),
            "demand_weighted_level": weighted("level"),
            "demand_weighted_normalized_level": weighted("normalized_level"),
            "lambda_depth": lambda_depth,
            "lambda_level": lambda_level,
            "weighted_access_cost": weighted_access_cost,
            "A_weighted_access_cost": class_cost["A"],
            "B_weighted_access_cost": class_cost["B"],
            "C_weighted_access_cost": class_cost["C"],
            "A_mean_horizontal_distance": mean([to_float(row, "horizontal_access_distance") for row in class_rows["A"]]),
            "B_mean_horizontal_distance": mean([to_float(row, "horizontal_access_distance") for row in class_rows["B"]]),
            "C_mean_horizontal_distance": mean([to_float(row, "horizontal_access_distance") for row in class_rows["C"]]),
            "A_mean_effective_depth": mean([to_float(row, "effective_depth") for row in class_rows["A"]]),
            "B_mean_effective_depth": mean([to_float(row, "effective_depth") for row in class_rows["B"]]),
            "C_mean_effective_depth": mean([to_float(row, "effective_depth") for row in class_rows["C"]]),
            "A_mean_level": mean([to_float(row, "level") for row in class_rows["A"]]),
            "B_mean_level": mean([to_float(row, "level") for row in class_rows["B"]]),
            "C_mean_level": mean([to_float(row, "level") for row in class_rows["C"]]),
            "representative_rows": len(layout_reps),
            "representative_validation_status": "passed" if len(layout_reps) == 100 else "failed",
            "regime_A_metric_status": status,
            "regime_A_metric_warning": warning,
        }
        rows.append({key: fmt(value) for key, value in row.items()})
    return rows


def compute_fragmentation_rows(reserves: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in LAYOUTS:
        for sku_class in ("A", "B", "C"):
            class_reserves = [
                row for row in reserves if row["selection_label"] == label and row["sku_class"] == sku_class
            ]
            sku_ids = sorted({row["sku_id"] for row in class_reserves})
            groups_by_sku: dict[str, set[str]] = defaultdict(set)
            blocks_by_sku: dict[str, set[str]] = defaultdict(set)
            for row in class_reserves:
                groups_by_sku[row["sku_id"]].add(reserve_group(row))
                blocks_by_sku[row["sku_id"]].add(row["block_id"])
            group_count = len({reserve_group(row) for row in class_reserves})
            block_count = len({row["block_id"] for row in class_reserves})
            reserve_count = len(class_reserves)
            same_block = sum(1 for row in class_reserves if true_value(row["same_block_as_representative"]))
            same_side = sum(1 for row in class_reserves if true_value(row["same_side_as_representative"]))
            deep = sum(1 for row in class_reserves if to_float(row, "effective_depth") >= 3.0)
            upper = sum(1 for row in class_reserves if to_int(row, "level") >= 3)
            mean_groups = mean([len(groups_by_sku[sku_id]) for sku_id in sku_ids])
            mean_blocks = mean([len(blocks_by_sku[sku_id]) for sku_id in sku_ids])
            row = {
                "selection_label": label,
                "selection_type": class_reserves[0]["selection_type"] if class_reserves else "",
                "sku_class": sku_class,
                "sku_count": len(sku_ids),
                "reserve_pallets_assigned": reserve_count,
                "reserve_blocks_used": block_count,
                "reserve_block_side_groups_used": group_count,
                "mean_groups_per_sku": mean_groups,
                "mean_blocks_per_sku": mean_blocks,
                "mean_slots_per_group": share(reserve_count, group_count),
                "same_block_reserve_share": share(same_block, reserve_count),
                "same_side_reserve_share": share(same_side, reserve_count),
                "reserve_deep_pallet_share": share(deep, reserve_count),
                "reserve_upper_level_pallet_share": share(upper, reserve_count),
                "reserve_fragmentation_proxy": mean_groups,
                "fragmentation_status": "ok",
                "fragmentation_warning": "",
            }
            rows.append({key: fmt(value) for key, value in row.items()})
    return rows


def compute_regime_b(
    selected: list[dict[str, str]],
    slots: list[dict[str, str]],
    reps: list[dict[str, str]],
    reserves: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_by_label = {row["selection_label"]: row for row in selected}
    lambda_depth = REGIME_B_WEIGHTS["lambda_depth"]
    lambda_level = REGIME_B_WEIGHTS["lambda_level"]
    for label in LAYOUTS:
        layout_slots = [row for row in slots if row["selection_label"] == label]
        layout_reps = [row for row in reps if row["selection_label"] == label]
        layout_reserves = [row for row in reserves if row["selection_label"] == label]
        first = layout_slots[0]
        total_capacity = len(layout_slots)
        assigned_total = len(layout_reps) + len(layout_reserves)
        deep_capacity = sum(1 for row in layout_slots if to_float(row, "effective_depth") >= 3.0)
        upper_capacity = sum(1 for row in layout_slots if to_int(row, "level") >= 3)
        reserve_deep = sum(1 for row in layout_reserves if to_float(row, "effective_depth") >= 3.0)
        reserve_upper = sum(1 for row in layout_reserves if to_int(row, "level") >= 3)
        block_count = len({row["block_id"] for row in layout_slots})
        reserve_blocks = len({row["block_id"] for row in layout_reserves})
        reserve_groups = len({reserve_group(row) for row in layout_reserves})
        groups_by_sku: dict[str, set[str]] = defaultdict(set)
        blocks_by_sku: dict[str, set[str]] = defaultdict(set)
        for row in layout_reserves:
            groups_by_sku[row["sku_id"]].add(reserve_group(row))
            blocks_by_sku[row["sku_id"]].add(row["block_id"])
        sku_ids = sorted({row["sku_id"] for row in layout_reserves})
        mean_groups_per_sku = mean([len(groups_by_sku[sku_id]) for sku_id in sku_ids])
        mean_blocks_per_sku = mean([len(blocks_by_sku[sku_id]) for sku_id in sku_ids])
        same_block = sum(1 for row in layout_reserves if true_value(row["same_block_as_representative"]))
        same_side = sum(1 for row in layout_reserves if true_value(row["same_side_as_representative"]))

        def reserve_cost(row: dict[str, str]) -> float:
            return (
                to_float(row, "normalized_distance")
                + lambda_depth * to_float(row, "normalized_depth")
                + lambda_level * to_float(row, "normalized_level")
            )

        reserve_by_class = {
            sku_class: [row for row in layout_reserves if row["sku_class"] == sku_class]
            for sku_class in ("A", "B", "C")
        }
        selected_row = selected_by_label[label]
        largest_block = (
            to_int(selected_row, "largest_block_size")
            if selected_row.get("largest_block_size", "")
            else max(to_int(row, "block_size") for row in layout_slots)
        )
        warning = ""
        status = "ok"
        if total_capacity != EXPECTED_CAPACITIES[label]:
            status = "warning"
            warning = f"capacity {total_capacity} != expected {EXPECTED_CAPACITIES[label]}"

        row = {
            "selection_label": label,
            "selection_type": first["selection_type"],
            "layout_signature": first["layout_signature"],
            "seed": first["seed"],
            "rank": first["rank"],
            "total_pallet_slot_capacity": total_capacity,
            "storage_cells_layout": total_capacity / VERTICAL_LEVELS,
            "assigned_pallets_total": assigned_total,
            "assigned_inventory_utilization": share(assigned_total, total_capacity),
            "interior_deep_slot_capacity": deep_capacity,
            "interior_deep_slot_share": share(deep_capacity, total_capacity),
            "upper_level_slot_capacity": upper_capacity,
            "upper_level_slot_share": share(upper_capacity, total_capacity),
            "reserve_pallets_assigned": len(layout_reserves),
            "reserve_deep_pallets": reserve_deep,
            "reserve_deep_pallet_share": share(reserve_deep, len(layout_reserves)),
            "reserve_upper_level_pallets": reserve_upper,
            "reserve_upper_level_pallet_share": share(reserve_upper, len(layout_reserves)),
            "largest_contiguous_block_size": largest_block,
            "block_count": block_count,
            "reserve_blocks_used": reserve_blocks,
            "reserve_block_side_groups_used": reserve_groups,
            "reserve_fragmentation_proxy": mean_groups_per_sku,
            "mean_reserve_groups_per_sku": mean_groups_per_sku,
            "mean_reserve_blocks_per_sku": mean_blocks_per_sku,
            "mean_reserve_slots_per_group": share(len(layout_reserves), reserve_groups),
            "same_block_reserve_share": share(same_block, len(layout_reserves)),
            "same_side_reserve_share": share(same_side, len(layout_reserves)),
            "lambda_depth": lambda_depth,
            "lambda_level": lambda_level,
            "low_depth_level_weight_access_cost": mean([reserve_cost(row) for row in layout_reserves]),
            "A_reserve_mean_depth": mean([to_float(row, "effective_depth") for row in reserve_by_class["A"]]),
            "B_reserve_mean_depth": mean([to_float(row, "effective_depth") for row in reserve_by_class["B"]]),
            "C_reserve_mean_depth": mean([to_float(row, "effective_depth") for row in reserve_by_class["C"]]),
            "A_reserve_mean_level": mean([to_float(row, "level") for row in reserve_by_class["A"]]),
            "B_reserve_mean_level": mean([to_float(row, "level") for row in reserve_by_class["B"]]),
            "C_reserve_mean_level": mean([to_float(row, "level") for row in reserve_by_class["C"]]),
            "regime_B_metric_status": status,
            "regime_B_metric_warning": warning,
        }
        rows.append({key: fmt(value) for key, value in row.items()})
    return rows


def validate_outputs(
    regime_a: list[dict[str, Any]],
    regime_b: list[dict[str, Any]],
    fragmentation: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    def labels(rows: list[dict[str, Any]]) -> list[str]:
        return [row["selection_label"] for row in rows]

    validation: dict[str, Any] = {
        "regime_A_row_count": len(regime_a),
        "regime_A_labels_exact": labels(regime_a) == LAYOUTS,
        "regime_B_row_count": len(regime_b),
        "regime_B_labels_exact": labels(regime_b) == LAYOUTS,
        "fragmentation_row_count": len(fragmentation),
        "fragmentation_expected_rows": 12,
        "no_order_fields_created": not any("order" in column.lower() for column in REGIME_A_COLUMNS + REGIME_B_COLUMNS + FRAGMENTATION_COLUMNS),
    }
    if len(regime_a) != 4:
        warnings.append("regime_A_metrics.csv would not have exactly 4 rows")
    if labels(regime_a) != LAYOUTS:
        warnings.append("Regime A labels are not exactly L1-L4")
    if len(regime_b) != 4:
        warnings.append("regime_B_metrics.csv would not have exactly 4 rows")
    if labels(regime_b) != LAYOUTS:
        warnings.append("Regime B labels are not exactly L1-L4")
    if len(fragmentation) != 12:
        warnings.append("reserve_fragmentation_summary.csv would not have 12 rows")
    for row in regime_a:
        label = row["selection_label"]
        checks = [
            ("representative_rows", float(row["representative_rows"]) == 100),
            ("demand_weight_sum", math.isclose(float(row["demand_weight_sum"]), 1.0, rel_tol=0, abs_tol=1e-9)),
            ("A_demand_weight_sum", math.isclose(float(row["A_demand_weight_sum"]), 0.80, rel_tol=0, abs_tol=1e-9)),
            ("B_demand_weight_sum", math.isclose(float(row["B_demand_weight_sum"]), 0.15, rel_tol=0, abs_tol=1e-9)),
            ("C_demand_weight_sum", math.isclose(float(row["C_demand_weight_sum"]), 0.05, rel_tol=0, abs_tol=1e-9)),
            ("A_high_access_share", 0.0 <= float(row["A_high_access_share"]) <= 1.0),
            ("AB_high_access_share", 0.0 <= float(row["AB_high_access_share"]) <= 1.0),
            ("weighted_access_cost", finite_number(row["weighted_access_cost"])),
        ]
        for field, passed in checks:
            validation[f"regime_A_{label}_{field}"] = passed
            if not passed:
                warnings.append(f"Regime A {label} validation failed: {field}")
    for row in regime_b:
        label = row["selection_label"]
        checks = [
            ("total_pallet_slot_capacity", int(float(row["total_pallet_slot_capacity"])) == EXPECTED_CAPACITIES[label]),
            ("assigned_pallets_total", int(float(row["assigned_pallets_total"])) == 790),
            ("reserve_pallets_assigned", int(float(row["reserve_pallets_assigned"])) == 690),
            ("reserve_fragmentation_proxy", finite_number(row["reserve_fragmentation_proxy"])),
            ("low_depth_level_weight_access_cost", finite_number(row["low_depth_level_weight_access_cost"])),
        ]
        share_fields = [
            "assigned_inventory_utilization",
            "interior_deep_slot_share",
            "upper_level_slot_share",
            "reserve_deep_pallet_share",
            "reserve_upper_level_pallet_share",
            "same_block_reserve_share",
            "same_side_reserve_share",
        ]
        checks.extend((field, 0.0 <= float(row[field]) <= 1.0) for field in share_fields)
        for field, passed in checks:
            validation[f"regime_B_{label}_{field}"] = passed
            if not passed:
                warnings.append(f"Regime B {label} validation failed: {field}")
    for row in fragmentation:
        label = row["selection_label"]
        sku_class = row["sku_class"]
        if not finite_number(row["reserve_fragmentation_proxy"]):
            warnings.append(f"Fragmentation {label}/{sku_class} proxy is non-finite")
        for field in (
            "same_block_reserve_share",
            "same_side_reserve_share",
            "reserve_deep_pallet_share",
            "reserve_upper_level_pallet_share",
        ):
            value = float(row[field])
            if value < 0.0 or value > 1.0:
                warnings.append(f"Fragmentation {label}/{sku_class} {field} outside [0,1]")
    return validation, warnings


def ranking(rows: list[dict[str, Any]], metric: str, ascending: bool = True) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=not ascending)
    return [
        {
            "rank": index + 1,
            "selection_label": row["selection_label"],
            metric: float(row[metric]),
        }
        for index, row in enumerate(ordered)
    ]


def conceptual_check(regime_a: list[dict[str, Any]], regime_b: list[dict[str, Any]]) -> dict[str, Any]:
    a_rank = ranking(regime_a, "weighted_access_cost", ascending=True)
    deep_rank = ranking(regime_b, "interior_deep_slot_share", ascending=False)
    frag_rank = ranking(regime_b, "reserve_fragmentation_proxy", ascending=True)
    l1_a_position = next(item["rank"] for item in a_rank if item["selection_label"] == "L1")
    l2_deep_position = next(item["rank"] for item in deep_rank if item["selection_label"] == "L2")
    l3_deep_position = next(item["rank"] for item in deep_rank if item["selection_label"] == "L3")
    l4_a_cost = next(float(row["weighted_access_cost"]) for row in regime_a if row["selection_label"] == "L4")
    costs = {row["selection_label"]: float(row["weighted_access_cost"]) for row in regime_a}
    return {
        "regime_A_best_by_weighted_access_cost": a_rank[0]["selection_label"],
        "regime_A_L1_position": l1_a_position,
        "regime_A_L1_strong": l1_a_position <= 2,
        "regime_B_top_deep_capacity_layout": deep_rank[0]["selection_label"],
        "regime_B_L2_or_L3_top_deep_capacity": min(l2_deep_position, l3_deep_position) == 1,
        "regime_B_best_fragmentation_layout": frag_rank[0]["selection_label"],
        "L4_access_cost_between_L1_and_L2": min(costs["L1"], costs["L2"]) <= l4_a_cost <= max(costs["L1"], costs["L2"]),
        "interpretation": (
            "Conceptual checks are descriptive only; the diagnostic reports actual deterministic-assignment results "
            "without forcing them to match expected direction."
        ),
    }


def update_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["regime_metrics"] = {
        "regime_A": {
            "lambda_depth": REGIME_A_WEIGHTS["lambda_depth"],
            "lambda_level": REGIME_A_WEIGHTS["lambda_level"],
            "high_access_definition": {
                "effective_depth_max": 1,
                "level_max": 1,
                "requires_finite_horizontal_distance": True,
                "excludes_no_access": True,
            },
        },
        "regime_B": {
            "lambda_depth": REGIME_B_WEIGHTS["lambda_depth"],
            "lambda_level": REGIME_B_WEIGHTS["lambda_level"],
            "deep_threshold_effective_depth": 3,
            "upper_level_threshold": 3,
            "reserve_fragmentation_proxy": "mean_reserve_block_side_groups_per_sku",
        },
        "outputs": {
            "regime_A_metrics_csv": as_posix(REGIME_A_CSV),
            "regime_B_metrics_csv": as_posix(REGIME_B_CSV),
            "reserve_fragmentation_summary_csv": as_posix(FRAGMENTATION_CSV),
            "m7_summary_json": as_posix(SUMMARY_JSON),
        },
    }
    completed = list(updated.get("milestones_completed", []))
    if "M7" not in completed:
        completed.append("M7")
    updated["milestones_completed"] = completed
    return updated


def metric_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def ranking_lines(items: list[dict[str, Any]], metric: str) -> str:
    return "\n".join(
        f"{item['rank']}. {item['selection_label']} ({metric}={item[metric]:.6f})"
        for item in items
    )


def write_report(
    summary: dict[str, Any],
    regime_a: list[dict[str, Any]],
    regime_b: list[dict[str, Any]],
    fragmentation: list[dict[str, Any]],
) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."

    report = f"""# Operational-layer diagnostic metrics

## Inputs

- Representative assignments: `{summary['input_representative_assignment_csv']}`
- Reserve assignments: `{summary['input_reserve_assignment_csv']}`
- Slot metrics: `{summary['input_slot_metrics_csv']}`
- SKU catalog: `{as_posix(SKU_CATALOG_CSV)}`
- Configuration: `{as_posix(CONFIG_JSON)}`

## Validation summary

- Scenario A rows: `{summary['regime_A_rows']}`
- Scenario B rows: `{summary['regime_B_rows']}`
- Fragmentation rows: `{summary['fragmentation_rows']}`
- Ready for next step: `{summary['ready_for_milestone_8']}`

## Scenario A results

{metric_table(regime_a, ['selection_label', 'A_high_access_share', 'AB_high_access_share', 'weighted_access_cost', 'demand_weighted_horizontal_distance', 'demand_weighted_effective_depth', 'demand_weighted_level'])}

Ranking by weighted access cost:

{ranking_lines(summary['ranked_layouts_regime_A_by_weighted_access_cost'], 'weighted_access_cost')}

## Scenario B results

{metric_table(regime_b, ['selection_label', 'interior_deep_slot_share', 'reserve_fragmentation_proxy', 'low_depth_level_weight_access_cost', 'assigned_inventory_utilization', 'reserve_deep_pallet_share', 'reserve_upper_level_pallet_share'])}

Ranking by interior/deep slot share:

{ranking_lines(summary['ranked_layouts_regime_B_by_interior_deep_slot_share'], 'interior_deep_slot_share')}

Ranking by reserve fragmentation proxy:

{ranking_lines(summary['ranked_layouts_regime_B_by_reserve_fragmentation_proxy'], 'reserve_fragmentation_proxy')}

## Reserve fragmentation summary

{metric_table(fragmentation, ['selection_label', 'sku_class', 'reserve_pallets_assigned', 'reserve_block_side_groups_used', 'mean_groups_per_sku', 'same_block_reserve_share', 'same_side_reserve_share', 'reserve_fragmentation_proxy'])}

## Conceptual direction check

`{json.dumps(summary['conceptual_direction_check'], sort_keys=True)}`

## Output files

- Scenario A metrics: `{summary['regime_A_metrics_csv']}`
- Scenario B metrics: `{summary['regime_B_metrics_csv']}`
- Reserve fragmentation summary: `{summary['reserve_fragmentation_summary_csv']}`
- Summary JSON: `{as_posix(SUMMARY_JSON)}`
- Report: `{as_posix(REPORT_MD)}`

## Warnings

{warnings}
"""
    REPORT_MD.write_text(report, encoding="utf-8")
def main() -> None:
    config, selected, slots, _skus, reps, reserves = load_inputs()
    regime_a = compute_regime_a(reps)
    regime_b = compute_regime_b(selected, slots, reps, reserves)
    fragmentation = compute_fragmentation_rows(reserves)
    validation, warnings = validate_outputs(regime_a, regime_b, fragmentation)
    if warnings:
        raise RuntimeError("M7 validation failed before writing outputs: " + "; ".join(warnings))

    write_csv(REGIME_A_CSV, regime_a, REGIME_A_COLUMNS)
    write_csv(REGIME_B_CSV, regime_b, REGIME_B_COLUMNS)
    write_csv(FRAGMENTATION_CSV, fragmentation, FRAGMENTATION_COLUMNS)
    updated_config = update_config(config)
    CONFIG_JSON.write_text(json.dumps(updated_config, indent=2) + "\n", encoding="utf-8")

    validation, warnings = validate_outputs(regime_a, regime_b, fragmentation)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_representative_assignment_csv": as_posix(REP_CSV),
        "input_reserve_assignment_csv": as_posix(RESERVE_CSV),
        "input_slot_metrics_csv": as_posix(SLOT_METRICS_CSV),
        "regime_A_metrics_csv": as_posix(REGIME_A_CSV),
        "regime_B_metrics_csv": as_posix(REGIME_B_CSV),
        "reserve_fragmentation_summary_csv": as_posix(FRAGMENTATION_CSV),
        "regime_A_weights": REGIME_A_WEIGHTS,
        "regime_B_weights": REGIME_B_WEIGHTS,
        "regime_A_rows": len(regime_a),
        "regime_B_rows": len(regime_b),
        "fragmentation_rows": len(fragmentation),
        "validation": validation,
        "ranked_layouts_regime_A_by_weighted_access_cost": ranking(regime_a, "weighted_access_cost", ascending=True),
        "ranked_layouts_regime_B_by_reserve_fragmentation_proxy": ranking(regime_b, "reserve_fragmentation_proxy", ascending=True),
        "ranked_layouts_regime_B_by_interior_deep_slot_share": ranking(regime_b, "interior_deep_slot_share", ascending=False),
        "ranked_layouts_regime_B_by_low_depth_level_weight_access_cost": ranking(regime_b, "low_depth_level_weight_access_cost", ascending=True),
        "conceptual_direction_check": conceptual_check(regime_a, regime_b),
        "warnings": warnings,
        "blockers_or_warnings": warnings,
        "ready_for_milestone_8": not warnings,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, regime_a, regime_b, fragmentation)
    if warnings:
        raise RuntimeError("M7 validation failed after writing outputs: " + "; ".join(warnings))

    print("Milestone 7 Regime A/B metrics complete.")
    print(f"regime_A_metrics.csv: {rel_posix(REGIME_A_CSV)}")
    print(f"regime_B_metrics.csv: {rel_posix(REGIME_B_CSV)}")
    print(f"reserve_fragmentation_summary.csv: {rel_posix(FRAGMENTATION_CSV)}")
    print(f"summary JSON: {rel_posix(SUMMARY_JSON)}")
    print(f"Markdown report: {rel_posix(REPORT_MD)}")
    print("Regime A ranking by weighted_access_cost:")
    print(ranking_lines(summary["ranked_layouts_regime_A_by_weighted_access_cost"], "weighted_access_cost"))
    print("Regime B ranking by interior_deep_slot_share:")
    print(ranking_lines(summary["ranked_layouts_regime_B_by_interior_deep_slot_share"], "interior_deep_slot_share"))
    print("Regime B ranking by reserve_fragmentation_proxy:")
    print(ranking_lines(summary["ranked_layouts_regime_B_by_reserve_fragmentation_proxy"], "reserve_fragmentation_proxy"))
    print(f"conceptual direction check: {json.dumps(summary['conceptual_direction_check'], sort_keys=True)}")
    print("warnings or blockers: none")
    print(f"ready_for_milestone_8: {summary['ready_for_milestone_8']}")


if __name__ == "__main__":
    main()
