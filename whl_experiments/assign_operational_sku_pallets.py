"""Assign representative and reserve SKU pallets to fixed operational slots."""

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
CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"
M4_SUMMARY_JSON = LOG_ROOT / "m4_slot_metrics_summary.json"
M5_SUMMARY_JSON = LOG_ROOT / "m5_sku_catalog_summary.json"

REP_CSV = DATA_ROOT / "representative_access_assignment.csv"
RESERVE_CSV = DATA_ROOT / "reserve_pallet_assignment.csv"
SUMMARY_JSON = LOG_ROOT / "m6_assignment_summary.json"
REPORT_MD = DOC_ROOT / "101_operational_layer_representative_reserve_assignment.md"

EXPECTED_LAYOUTS = ["L1", "L2", "L3", "L4"]
EXPECTED_SKUS = 100
EXPECTED_PALLETS_PER_LAYOUT = 790
EXPECTED_REP_PER_LAYOUT = 100
EXPECTED_RESERVE_PER_LAYOUT = 690

REP_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "sku_id",
    "sku_class",
    "class_index",
    "global_sku_index",
    "demand_weight",
    "pallets_per_sku",
    "representative_access_pallets",
    "reserve_pallets",
    "representative_slot_id",
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
    "representative_assignment_rule",
    "representative_level_preference_met",
    "representative_depth_preference_met",
    "representative_assignment_status",
    "representative_assignment_warning",
]

RESERVE_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "sku_id",
    "sku_class",
    "class_index",
    "global_sku_index",
    "demand_weight",
    "pallets_per_sku",
    "reserve_pallets",
    "reserve_pallet_index",
    "reserve_slot_id",
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
    "representative_slot_id",
    "representative_block_id",
    "representative_effective_access_side",
    "same_block_as_representative",
    "same_side_as_representative",
    "reserve_group_id",
    "reserve_assignment_rule",
    "reserve_level_preference_met",
    "reserve_depth_preference_met",
    "reserve_assignment_status",
    "reserve_assignment_warning",
]

ACCESS_PRIORITY = {"two_sided": 0, "one_sided": 1, "no_access": 2}


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


def to_int(row: dict[str, Any], field: str) -> int:
    return int(round(float(row[field])))


def to_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.inf
    return float(value)


def slot_id(slot: dict[str, Any]) -> str:
    return f"{slot['selection_label']}__r{to_int(slot, 'row')}__c{to_int(slot, 'col')}__l{to_int(slot, 'level')}"


def reserve_group_id(slot: dict[str, Any]) -> str:
    return (
        f"{slot['selection_label']}__block{slot['block_id']}__side{slot['effective_access_side']}"
        f"__depth{slot['effective_depth']}__level{slot['level']}"
    )


def accessible(slot: dict[str, Any]) -> bool:
    return (
        slot.get("access_type") != "no_access"
        and slot.get("effective_depth", "") != ""
        and slot.get("horizontal_access_distance", "") != ""
        and math.isfinite(to_float(slot, "effective_depth"))
        and math.isfinite(to_float(slot, "horizontal_access_distance"))
    )


def load_inputs() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    missing = [
        path
        for path in (
            SELECTED_LAYOUTS_CSV,
            SLOT_METRICS_CSV,
            SKU_CATALOG_CSV,
            CONFIG_JSON,
            M4_SUMMARY_JSON,
            M5_SUMMARY_JSON,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required M6 input(s): " + ", ".join(as_posix(path) for path in missing))

    config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    selected = read_csv(SELECTED_LAYOUTS_CSV)
    slots = read_csv(SLOT_METRICS_CSV)
    skus = read_csv(SKU_CATALOG_CSV)

    if [row.get("selection_label") for row in selected] != EXPECTED_LAYOUTS:
        raise RuntimeError("selected_layouts.csv does not contain exact L1-L4 labels")
    slot_labels = sorted({row.get("selection_label") for row in slots})
    if slot_labels != EXPECTED_LAYOUTS:
        raise RuntimeError(f"slot_metrics_by_layout.csv labels are {slot_labels}, expected {EXPECTED_LAYOUTS}")
    if len(skus) != EXPECTED_SKUS:
        raise RuntimeError(f"SKU count is {len(skus)}, expected {EXPECTED_SKUS}")
    total_pallets = sum(to_int(row, "pallets_per_sku") for row in skus)
    if total_pallets != EXPECTED_PALLETS_PER_LAYOUT:
        raise RuntimeError(f"total SKU pallets are {total_pallets}, expected {EXPECTED_PALLETS_PER_LAYOUT}")
    if config.get("vertical_levels") != 8:
        raise RuntimeError("operational_config.json vertical_levels is missing or not 8")

    skus.sort(key=lambda row: to_int(row, "global_sku_index"))
    return config, selected, slots, skus


def rep_level_preferred(sku_class: str, level: int) -> bool:
    if sku_class == "A":
        return level in {0, 1}
    if sku_class == "B":
        return level in {0, 1, 2}
    return True


def reserve_level_preferred(sku_class: str, level: int) -> bool:
    if sku_class == "A":
        return level in {0, 1, 2}
    if sku_class == "B":
        return level in {0, 1, 2, 3}
    return level in {3, 4, 5, 6, 7}


def rep_depth_preferred(sku_class: str, depth: int) -> bool:
    if sku_class == "A":
        return depth == 1
    if sku_class == "B":
        return depth <= 2
    return True


def reserve_depth_preferred(sku_class: str, depth: int) -> bool:
    if sku_class == "A":
        return depth <= 2
    if sku_class == "B":
        return depth <= 3
    return True


def rep_sort_key(sku: dict[str, str], slot: dict[str, str]) -> tuple[Any, ...]:
    sku_class = sku["sku_class"]
    level = to_int(slot, "level")
    depth = to_float(slot, "effective_depth")
    cost = to_float(slot, "slot_cost")
    distance = to_float(slot, "horizontal_access_distance")
    norm_level = to_float(slot, "normalized_level")
    access_priority = ACCESS_PRIORITY.get(slot.get("access_type", "no_access"), 2)
    base = (
        to_int(slot, "row"),
        to_int(slot, "col"),
        to_int(slot, "level"),
    )
    if sku_class == "A":
        return (
            0 if level in {0, 1} else 1,
            depth,
            cost,
            distance,
            norm_level,
            access_priority,
            -to_int(slot, "block_size"),
            *base,
        )
    if sku_class == "B":
        return (
            0 if level in {0, 1, 2} else 1,
            cost,
            depth,
            distance,
            norm_level,
            access_priority,
            -to_int(slot, "block_size"),
            *base,
        )
    return (
        -to_float(slot, "normalized_depth"),
        -to_float(slot, "normalized_level"),
        -to_float(slot, "normalized_distance"),
        -cost,
        -to_int(slot, "block_size"),
        *base,
    )


def reserve_sort_key(sku: dict[str, str], slot: dict[str, str], rep: dict[str, Any]) -> tuple[Any, ...]:
    sku_class = sku["sku_class"]
    level = to_int(slot, "level")
    depth = to_float(slot, "effective_depth")
    cost = to_float(slot, "slot_cost")
    distance = to_float(slot, "horizontal_access_distance")
    norm_level = to_float(slot, "normalized_level")
    same_block = str(slot["block_id"]) == str(rep["block_id"])
    same_side = str(slot["effective_access_side"]) == str(rep["effective_access_side"])
    base = (
        to_int(slot, "row"),
        to_int(slot, "col"),
        to_int(slot, "level"),
    )
    if sku_class == "A":
        return (
            0 if same_block else 1,
            0 if same_side else 1,
            0 if level in {0, 1, 2} else 1,
            depth,
            cost,
            distance,
            norm_level,
            *base,
        )
    if sku_class == "B":
        return (
            0 if same_block else 1,
            0 if same_side else 1,
            0 if level in {0, 1, 2, 3} else 1,
            cost,
            depth,
            distance,
            norm_level,
            *base,
        )
    return (
        -to_int(slot, "block_size"),
        -to_float(slot, "normalized_depth"),
        -to_float(slot, "normalized_level"),
        -to_float(slot, "normalized_distance"),
        reserve_group_id(slot),
        *base,
    )


def copy_slot_fields(slot: dict[str, str]) -> dict[str, Any]:
    return {
        "row": slot["row"],
        "col": slot["col"],
        "level": slot["level"],
        "block_id": slot["block_id"],
        "block_size": slot["block_size"],
        "access_type": slot["access_type"],
        "access_side_count": slot["access_side_count"],
        "access_sides": slot["access_sides"],
        "effective_access_side": slot["effective_access_side"],
        "effective_pick_face_row": slot["effective_pick_face_row"],
        "effective_pick_face_col": slot["effective_pick_face_col"],
        "effective_depth": slot["effective_depth"],
        "horizontal_access_distance": slot["horizontal_access_distance"],
        "vertical_level": slot["vertical_level"],
        "normalized_distance": slot["normalized_distance"],
        "normalized_depth": slot["normalized_depth"],
        "normalized_level": slot["normalized_level"],
        "slot_cost": slot["slot_cost"],
    }


def layout_fields(slot: dict[str, str]) -> dict[str, Any]:
    return {
        "selection_label": slot["selection_label"],
        "selection_type": slot["selection_type"],
        "layout_signature": slot["layout_signature"],
        "seed": slot["seed"],
        "rank": slot["rank"],
    }


def sku_fields(sku: dict[str, str]) -> dict[str, Any]:
    return {
        "sku_id": sku["sku_id"],
        "sku_class": sku["sku_class"],
        "class_index": sku["class_index"],
        "global_sku_index": sku["global_sku_index"],
        "demand_weight": sku["demand_weight"],
        "pallets_per_sku": sku["pallets_per_sku"],
        "representative_access_pallets": sku["representative_access_pallets"],
        "reserve_pallets": sku["reserve_pallets"],
    }


def assign_layout(
    label: str,
    layout_slots: list[dict[str, str]],
    skus: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available = {slot_id(slot): slot for slot in layout_slots if accessible(slot)}
    representative_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    representatives_by_sku: dict[str, dict[str, Any]] = {}

    for sku in skus:
        if not available:
            raise RuntimeError(f"{label}: no accessible slots available for representative assignment")
        chosen = min(available.values(), key=lambda slot: rep_sort_key(sku, slot))
        chosen_slot_id = slot_id(chosen)
        del available[chosen_slot_id]
        level_met = rep_level_preferred(sku["sku_class"], to_int(chosen, "level"))
        depth_met = rep_depth_preferred(sku["sku_class"], to_int(chosen, "effective_depth"))
        warning = "" if level_met else "preferred representative level exhausted; overflow slot used"
        row = {
            **layout_fields(chosen),
            **sku_fields(sku),
            "representative_slot_id": chosen_slot_id,
            **copy_slot_fields(chosen),
            "representative_assignment_rule": f"{sku['sku_class']}_representative_deterministic_sort",
            "representative_level_preference_met": str(level_met).lower(),
            "representative_depth_preference_met": str(depth_met).lower(),
            "representative_assignment_status": "assigned",
            "representative_assignment_warning": warning,
        }
        representative_rows.append(row)
        representatives_by_sku[sku["sku_id"]] = row

    for sku in skus:
        rep = representatives_by_sku[sku["sku_id"]]
        reserve_needed = to_int(sku, "reserve_pallets")
        assigned_for_sku = 0
        for reserve_index in range(1, reserve_needed + 1):
            if not available:
                shortages.append(
                    {
                        "selection_label": label,
                        "sku_id": sku["sku_id"],
                        "reserve_pallets_required": reserve_needed,
                        "reserve_pallets_assigned": assigned_for_sku,
                        "reserve_pallets_short": reserve_needed - assigned_for_sku,
                    }
                )
                break
            chosen = min(available.values(), key=lambda slot: reserve_sort_key(sku, slot, rep))
            chosen_slot_id = slot_id(chosen)
            del available[chosen_slot_id]
            same_block = str(chosen["block_id"]) == str(rep["block_id"])
            same_side = str(chosen["effective_access_side"]) == str(rep["effective_access_side"])
            level_met = reserve_level_preferred(sku["sku_class"], to_int(chosen, "level"))
            depth_met = reserve_depth_preferred(sku["sku_class"], to_int(chosen, "effective_depth"))
            warning = "" if level_met else "preferred reserve level exhausted; overflow slot used"
            reserve_rows.append(
                {
                    **layout_fields(chosen),
                    "sku_id": sku["sku_id"],
                    "sku_class": sku["sku_class"],
                    "class_index": sku["class_index"],
                    "global_sku_index": sku["global_sku_index"],
                    "demand_weight": sku["demand_weight"],
                    "pallets_per_sku": sku["pallets_per_sku"],
                    "reserve_pallets": sku["reserve_pallets"],
                    "reserve_pallet_index": reserve_index,
                    "reserve_slot_id": chosen_slot_id,
                    **copy_slot_fields(chosen),
                    "representative_slot_id": rep["representative_slot_id"],
                    "representative_block_id": rep["block_id"],
                    "representative_effective_access_side": rep["effective_access_side"],
                    "same_block_as_representative": str(same_block).lower(),
                    "same_side_as_representative": str(same_side).lower(),
                    "reserve_group_id": reserve_group_id(chosen),
                    "reserve_assignment_rule": f"{sku['sku_class']}_reserve_deterministic_sort",
                    "reserve_level_preference_met": str(level_met).lower(),
                    "reserve_depth_preference_met": str(depth_met).lower(),
                    "reserve_assignment_status": "assigned",
                    "reserve_assignment_warning": warning,
                }
            )
            assigned_for_sku += 1
    return representative_rows, reserve_rows, shortages


def validation_summary(
    rep_rows: list[dict[str, Any]],
    reserve_rows: list[dict[str, Any]],
    skus: list[dict[str, str]],
    shortages: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    duplicate_slot_validation: dict[str, Any] = {}
    representative_validation: dict[str, Any] = {}
    reserve_validation: dict[str, Any] = {}

    sku_ids = [sku["sku_id"] for sku in skus]
    reserve_expected_by_sku = {sku["sku_id"]: to_int(sku, "reserve_pallets") for sku in skus}

    for label in EXPECTED_LAYOUTS:
        reps = [row for row in rep_rows if row["selection_label"] == label]
        reserves = [row for row in reserve_rows if row["selection_label"] == label]
        rep_slots = [row["representative_slot_id"] for row in reps]
        reserve_slots = [row["reserve_slot_id"] for row in reserves]
        all_slots = rep_slots + reserve_slots
        rep_sku_counts = Counter(row["sku_id"] for row in reps)
        reserve_sku_counts = Counter(row["sku_id"] for row in reserves)
        rep_ok = (
            len(reps) == EXPECTED_REP_PER_LAYOUT
            and all(rep_sku_counts.get(sku_id, 0) == 1 for sku_id in sku_ids)
            and len(rep_slots) == len(set(rep_slots))
            and all(accessible(row) for row in reps)
            and all(row["access_type"] != "no_access" for row in reps)
        )
        reserve_ok = (
            len(reserves) == EXPECTED_RESERVE_PER_LAYOUT
            and all(reserve_sku_counts.get(sku_id, 0) == reserve_expected_by_sku[sku_id] for sku_id in sku_ids)
            and len(reserve_slots) == len(set(reserve_slots))
            and all(accessible(row) for row in reserves)
            and all(row["access_type"] != "no_access" for row in reserves)
        )
        duplicate_ok = (
            len(rep_slots) == len(set(rep_slots))
            and len(reserve_slots) == len(set(reserve_slots))
            and not set(rep_slots).intersection(reserve_slots)
            and len(all_slots) == len(set(all_slots))
        )
        representative_validation[label] = {
            "representative_rows": len(reps),
            "exactly_one_representative_per_sku": all(rep_sku_counts.get(sku_id, 0) == 1 for sku_id in sku_ids),
            "duplicate_representative_slots": len(rep_slots) - len(set(rep_slots)),
            "finite_representative_depth_distance": all(accessible(row) for row in reps),
            "no_no_access_representatives": all(row["access_type"] != "no_access" for row in reps),
            "passed": rep_ok,
        }
        reserve_validation[label] = {
            "reserve_rows": len(reserves),
            "expected_reserve_rows": EXPECTED_RESERVE_PER_LAYOUT,
            "reserve_rows_per_sku_match": all(
                reserve_sku_counts.get(sku_id, 0) == reserve_expected_by_sku[sku_id] for sku_id in sku_ids
            ),
            "duplicate_reserve_slots": len(reserve_slots) - len(set(reserve_slots)),
            "finite_reserve_depth_distance": all(accessible(row) for row in reserves),
            "no_no_access_reserves": all(row["access_type"] != "no_access" for row in reserves),
            "passed": reserve_ok,
        }
        duplicate_slot_validation[label] = {
            "representative_duplicate_slots": len(rep_slots) - len(set(rep_slots)),
            "reserve_duplicate_slots": len(reserve_slots) - len(set(reserve_slots)),
            "rep_reserve_overlap_slots": len(set(rep_slots).intersection(reserve_slots)),
            "any_duplicate_assigned_slot_within_layout": len(all_slots) != len(set(all_slots)),
            "passed": duplicate_ok,
        }
        if not rep_ok:
            warnings.append(f"{label}: representative validation failed")
        if not reserve_ok:
            warnings.append(f"{label}: reserve validation failed")
        if not duplicate_ok:
            warnings.append(f"{label}: duplicate slot validation failed")

    if shortages:
        warnings.append("reserve shortages occurred")
    no_forbidden_columns = not {"order_id", "order_line_id"}.intersection(REP_COLUMNS + RESERVE_COLUMNS)
    no_regime_columns = not any(column.lower().startswith("regime") for column in REP_COLUMNS + RESERVE_COLUMNS)
    scope_validation = {
        "no_order_fields_created": no_forbidden_columns,
        "no_regime_metrics_created": no_regime_columns,
        "shortage_count": len(shortages),
    }
    if not no_forbidden_columns:
        warnings.append("order fields were created")
    if not no_regime_columns:
        warnings.append("Regime metric fields were created")
    return duplicate_slot_validation, representative_validation, reserve_validation, scope_validation, warnings


def rows_by_layout(rep_rows: list[dict[str, Any]], reserve_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        label: sum(1 for row in rep_rows if row["selection_label"] == label)
        + sum(1 for row in reserve_rows if row["selection_label"] == label)
        for label in EXPECTED_LAYOUTS
    }


def count_by_layout(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {label: sum(1 for row in rows if row["selection_label"] == label) for label in EXPECTED_LAYOUTS}


def class_level_summary(rep_rows: list[dict[str, Any]], reserve_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in EXPECTED_LAYOUTS:
        summary[label] = {}
        for sku_class in ("A", "B", "C"):
            rep_class = [row for row in rep_rows if row["selection_label"] == label and row["sku_class"] == sku_class]
            reserve_class = [row for row in reserve_rows if row["selection_label"] == label and row["sku_class"] == sku_class]
            summary[label][sku_class] = {
                "representative_level_counts": dict(sorted(Counter(to_int(row, "level") for row in rep_class).items())),
                "reserve_level_counts": dict(sorted(Counter(to_int(row, "level") for row in reserve_class).items())),
                "representative_level_preference_met": sum(
                    1 for row in rep_class if row["representative_level_preference_met"] == "true"
                ),
                "reserve_level_preference_met": sum(
                    1 for row in reserve_class if row["reserve_level_preference_met"] == "true"
                ),
            }
    return summary


def class_depth_summary(rep_rows: list[dict[str, Any]], reserve_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in EXPECTED_LAYOUTS:
        summary[label] = {}
        for sku_class in ("A", "B", "C"):
            rep_depths = [
                to_int(row, "effective_depth")
                for row in rep_rows
                if row["selection_label"] == label and row["sku_class"] == sku_class
            ]
            reserve_depths = [
                to_int(row, "effective_depth")
                for row in reserve_rows
                if row["selection_label"] == label and row["sku_class"] == sku_class
            ]
            summary[label][sku_class] = {
                "representative_depth_counts": dict(sorted(Counter(rep_depths).items())),
                "reserve_depth_counts": dict(sorted(Counter(reserve_depths).items())),
                "representative_mean_depth": round(sum(rep_depths) / len(rep_depths), 6) if rep_depths else None,
                "reserve_mean_depth": round(sum(reserve_depths) / len(reserve_depths), 6) if reserve_depths else None,
            }
    return summary


def update_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["assignment"] = {
        "assignment_rule": "hybrid_representative_access_plus_reserve_fill",
        "assignment_randomness": False,
        "representative_access_pallets_per_sku": 1,
        "reserve_pallet_rule": "pallets_per_sku_minus_one",
        "assignment_order": "A_then_B_then_C_sku_id_ascending",
        "representative_assignment_output": as_posix(REP_CSV),
        "reserve_assignment_output": as_posix(RESERVE_CSV),
        "m6_summary_output": as_posix(SUMMARY_JSON),
    }
    completed = list(updated.get("milestones_completed", []))
    if "M6" not in completed:
        completed.append("M6")
    updated["milestones_completed"] = completed
    return updated


def summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Layout | Representative rows | Reserve rows | Assigned pallets | Shortages |",
        "|---|---:|---:|---:|---:|",
    ]
    shortage_by_layout = Counter(item["selection_label"] for item in summary["shortages"])
    for label in EXPECTED_LAYOUTS:
        lines.append(
            f"| {label} | {summary['representative_rows_by_layout'][label]} | "
            f"{summary['reserve_rows_by_layout'][label]} | {summary['assigned_pallets_by_layout'][label]} | "
            f"{shortage_by_layout.get(label, 0)} |"
        )
    return "\n".join(lines)


def class_summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Layout | Class | Rep level counts | Reserve level counts | Rep depth counts | Reserve depth counts |",
        "|---|---|---|---|---|---|",
    ]
    for label in EXPECTED_LAYOUTS:
        for sku_class in ("A", "B", "C"):
            level = summary["class_level_summary_by_layout"][label][sku_class]
            depth = summary["class_depth_summary_by_layout"][label][sku_class]
            lines.append(
                f"| {label} | {sku_class} | `{level['representative_level_counts']}` | "
                f"`{level['reserve_level_counts']}` | `{depth['representative_depth_counts']}` | "
                f"`{depth['reserve_depth_counts']}` |"
            )
    return "\n".join(lines)


def validation_lines(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Duplicate-slot validation: `{all(v['passed'] for v in summary['duplicate_slot_validation'].values())}`",
            f"- Representative validation: `{all(v['passed'] for v in summary['representative_validation'].values())}`",
            f"- Reserve validation: `{all(v['passed'] for v in summary['reserve_validation'].values())}`",
            f"- No order fields created: `{summary['scope_validation']['no_order_fields_created']}`",
            f"- No Regime metrics created: `{summary['scope_validation']['no_regime_metrics_created']}`",
            f"- Shortage count: `{summary['scope_validation']['shortage_count']}`",
        ]
    )


def write_report(summary: dict[str, Any]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    report = f"""# Operational-layer representative/reserve assignment

## Inputs

- Slot metrics: `{summary['input_slot_metrics_csv']}`
- SKU catalog: `{summary['input_sku_catalog_csv']}`
- Selected layouts: `{as_posix(SELECTED_LAYOUTS_CSV)}`
- Configuration: `{as_posix(CONFIG_JSON)}`
- M4 summary: `{as_posix(M4_SUMMARY_JSON)}`
- M5 summary: `{as_posix(M5_SUMMARY_JSON)}`

## Validation summary

{validation_lines(summary)}

## Assignment summary by layout

{summary_table(summary)}

## Class-level/depth summary

{class_summary_table(summary)}

## Output files

- Representative access assignments: `{summary['representative_access_assignment_csv']}`
- Reserve pallet assignments: `{summary['reserve_pallet_assignment_csv']}`
- Summary JSON: `{as_posix(SUMMARY_JSON)}`
- Report: `{as_posix(REPORT_MD)}`

## Readiness

Ready for Milestone 7: `{summary['ready_for_milestone_7']}`

## Warnings

{warnings}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    config, _selected, slots, skus = load_inputs()
    slots_by_layout: dict[str, list[dict[str, str]]] = {
        label: [slot for slot in slots if slot["selection_label"] == label] for label in EXPECTED_LAYOUTS
    }

    all_reps: list[dict[str, Any]] = []
    all_reserves: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    for label in EXPECTED_LAYOUTS:
        reps, reserves, layout_shortages = assign_layout(label, slots_by_layout[label], skus)
        all_reps.extend(reps)
        all_reserves.extend(reserves)
        shortages.extend(layout_shortages)

    duplicate_validation, rep_validation, reserve_validation, scope_validation, warnings = validation_summary(
        all_reps,
        all_reserves,
        skus,
        shortages,
    )
    if warnings:
        raise RuntimeError("M6 validation failed before writing outputs: " + "; ".join(warnings))

    write_csv(REP_CSV, all_reps, REP_COLUMNS)
    write_csv(RESERVE_CSV, all_reserves, RESERVE_COLUMNS)
    updated_config = update_config(config)
    CONFIG_JSON.write_text(json.dumps(updated_config, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_slot_metrics_csv": as_posix(SLOT_METRICS_CSV),
        "input_sku_catalog_csv": as_posix(SKU_CATALOG_CSV),
        "representative_access_assignment_csv": as_posix(REP_CSV),
        "reserve_pallet_assignment_csv": as_posix(RESERVE_CSV),
        "selected_layouts_processed": EXPECTED_LAYOUTS,
        "sku_count_total": len(skus),
        "assignment_randomness_used": False,
        "representative_rows_total": len(all_reps),
        "reserve_rows_total": len(all_reserves),
        "assigned_pallets_total": len(all_reps) + len(all_reserves),
        "expected_representative_rows_total": EXPECTED_REP_PER_LAYOUT * len(EXPECTED_LAYOUTS),
        "expected_reserve_rows_total": EXPECTED_RESERVE_PER_LAYOUT * len(EXPECTED_LAYOUTS),
        "expected_assigned_pallets_total": EXPECTED_PALLETS_PER_LAYOUT * len(EXPECTED_LAYOUTS),
        "rows_by_layout": rows_by_layout(all_reps, all_reserves),
        "representative_rows_by_layout": count_by_layout(all_reps),
        "reserve_rows_by_layout": count_by_layout(all_reserves),
        "assigned_pallets_by_layout": rows_by_layout(all_reps, all_reserves),
        "expected_assigned_pallets_by_layout": {label: EXPECTED_PALLETS_PER_LAYOUT for label in EXPECTED_LAYOUTS},
        "duplicate_slot_validation": duplicate_validation,
        "representative_validation": rep_validation,
        "reserve_validation": reserve_validation,
        "scope_validation": scope_validation,
        "class_level_summary_by_layout": class_level_summary(all_reps, all_reserves),
        "class_depth_summary_by_layout": class_depth_summary(all_reps, all_reserves),
        "shortages": shortages,
        "warnings": warnings,
        "blockers_or_warnings": warnings,
        "ready_for_milestone_7": not warnings,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary)

    print("Milestone 6 representative/reserve assignment complete.")
    print(f"representative_access_assignment.csv: {rel_posix(REP_CSV)}")
    print(f"reserve_pallet_assignment.csv: {rel_posix(RESERVE_CSV)}")
    print(f"summary JSON: {rel_posix(SUMMARY_JSON)}")
    print(f"Markdown report: {rel_posix(REPORT_MD)}")
    print(f"representative rows total: {len(all_reps)}")
    print(f"reserve rows total: {len(all_reserves)}")
    print(f"assigned pallets total: {len(all_reps) + len(all_reserves)}")
    print(f"rows by layout: {json.dumps(summary['rows_by_layout'], sort_keys=True)}")
    print(f"duplicate-slot validation: {all(v['passed'] for v in duplicate_validation.values())}")
    print(f"representative validation: {all(v['passed'] for v in rep_validation.values())}")
    print(f"reserve validation: {all(v['passed'] for v in reserve_validation.values())}")
    print(f"shortage summary: {json.dumps(shortages)}")
    print("warnings or blockers: none")
    print(f"ready_for_milestone_7: {summary['ready_for_milestone_7']}")


if __name__ == "__main__":
    main()
