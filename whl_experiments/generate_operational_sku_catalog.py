"""Generate the deterministic operational SKU catalog and configuration update."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = PROJECT_ROOT / "data" / "operational_layer"
DATA_ROOT = OP_ROOT / "paper_inputs"
LOG_ROOT = OP_ROOT / "paper_outputs" / "logs"
DOC_ROOT = PROJECT_ROOT / "docs"

CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"
SELECTED_LAYOUTS_CSV = DATA_ROOT / "selected_layouts.csv"
SLOT_METRICS_CSV = DATA_ROOT / "slot_metrics_by_layout.csv"
M4_SUMMARY_JSON = LOG_ROOT / "m4_slot_metrics_summary.json"

SKU_CATALOG_CSV = DATA_ROOT / "sku_catalog.csv"
SUMMARY_JSON = LOG_ROOT / "m5_sku_catalog_summary.json"
REPORT_MD = DOC_ROOT / "095_operational_layer_sku_catalog.md"

SKU_CATALOG_SEED = 20260617
EXPECTED_TOTAL_SKUS = 100
EXPECTED_TOTAL_PALLETS = 790

SKU_COLUMNS = [
    "sku_id",
    "sku_class",
    "class_index",
    "global_sku_index",
    "class_sku_count",
    "class_demand_share",
    "demand_weight",
    "pallets_per_sku",
    "total_pallets_for_sku",
    "representative_access_pallets",
    "reserve_pallets",
    "preferred_representative_levels",
    "preferred_reserve_levels",
    "allocation_priority",
    "demand_assignment_rule",
    "pallet_quantity_rule",
    "sku_catalog_seed",
]

CLASS_SPECS = {
    "A": {
        "count": 20,
        "demand_share": Decimal("0.80"),
        "pallets_per_sku": 7,
        "preferred_representative_levels_csv": "0-1",
        "preferred_reserve_levels_csv": "0-2_preferred_overflow_allowed",
        "preferred_representative_levels_config": [0, 1],
        "preferred_reserve_levels_config": [0, 1, 2],
    },
    "B": {
        "count": 30,
        "demand_share": Decimal("0.15"),
        "pallets_per_sku": 15,
        "preferred_representative_levels_csv": "0-2",
        "preferred_reserve_levels_csv": "0-3_preferred_overflow_allowed",
        "preferred_representative_levels_config": [0, 1, 2],
        "preferred_reserve_levels_config": [0, 1, 2, 3],
    },
    "C": {
        "count": 50,
        "demand_share": Decimal("0.05"),
        "pallets_per_sku": 4,
        "preferred_representative_levels_csv": "any",
        "preferred_reserve_levels_csv": "3-7_preferred_deeper_upper_allowed",
        "preferred_representative_levels_config": "any",
        "preferred_reserve_levels_config": [3, 4, 5, 6, 7],
    },
}


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
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def decimal_string(value: Decimal) -> str:
    return format(value, "f")


def load_required_inputs() -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    missing = [
        path
        for path in (CONFIG_JSON, SELECTED_LAYOUTS_CSV, SLOT_METRICS_CSV, M4_SUMMARY_JSON)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required M5 input(s): " + ", ".join(as_posix(path) for path in missing))

    config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    if config.get("vertical_levels") != 8:
        raise RuntimeError(f"vertical_levels is missing or not 8 in operational_config.json: {config.get('vertical_levels')!r}")

    selected_layouts = read_csv(SELECTED_LAYOUTS_CSV)
    if len(selected_layouts) != 4:
        raise RuntimeError(f"selected_layouts.csv must contain exactly 4 L1-L4 rows; observed {len(selected_layouts)}")
    if [row.get("selection_label") for row in selected_layouts] != ["L1", "L2", "L3", "L4"]:
        raise RuntimeError("selected_layouts.csv labels are not exactly L1, L2, L3, L4")

    m4_summary = json.loads(M4_SUMMARY_JSON.read_text(encoding="utf-8"))
    if m4_summary.get("ready_for_milestone_5") is not True:
        raise RuntimeError("M4 summary does not report ready_for_milestone_5=true")

    return config, selected_layouts, m4_summary


def generate_catalog_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 1
    allocation_priority = 1
    for sku_class in ("A", "B", "C"):
        spec = CLASS_SPECS[sku_class]
        count = int(spec["count"])
        demand_share = spec["demand_share"]
        demand_weight = demand_share / Decimal(count)
        pallets_per_sku = int(spec["pallets_per_sku"])
        for class_index in range(1, count + 1):
            rows.append(
                {
                    "sku_id": f"SKU_{sku_class}{class_index:03d}",
                    "sku_class": sku_class,
                    "class_index": class_index,
                    "global_sku_index": global_index,
                    "class_sku_count": count,
                    "class_demand_share": decimal_string(demand_share),
                    "demand_weight": decimal_string(demand_weight),
                    "pallets_per_sku": pallets_per_sku,
                    "total_pallets_for_sku": pallets_per_sku,
                    "representative_access_pallets": 1,
                    "reserve_pallets": pallets_per_sku - 1,
                    "preferred_representative_levels": spec["preferred_representative_levels_csv"],
                    "preferred_reserve_levels": spec["preferred_reserve_levels_csv"],
                    "allocation_priority": allocation_priority,
                    "demand_assignment_rule": "deterministic_equal_within_class",
                    "pallet_quantity_rule": "fixed_by_class",
                    "sku_catalog_seed": SKU_CATALOG_SEED,
                }
            )
            global_index += 1
            allocation_priority += 1
    return rows


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {sku_class: sum(1 for row in rows if row["sku_class"] == sku_class) for sku_class in ("A", "B", "C")}


def demand_sums(rows: list[dict[str, Any]]) -> dict[str, Decimal]:
    return {
        sku_class: sum(
            Decimal(str(row["demand_weight"])) for row in rows if row["sku_class"] == sku_class
        )
        for sku_class in ("A", "B", "C")
    }


def pallet_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        sku_class: sum(int(row["total_pallets_for_sku"]) for row in rows if row["sku_class"] == sku_class)
        for sku_class in ("A", "B", "C")
    }


def selected_layout_capacities(selected_layouts: list[dict[str, str]]) -> dict[str, int]:
    return {
        row["selection_label"]: int(round(float(row["pallet_slot_capacity"])))
        for row in selected_layouts
    }


def selected_layout_references(selected_layouts: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_label": row["selection_label"],
            "selection_type": row["selection_type"],
            "layout_signature": row["layout_signature"],
            "seed": int(row["seed"]),
            "rank": int(row["rank"]),
            "pallet_slot_capacity": int(round(float(row["pallet_slot_capacity"]))),
            "archive_npz_path": row.get("archive_npz_path", ""),
            "archive_key": row.get("archive_key", ""),
        }
        for row in selected_layouts
    ]


def inventory_utilization(capacities: dict[str, int], inventory_total: int) -> dict[str, float]:
    return {
        label: round(inventory_total / capacity, 9)
        for label, capacity in capacities.items()
    }


def validate_catalog(
    rows: list[dict[str, Any]],
    config_before: dict[str, Any],
    config_after: dict[str, Any],
    selected_layouts: list[dict[str, str]],
    check_output_files: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    counts = class_counts(rows)
    demand_by_class = demand_sums(rows)
    pallets_by_class = pallet_totals(rows)
    total_demand = sum(demand_by_class.values())
    total_pallets = sum(pallets_by_class.values())

    expected_counts = {sku_class: int(CLASS_SPECS[sku_class]["count"]) for sku_class in ("A", "B", "C")}
    expected_demand = {sku_class: CLASS_SPECS[sku_class]["demand_share"] for sku_class in ("A", "B", "C")}
    expected_pallets = {
        sku_class: int(CLASS_SPECS[sku_class]["count"]) * int(CLASS_SPECS[sku_class]["pallets_per_sku"])
        for sku_class in ("A", "B", "C")
    }

    if len(rows) != EXPECTED_TOTAL_SKUS:
        warnings.append(f"total SKU count is {len(rows)}, expected {EXPECTED_TOTAL_SKUS}")
    if counts != expected_counts:
        warnings.append(f"SKU counts by class mismatch: observed {counts}, expected {expected_counts}")
    if demand_by_class != expected_demand:
        warnings.append(
            "demand sums by class mismatch: observed "
            + str({key: decimal_string(value) for key, value in demand_by_class.items()})
        )
    if total_demand != Decimal("1.00"):
        warnings.append(f"total demand sum is {decimal_string(total_demand)}, expected 1.00")
    if pallets_by_class != expected_pallets:
        warnings.append(f"pallet totals by class mismatch: observed {pallets_by_class}, expected {expected_pallets}")
    if total_pallets != EXPECTED_TOTAL_PALLETS:
        warnings.append(f"total pallets is {total_pallets}, expected {EXPECTED_TOTAL_PALLETS}")
    if any(int(row["representative_access_pallets"]) != 1 for row in rows):
        warnings.append("representative_access_pallets is not 1 for every SKU")
    if any(int(row["reserve_pallets"]) != int(row["pallets_per_sku"]) - 1 for row in rows):
        warnings.append("reserve_pallets is not pallets_per_sku - 1 for every SKU")
    forbidden_columns = {"slot_id", "row", "col", "level", "order_id", "order_line_id"}
    present_forbidden = sorted(forbidden_columns.intersection(SKU_COLUMNS))
    if present_forbidden:
        warnings.append("forbidden assignment/order columns appear in sku_catalog.csv: " + ", ".join(present_forbidden))
    if config_after.get("vertical_levels") != 8:
        warnings.append("vertical_levels was not preserved as 8 in operational_config.json")
    missing_preserved_keys = [key for key in config_before if key not in config_after]
    if missing_preserved_keys:
        warnings.append("previous config top-level keys were removed: " + ", ".join(missing_preserved_keys))
    if [row["selection_label"] for row in selected_layouts] != ["L1", "L2", "L3", "L4"]:
        warnings.append("selected L1-L4 layout information is damaged or missing")
    if check_output_files and not SKU_CATALOG_CSV.is_file():
        warnings.append("sku_catalog.csv was not written")
    if check_output_files and not CONFIG_JSON.is_file():
        warnings.append("operational_config.json is missing after update")

    validation = {
        "sku_catalog_csv_exists": SKU_CATALOG_CSV.is_file() if check_output_files else None,
        "operational_config_json_exists": CONFIG_JSON.is_file() if check_output_files else None,
        "previous_config_top_level_keys_preserved": not missing_preserved_keys,
        "sku_count_total": len(rows),
        "sku_counts_by_class": counts,
        "demand_sum_by_class": {key: float(value) for key, value in demand_by_class.items()},
        "demand_sum_total": float(total_demand),
        "pallet_total_by_class": pallets_by_class,
        "pallet_total_all": total_pallets,
        "representative_access_pallets_all_one": all(int(row["representative_access_pallets"]) == 1 for row in rows),
        "reserve_pallets_rule_valid": all(
            int(row["reserve_pallets"]) == int(row["pallets_per_sku"]) - 1 for row in rows
        ),
        "no_slot_assignment_columns": not present_forbidden,
        "no_order_columns": not {"order_id", "order_line_id"}.intersection(SKU_COLUMNS),
        "vertical_levels_remains_8": config_after.get("vertical_levels") == 8,
        "selected_layout_information_available": [row["selection_label"] for row in selected_layouts] == ["L1", "L2", "L3", "L4"],
    }
    return validation, warnings


def update_config(
    config: dict[str, Any],
    selected_layouts: list[dict[str, str]],
    capacities: dict[str, int],
    utilization: dict[str, float],
) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["sku_catalog"] = {
        "sku_catalog_csv_path": as_posix(SKU_CATALOG_CSV),
        "sku_count_total": EXPECTED_TOTAL_SKUS,
        "sku_catalog_seed": SKU_CATALOG_SEED,
        "sku_classes": {
            sku_class: {
                "count": int(spec["count"]),
                "demand_share": float(spec["demand_share"]),
                "pallets_per_sku": int(spec["pallets_per_sku"]),
                "demand_assignment": "deterministic_equal_within_class",
                "preferred_representative_levels": spec["preferred_representative_levels_config"],
                "preferred_reserve_levels": spec["preferred_reserve_levels_config"],
            }
            for sku_class, spec in CLASS_SPECS.items()
        },
        "within_class_demand_randomness": False,
        "within_class_allocation_randomness": False,
        "pallet_quantity_rule": "fixed_by_class",
        "representative_access_pallets_per_sku": 1,
        "reserve_pallets_rule": "pallets_per_sku_minus_one",
        "order_generation_depends_on": "demand_weight_only",
        "pallet_quantity_affects": "reserve_assignment_and_storage_utilization_only",
    }
    updated["selected_layouts"] = {
        "selected_layouts_csv_path": as_posix(SELECTED_LAYOUTS_CSV),
        "layout_count": len(selected_layouts),
        "layouts": selected_layout_references(selected_layouts),
    }
    updated["m5_capacity_context"] = {
        "inventory_total_pallets": EXPECTED_TOTAL_PALLETS,
        "selected_layout_capacities": capacities,
        "inventory_utilization_by_layout": utilization,
        "diagnostic_only": True,
    }
    completed = list(updated.get("milestones_completed", []))
    if not completed:
        completed = ["M0", "M1", "M2", "M3C", "M4"]
    if "M5" not in completed:
        completed.append("M5")
    updated["milestones_completed"] = completed
    return updated


def write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def sku_design_table() -> str:
    lines = [
        "| Class | SKUs | Demand share | Demand/SKU | Pallets/SKU | Total pallets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sku_class in ("A", "B", "C"):
        spec = CLASS_SPECS[sku_class]
        count = int(spec["count"])
        demand_share = spec["demand_share"]
        demand_per_sku = demand_share / Decimal(count)
        total_pallets = count * int(spec["pallets_per_sku"])
        lines.append(
            f"| {sku_class} | {count} | {decimal_string(demand_share)} | "
            f"{decimal_string(demand_per_sku)} | {spec['pallets_per_sku']} | {total_pallets} |"
        )
    return "\n".join(lines)


def utilization_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Layout | Capacity | Inventory pallets | Utilization |",
        "|---|---:|---:|---:|",
    ]
    for label in ("L1", "L2", "L3", "L4"):
        lines.append(
            f"| {label} | {summary['selected_layout_capacities'][label]} | "
            f"{summary['pallet_total_all']} | {summary['inventory_utilization_by_layout'][label]:.6f} |"
        )
    return "\n".join(lines)


def validation_lines(validation: dict[str, Any]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in validation.items())


def write_report(summary: dict[str, Any]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    report = f"""# Operational-layer SKU catalog

## Inputs

- Configuration: `{summary['input_files']['operational_config_json']}`
- Selected layouts: `{summary['input_files']['selected_layouts_csv']}`
- Slot metrics: `{summary['input_files']['slot_metrics_by_layout_csv']}`
- M4 summary: `{summary['input_files']['m4_slot_metrics_summary_json']}`

## SKU catalog design

{sku_design_table()}

## Capacity/utilization context

{utilization_table(summary)}

## Validation summary

{validation_lines(summary['validation'])}

## Output files

- SKU catalog: `{summary['sku_catalog_csv_path']}`
- Configuration: `{summary['operational_config_path']}`
- Summary JSON: `{summary['summary_json_path']}`
- Report: `{summary['markdown_report_path']}`

## Readiness

Ready for Milestone 6: `{summary['ready_for_milestone_6']}`

## Warnings

{warnings}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    config_before, selected_layouts, _m4_summary = load_required_inputs()
    rows = generate_catalog_rows()
    capacities = selected_layout_capacities(selected_layouts)
    utilization = inventory_utilization(capacities, EXPECTED_TOTAL_PALLETS)

    updated_config = update_config(config_before, selected_layouts, capacities, utilization)
    validation, warnings = validate_catalog(
        rows,
        config_before,
        updated_config,
        selected_layouts,
        check_output_files=False,
    )
    if warnings:
        raise RuntimeError("M5 validation failed before writing outputs: " + "; ".join(warnings))

    write_csv(SKU_CATALOG_CSV, rows, SKU_COLUMNS)
    CONFIG_JSON.write_text(json.dumps(updated_config, indent=2) + "\n", encoding="utf-8")

    final_config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    validation, warnings = validate_catalog(rows, config_before, final_config, selected_layouts)

    counts = class_counts(rows)
    demand_by_class = demand_sums(rows)
    pallets_by_class = pallet_totals(rows)
    representative_total = sum(int(row["representative_access_pallets"]) for row in rows)
    reserve_total = sum(int(row["reserve_pallets"]) for row in rows)
    total_demand = sum(demand_by_class.values())
    total_pallets = sum(pallets_by_class.values())

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "operational_config_json": as_posix(CONFIG_JSON),
            "selected_layouts_csv": as_posix(SELECTED_LAYOUTS_CSV),
            "slot_metrics_by_layout_csv": as_posix(SLOT_METRICS_CSV),
            "m4_slot_metrics_summary_json": as_posix(M4_SUMMARY_JSON),
        },
        "sku_catalog_csv_path": as_posix(SKU_CATALOG_CSV),
        "operational_config_path": as_posix(CONFIG_JSON),
        "summary_json_path": as_posix(SUMMARY_JSON),
        "markdown_report_path": as_posix(REPORT_MD),
        "sku_catalog_seed": SKU_CATALOG_SEED,
        "sku_count_total": len(rows),
        "sku_counts_by_class": counts,
        "demand_sum_total": float(total_demand),
        "demand_sum_by_class": {key: float(value) for key, value in demand_by_class.items()},
        "pallets_per_sku_by_class": {
            sku_class: int(CLASS_SPECS[sku_class]["pallets_per_sku"]) for sku_class in ("A", "B", "C")
        },
        "pallet_total_by_class": pallets_by_class,
        "pallet_total_all": total_pallets,
        "representative_access_pallets_total": representative_total,
        "reserve_pallets_total": reserve_total,
        "within_class_demand_randomness": False,
        "within_class_allocation_randomness": False,
        "selected_layout_capacities": capacities,
        "inventory_utilization_by_layout": utilization,
        "validation": validation,
        "warnings": warnings,
        "blockers_or_warnings": warnings,
        "ready_for_milestone_6": not warnings,
    }
    write_summary(summary)
    write_report(summary)

    if warnings:
        raise RuntimeError("M5 validation failed after writing outputs: " + "; ".join(warnings))

    print("Milestone 5 SKU catalog/config generation complete.")
    print(f"sku_catalog.csv: {rel_posix(SKU_CATALOG_CSV)}")
    print(f"operational_config.json: {rel_posix(CONFIG_JSON)}")
    print(f"summary JSON: {rel_posix(SUMMARY_JSON)}")
    print(f"Markdown report: {rel_posix(REPORT_MD)}")
    print(f"SKU counts by class: {json.dumps(counts, sort_keys=True)}")
    print(f"demand sums by class: {json.dumps(summary['demand_sum_by_class'], sort_keys=True)} total={summary['demand_sum_total']}")
    print(f"pallet totals by class: {json.dumps(pallets_by_class, sort_keys=True)} total={total_pallets}")
    print(f"inventory utilization by layout: {json.dumps(utilization, sort_keys=True)}")
    print("randomness used: none")
    print("warnings or blockers: none")
    print(f"ready_for_milestone_6: {summary['ready_for_milestone_6']}")


if __name__ == "__main__":
    main()
