"""Summarize validated Low/Medium/High Section-7 occupancy cases.

The analysis is deterministic and descriptive. It reads validated occupancy
outputs, checks cross-occupancy invariants, and writes reviewer-facing CSV
evidence without modifying the scientific inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

LAYOUTS = ("L1", "L2", "L3", "L4")
CASE_TOTALS = {"low": 790, "medium": 2240, "high": 3584}
CASE_RESERVES = {"low": 690, "medium": 2140, "high": 3484}
ABS_TOLERANCE = 1e-9

REP_REL = Path("data") / "representative_access_assignment.csv"
RESERVE_REL = Path("data") / "reserve_pallet_assignment.csv"
REGIME_A_REL = Path("data") / "regime_A_metrics.csv"
REGIME_B_REL = Path("data") / "regime_B_metrics.csv"
FRAGMENTATION_REL = Path("data") / "reserve_fragmentation_summary.csv"
MANIFEST_REL = Path("logs") / "occupancy_manifest.json"
VALIDATION_REL = Path("logs") / "validation_summary.json"

REP_INVARIANT_COLUMNS = (
    "selection_label", "layout_signature", "sku_id", "sku_class",
    "global_sku_index", "demand_weight", "representative_access_pallets",
    "representative_slot_id", "row", "col", "level", "block_id", "block_size",
    "access_type", "effective_access_side", "effective_pick_face_row",
    "effective_pick_face_col", "effective_depth", "horizontal_access_distance",
    "normalized_distance", "normalized_depth", "normalized_level", "slot_cost",
)

OCCUPANCY_COLUMNS = (
    "occupancy_label", "inventory_total", "selection_label",
    "total_pallet_slot_capacity", "assigned_inventory_utilization",
    "reserve_pallets_assigned", "interior_deep_slot_share",
    "reserve_deep_pallet_share", "low_depth_level_weight_access_cost",
)

COMPONENT_COLUMNS = (
    "occupancy_label", "inventory_total", "selection_label",
    "reserve_pallet_count", "mean_horizontal_access_distance",
    "mean_normalized_horizontal_distance", "mean_effective_depth",
    "mean_normalized_depth", "mean_vertical_level", "mean_normalized_level",
    "lambda_depth", "lambda_level",
    "reconstructed_low_depth_level_weight_access_cost",
    "reported_low_depth_level_weight_access_cost",
    "absolute_reconstruction_difference", "verification_status",
)

FRAGMENTATION_LAYOUT_COLUMNS = (
    "occupancy_label", "inventory_total", "selection_label",
    "sku_count_total", "reserve_pallets_assigned_total",
    "sku_count_weighted_mean_groups_per_sku",
    "sku_count_weighted_mean_blocks_per_sku",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field!r}")
    return value


def as_int(row: dict[str, Any], field: str) -> int:
    return int(round(float(row[field])))


def values_equivalent(left: Any, right: Any) -> bool:
    a = str(left).strip()
    b = str(right).strip()
    if a.lower() in {"true", "false"} and b.lower() in {"true", "false"}:
        return a.lower() == b.lower()
    try:
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            return math.isclose(fa, fb, rel_tol=0.0, abs_tol=ABS_TOLERANCE)
    except ValueError:
        pass
    return a == b


def projection(rows: Sequence[dict[str, Any]], columns: Sequence[str]):
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("selection_label", ""),
            int(float(row.get("global_sku_index", 0))),
        ),
    )
    return [tuple(str(row.get(column, "")) for column in columns) for row in ordered]


def rows_equivalent(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        values_equivalent(a.get(column, ""), b.get(column, ""))
        for a, b in zip(left, right, strict=True)
        for column in columns
    )


def load_case(label: str, root: Path) -> dict[str, Any]:
    root = root.resolve()
    required = (
        root / MANIFEST_REL,
        root / VALIDATION_REL,
        root / REP_REL,
        root / RESERVE_REL,
        root / REGIME_A_REL,
        root / REGIME_B_REL,
        root / FRAGMENTATION_REL,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}: missing validated file(s): {missing}")

    manifest = json.loads((root / MANIFEST_REL).read_text(encoding="utf-8"))
    validation = json.loads((root / VALIDATION_REL).read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{label}: manifest is not complete")
    if manifest.get("target_inventory_total") != CASE_TOTALS[label]:
        raise ValueError(f"{label}: unexpected inventory total")
    if validation.get("status") != "passed" or validation.get("failed_invariants"):
        raise ValueError(f"{label}: validation did not pass")

    rep_columns, representatives = read_csv(root / REP_REL)
    reserve_columns, reserves = read_csv(root / RESERVE_REL)
    regime_a_columns, regime_a = read_csv(root / REGIME_A_REL)
    regime_b_columns, regime_b = read_csv(root / REGIME_B_REL)
    fragmentation_columns, fragmentation = read_csv(root / FRAGMENTATION_REL)

    if len(representatives) != 400:
        raise ValueError(f"{label}: expected 400 representative rows")
    if len(reserves) != CASE_RESERVES[label] * 4:
        raise ValueError(f"{label}: unexpected reserve row count")
    if len(regime_a) != 4 or len(regime_b) != 4 or len(fragmentation) != 12:
        raise ValueError(f"{label}: incomplete regime/fragmentation output")

    hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in required
    }
    return {
        "label": label,
        "root": root,
        "manifest": manifest,
        "validation": validation,
        "rep_columns": rep_columns,
        "representatives": representatives,
        "reserve_columns": reserve_columns,
        "reserves": reserves,
        "regime_a_columns": regime_a_columns,
        "regime_a": regime_a,
        "regime_b_columns": regime_b_columns,
        "regime_b": regime_b,
        "fragmentation_columns": fragmentation_columns,
        "fragmentation": fragmentation,
        "hashes": hashes,
    }


def validate_cross_occupancy(cases: Sequence[dict[str, Any]]) -> dict[str, bool]:
    low = cases[0]
    rep_reference = projection(low["representatives"], REP_INVARIANT_COLUMNS)
    regime_a_reference = low["regime_a"]
    regime_a_columns = low["regime_a_columns"]
    deep_reference = {
        row["selection_label"]: row["interior_deep_slot_share"]
        for row in low["regime_b"]
    }
    source_reference = low["manifest"]["source_files"]

    checks = {
        "representative_assignments_identical": all(
            projection(case["representatives"], REP_INVARIANT_COLUMNS) == rep_reference
            for case in cases[1:]
        ),
        "scenario_A_metrics_identical": all(
            rows_equivalent(case["regime_a"], regime_a_reference, regime_a_columns)
            for case in cases[1:]
        ),
        "interior_deep_slot_share_identical": all(
            {
                row["selection_label"]: row["interior_deep_slot_share"]
                for row in case["regime_b"]
            }
            == deep_reference
            for case in cases[1:]
        ),
        "canonical_selected_layout_hash_identical": all(
            case["manifest"]["source_files"]["selected_layouts.csv"]["sha256"]
            == source_reference["selected_layouts.csv"]["sha256"]
            for case in cases[1:]
        ),
        "canonical_slot_metric_hash_identical": all(
            case["manifest"]["source_files"]["slot_metrics_by_layout.csv"]["sha256"]
            == source_reference["slot_metrics_by_layout.csv"]["sha256"]
            for case in cases[1:]
        ),
        "all_cases_shortage_free": all(
            not case["validation"].get("assignment_shortages") for case in cases
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("cross-occupancy validation failed: " + ", ".join(failed))
    return checks


def occupancy_rows(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        for row in case["regime_b"]:
            output.append(
                {
                    "occupancy_label": case["label"],
                    "inventory_total": CASE_TOTALS[case["label"]],
                    "selection_label": row["selection_label"],
                    "total_pallet_slot_capacity": row["total_pallet_slot_capacity"],
                    "assigned_inventory_utilization": row["assigned_inventory_utilization"],
                    "reserve_pallets_assigned": row["reserve_pallets_assigned"],
                    "interior_deep_slot_share": row["interior_deep_slot_share"],
                    "reserve_deep_pallet_share": row["reserve_deep_pallet_share"],
                    "low_depth_level_weight_access_cost": row[
                        "low_depth_level_weight_access_cost"
                    ],
                }
            )
    return output


def fragmentation_by_class_rows(cases: Sequence[dict[str, Any]]):
    columns = ["occupancy_label", "inventory_total"] + list(cases[0]["fragmentation_columns"])
    rows = []
    for case in cases:
        for row in case["fragmentation"]:
            rows.append(
                {
                    "occupancy_label": case["label"],
                    "inventory_total": CASE_TOTALS[case["label"]],
                    **row,
                }
            )
    return columns, rows


def mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return math.fsum(as_float(row, field) for row in rows) / len(rows)


def component_rows(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        regime_b = {row["selection_label"]: row for row in case["regime_b"]}
        for layout in LAYOUTS:
            reserves = [row for row in case["reserves"] if row["selection_label"] == layout]
            reported_row = regime_b[layout]
            lambda_depth = as_float(reported_row, "lambda_depth")
            lambda_level = as_float(reported_row, "lambda_level")
            nh = mean(reserves, "normalized_distance")
            nd = mean(reserves, "normalized_depth")
            nl = mean(reserves, "normalized_level")
            reconstructed = nh + lambda_depth * nd + lambda_level * nl
            reported = as_float(reported_row, "low_depth_level_weight_access_cost")
            difference = abs(reconstructed - reported)
            if difference > ABS_TOLERANCE:
                raise RuntimeError(
                    f"{case['label']}/{layout}: reserve-cost reconstruction differs by {difference}"
                )
            output.append(
                {
                    "occupancy_label": case["label"],
                    "inventory_total": CASE_TOTALS[case["label"]],
                    "selection_label": layout,
                    "reserve_pallet_count": len(reserves),
                    "mean_horizontal_access_distance": mean(
                        reserves, "horizontal_access_distance"
                    ),
                    "mean_normalized_horizontal_distance": nh,
                    "mean_effective_depth": mean(reserves, "effective_depth"),
                    "mean_normalized_depth": nd,
                    "mean_vertical_level": mean(reserves, "vertical_level"),
                    "mean_normalized_level": nl,
                    "lambda_depth": lambda_depth,
                    "lambda_level": lambda_level,
                    "reconstructed_low_depth_level_weight_access_cost": reconstructed,
                    "reported_low_depth_level_weight_access_cost": reported,
                    "absolute_reconstruction_difference": difference,
                    "verification_status": "passed",
                }
            )
    return output


def fragmentation_layout_rows(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        by_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in case["fragmentation"]:
            by_layout[row["selection_label"]].append(row)
        for layout in LAYOUTS:
            rows = by_layout[layout]
            sku_total = sum(as_int(row, "sku_count") for row in rows)
            reserve_total = sum(as_int(row, "reserve_pallets_assigned") for row in rows)
            if sku_total != 100 or reserve_total != CASE_RESERVES[case["label"]]:
                raise RuntimeError(f"{case['label']}/{layout}: fragmentation totals mismatch")
            groups = math.fsum(
                as_int(row, "sku_count") * as_float(row, "mean_groups_per_sku")
                for row in rows
            ) / sku_total
            blocks = math.fsum(
                as_int(row, "sku_count") * as_float(row, "mean_blocks_per_sku")
                for row in rows
            ) / sku_total
            output.append(
                {
                    "occupancy_label": case["label"],
                    "inventory_total": CASE_TOTALS[case["label"]],
                    "selection_label": layout,
                    "sku_count_total": sku_total,
                    "reserve_pallets_assigned_total": reserve_total,
                    "sku_count_weighted_mean_groups_per_sku": groups,
                    "sku_count_weighted_mean_blocks_per_sku": blocks,
                }
            )
    return output


def run_analysis(
    *,
    low_root: Path,
    medium_root: Path,
    high_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    roots = [low_root.resolve(), medium_root.resolve(), high_root.resolve()]
    output = output_root.resolve()
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("summary output must be isolated from occupancy case roots")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")

    cases = [
        load_case("low", roots[0]),
        load_case("medium", roots[1]),
        load_case("high", roots[2]),
    ]
    checks = validate_cross_occupancy(cases)
    occupancy = occupancy_rows(cases)
    fragmentation_columns, fragmentation_by_class = fragmentation_by_class_rows(cases)
    components = component_rows(cases)
    fragmentation_layout = fragmentation_layout_rows(cases)

    if not (
        len(occupancy) == 12
        and len(fragmentation_by_class) == 36
        and len(components) == 12
        and len(fragmentation_layout) == 12
    ):
        raise RuntimeError("unexpected cross-occupancy output row count")

    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "occupancy_summary.csv": (OCCUPANCY_COLUMNS, occupancy),
        "occupancy_fragmentation_by_class.csv": (
            fragmentation_columns,
            fragmentation_by_class,
        ),
        "occupancy_reserve_components.csv": (COMPONENT_COLUMNS, components),
        "occupancy_fragmentation_layout_summary.csv": (
            FRAGMENTATION_LAYOUT_COLUMNS,
            fragmentation_layout,
        ),
    }
    output_records = {}
    for filename, (columns, rows) in outputs.items():
        path = output / filename
        write_csv(path, columns, rows)
        output_records[filename] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": 1,
        "module": "whl_experiments.analyze_operational_occupancy_sensitivity",
        "case_totals": CASE_TOTALS,
        "reserve_totals": CASE_RESERVES,
        "cross_occupancy_invariants": checks,
        "source_cases": {
            case["label"]: {
                "root": str(case["root"]),
                "files": case["hashes"],
            }
            for case in cases
        },
        "reserve_access_formula": (
            "mean(normalized_distance) + lambda_depth*mean(normalized_depth) "
            "+ lambda_level*mean(normalized_level)"
        ),
        "outputs": output_records,
        "status": "complete",
    }
    (output / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validated Section-7 Low/Medium/High occupancy cases."
    )
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--medium-root", type=Path, required=True)
    parser.add_argument("--high-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        "--summary-root",
        dest="output_root",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_analysis(
            low_root=args.low_root,
            medium_root=args.medium_root,
            high_root=args.high_root,
            output_root=args.output_root,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "outputs": result["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
