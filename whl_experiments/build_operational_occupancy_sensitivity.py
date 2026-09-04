"""Build validated Section-7 operational occupancy-sensitivity cases.

The module reuses the public deterministic assignment and regime-metric functions.
It does not regenerate structural layouts or feed operational quantities back into
the optimizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from whl_experiments import assign_operational_sku_pallets as assignment
from whl_experiments import compute_operational_regime_metrics as regime_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATA_ROOT = PROJECT_ROOT / "data" / "operational_layer" / "paper_inputs"

LAYOUTS = ("L1", "L2", "L3", "L4")
EXPECTED_SIGNATURES = {
    "L1": "1fa9344c00a95c630e382533991ef575b2f5e6b5",
    "L2": "76cdc821160b1fd8a5952575af463e70cb84ba4e",
    "L3": "fc4825140fa5e2560776b4b932f5ef46c1588f36",
    "L4": "31618e0fa9e7ba38a349d70c4dff96d8a35cbd09",
}
EXPECTED_CAPACITIES = {"L1": 4480, "L2": 5056, "L3": 5440, "L4": 4800}
BASE_TOTAL = 790
EXPECTED_SKUS = 100
EXPECTED_CLASSES = {"A": 20, "B": 30, "C": 50}
NAMED_TOTALS = {"low": 790, "medium": 2240, "high": 3584}
SCALING_RULE = "proportional_largest_remainder_v1"
TIE_BREAK = "descending_fractional_remainder_then_ascending_global_sku_index"

SOURCE_FILES = {
    "selected": "selected_layouts.csv",
    "slots": "slot_metrics_by_layout.csv",
    "sku": "sku_catalog.csv",
}
LOW_VERIFY_FILES = {
    "representatives": "representative_access_assignment.csv",
    "reserves": "reserve_pallet_assignment.csv",
    "regime_a": "regime_A_metrics.csv",
    "regime_b": "regime_B_metrics.csv",
    "fragmentation": "reserve_fragmentation_summary.csv",
}

REP_INVARIANT_COLUMNS = (
    "selection_label", "layout_signature", "sku_id", "sku_class",
    "global_sku_index", "demand_weight", "representative_access_pallets",
    "representative_slot_id", "row", "col", "level", "block_id", "block_size",
    "access_type", "effective_access_side", "effective_pick_face_row",
    "effective_pick_face_col", "effective_depth", "horizontal_access_distance",
    "normalized_distance", "normalized_depth", "normalized_level", "slot_cost",
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


def as_int(row: dict[str, Any], field: str) -> int:
    return int(round(float(row[field])))


def values_equivalent(left: Any, right: Any, abs_tol: float = 1e-9) -> bool:
    a = str(left).strip()
    b = str(right).strip()
    if a.lower() in {"true", "false"} and b.lower() in {"true", "false"}:
        return a.lower() == b.lower()
    try:
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            return math.isclose(fa, fb, rel_tol=0.0, abs_tol=abs_tol)
    except ValueError:
        pass
    return a == b


def rows_equivalent(
    generated: Sequence[dict[str, Any]],
    canonical: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> bool:
    if len(generated) != len(canonical):
        return False
    return all(
        values_equivalent(g.get(column, ""), c.get(column, ""))
        for g, c in zip(generated, canonical, strict=True)
        for column in columns
    )


def scale_inventory(
    base_rows: Sequence[dict[str, Any]], target_total: int
) -> list[dict[str, Any]]:
    if target_total < EXPECTED_SKUS:
        raise ValueError("inventory total must allow at least one pallet per SKU")
    if len(base_rows) != EXPECTED_SKUS:
        raise ValueError(f"expected {EXPECTED_SKUS} SKUs, observed {len(base_rows)}")

    ordered = sorted(
        (deepcopy(dict(row)) for row in base_rows),
        key=lambda row: as_int(row, "global_sku_index"),
    )
    if dict(Counter(row["sku_class"] for row in ordered)) != EXPECTED_CLASSES:
        raise ValueError("SKU class counts do not match the locked 20/30/50 catalog")
    if sum(as_int(row, "pallets_per_sku") for row in ordered) != BASE_TOTAL:
        raise ValueError("base SKU catalog does not total 790 pallets")

    work: list[dict[str, Any]] = []
    allocated = 0
    for row in ordered:
        old = as_int(row, "pallets_per_sku")
        quota = Fraction(target_total * old, BASE_TOTAL)
        floor = quota.numerator // quota.denominator
        quantity = max(1, floor)
        work.append(
            {
                "row": row,
                "old": old,
                "index": as_int(row, "global_sku_index"),
                "quantity": quantity,
                "remainder": quota - floor,
            }
        )
        allocated += quantity

    remaining = target_total - allocated
    if remaining < 0 or remaining > len(work):
        raise ValueError("invalid largest-remainder allocation state")
    order = sorted(work, key=lambda item: (-item["remainder"], item["index"]))
    for item in order[:remaining]:
        item["quantity"] += 1

    scaled: list[dict[str, Any]] = []
    for item in work:
        row = item["row"]
        quantity = int(item["quantity"])
        row["pallets_per_sku"] = quantity
        row["total_pallets_for_sku"] = quantity
        row["representative_access_pallets"] = 1
        row["reserve_pallets"] = quantity - 1
        row["base_pallets_per_sku"] = item["old"]
        row["occupancy_target_total"] = target_total
        row["occupancy_scaling_rule"] = SCALING_RULE
        row["occupancy_tie_break_rule"] = TIE_BREAK
        scaled.append(row)

    if sum(as_int(row, "pallets_per_sku") for row in scaled) != target_total:
        raise AssertionError("scaled inventory total mismatch")
    return scaled


def load_source(source_root: Path):
    root = source_root.resolve()
    paths = {key: root / name for key, name in SOURCE_FILES.items()}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical OPL input(s): {missing}")

    selected_columns, selected = read_csv(paths["selected"])
    slot_columns, slots = read_csv(paths["slots"])
    sku_columns, skus = read_csv(paths["sku"])

    if [row["selection_label"] for row in selected] != list(LAYOUTS):
        raise ValueError("selected_layouts.csv must contain L1-L4 in order")
    signatures = {row["selection_label"]: row["layout_signature"] for row in selected}
    if signatures != EXPECTED_SIGNATURES:
        raise ValueError("selected-layout signatures do not match the locked L1-L4 panel")
    capacities = {
        row["selection_label"]: as_int(row, "pallet_slot_capacity") for row in selected
    }
    if capacities != EXPECTED_CAPACITIES:
        raise ValueError(f"unexpected fixed-layout capacities: {capacities}")
    slot_counts = Counter(row["selection_label"] for row in slots)
    if dict(slot_counts) != EXPECTED_CAPACITIES:
        raise ValueError(f"slot row counts do not match capacities: {slot_counts}")
    if len(skus) != EXPECTED_SKUS:
        raise ValueError("canonical SKU catalog must contain 100 rows")
    if dict(Counter(row["sku_class"] for row in skus)) != EXPECTED_CLASSES:
        raise ValueError("canonical SKU class counts must be A20/B30/C50")
    if sum(as_int(row, "pallets_per_sku") for row in skus) != BASE_TOTAL:
        raise ValueError("canonical SKU catalog must total 790 pallets")

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return selected_columns, selected, slot_columns, slots, sku_columns, skus, hashes


def assign_all(
    slots: Sequence[dict[str, str]], skus: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    representatives: list[dict[str, Any]] = []
    reserves: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    for label in LAYOUTS:
        layout_slots = [row for row in slots if row["selection_label"] == label]
        rep, reserve, shortage = assignment.assign_layout(label, layout_slots, list(skus))
        representatives.extend(rep)
        reserves.extend(reserve)
        shortages.extend(shortage)
    return representatives, reserves, shortages


def projection(rows: Sequence[dict[str, Any]], columns: Sequence[str]):
    return [tuple(str(row.get(column, "")) for column in columns) for row in rows]


def slot_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["selection_label"]),
        str(row["row"]),
        str(row["col"]),
        str(row["level"]),
    )


def validate_case(
    *,
    target_total: int,
    selected: Sequence[dict[str, str]],
    scaled_skus: Sequence[dict[str, Any]],
    base_reps: Sequence[dict[str, Any]],
    representatives: Sequence[dict[str, Any]],
    reserves: Sequence[dict[str, Any]],
    shortages: Sequence[dict[str, Any]],
    base_regime_a: Sequence[dict[str, Any]],
    regime_a: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    capacities = {
        row["selection_label"]: as_int(row, "pallet_slot_capacity") for row in selected
    }
    rep_counts = Counter(row["selection_label"] for row in representatives)
    reserve_counts = Counter(row["selection_label"] for row in reserves)
    all_keys = [slot_key(row) for row in representatives] + [
        slot_key(row) for row in reserves
    ]
    expected_reserve = target_total - EXPECTED_SKUS
    checks = {
        "sku_count_exactly_100": len(scaled_skus) == EXPECTED_SKUS,
        "one_representative_pallet_per_sku": all(
            as_int(row, "representative_access_pallets") == 1 for row in scaled_skus
        ),
        "target_fits_all_layouts": all(target_total <= value for value in capacities.values()),
        "representative_total_per_layout_is_100": all(
            rep_counts[label] == EXPECTED_SKUS for label in LAYOUTS
        ),
        "reserve_total_per_layout_matches_target_minus_100": all(
            reserve_counts[label] == expected_reserve for label in LAYOUTS
        ),
        "no_assignment_shortage": not shortages,
        "no_duplicate_assigned_slots": len(all_keys) == len(set(all_keys)),
        "representative_locations_inventory_invariant": projection(
            representatives, REP_INVARIANT_COLUMNS
        )
        == projection(base_reps, REP_INVARIANT_COLUMNS),
        "scenario_A_inventory_invariant": rows_equivalent(
            regime_a, base_regime_a, regime_metrics.REGIME_A_COLUMNS
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "target_inventory_total": target_total,
        "representative_rows_total": len(representatives),
        "reserve_rows_total": len(reserves),
        "invariants": checks,
        "failed_invariants": failed,
        "assignment_shortages": list(shortages),
    }


def verify_low_against(
    root: Path,
    *,
    sku_columns: Sequence[str],
    scaled_skus: Sequence[dict[str, Any]],
    representatives: Sequence[dict[str, Any]],
    reserves: Sequence[dict[str, Any]],
    regime_a: Sequence[dict[str, Any]],
    regime_b: Sequence[dict[str, Any]],
    fragmentation: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    generated = {
        "sku": (scaled_skus, sku_columns, SOURCE_FILES["sku"]),
        "representatives": (
            representatives,
            assignment.REP_COLUMNS,
            LOW_VERIFY_FILES["representatives"],
        ),
        "reserves": (reserves, assignment.RESERVE_COLUMNS, LOW_VERIFY_FILES["reserves"]),
        "regime_a": (regime_a, regime_metrics.REGIME_A_COLUMNS, LOW_VERIFY_FILES["regime_a"]),
        "regime_b": (regime_b, regime_metrics.REGIME_B_COLUMNS, LOW_VERIFY_FILES["regime_b"]),
        "fragmentation": (
            fragmentation,
            regime_metrics.FRAGMENTATION_COLUMNS,
            LOW_VERIFY_FILES["fragmentation"],
        ),
    }
    results: dict[str, bool] = {}
    for name, (rows, columns, filename) in generated.items():
        canonical_path = root / filename
        if not canonical_path.is_file():
            raise FileNotFoundError(f"missing Low verification file: {canonical_path}")
        canonical_columns, canonical_rows = read_csv(canonical_path)
        compare_columns = [column for column in columns if column in canonical_columns]
        results[name] = rows_equivalent(rows, canonical_rows, compare_columns)
    failed = [name for name, passed in results.items() if not passed]
    return {"status": "passed" if not failed else "failed", "comparisons": results, "failed": failed}


def build_case(
    *,
    source_data_root: Path,
    output_root: Path,
    occupancy_label: str,
    target_total: int,
    verify_low_root: Path | None = None,
) -> dict[str, Any]:
    expected = NAMED_TOTALS.get(occupancy_label)
    if expected is not None and target_total != expected:
        raise ValueError(f"{occupancy_label} requires inventory total {expected}")

    source = source_data_root.resolve()
    output = output_root.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output root must be isolated from the canonical source directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output root: {output}")

    (
        _selected_columns,
        selected,
        _slot_columns,
        slots,
        sku_columns,
        base_skus,
        source_hashes,
    ) = load_source(source)
    if target_total > min(EXPECTED_CAPACITIES.values()):
        raise ValueError("target inventory exceeds the smallest fixed-layout capacity")

    scaled_skus = scale_inventory(base_skus, target_total)
    if any(
        sum(as_int(row, "reserve_pallets") for row in scaled_skus if row["sku_class"] == cls) == 0
        for cls in EXPECTED_CLASSES
    ):
        raise ValueError("target must leave at least one reserve pallet in every ABC class")

    base_reps, _, base_shortages = assign_all(slots, base_skus)
    if base_shortages:
        raise RuntimeError("canonical base assignment contains shortages")
    representatives, reserves, shortages = assign_all(slots, scaled_skus)

    base_regime_a = regime_metrics.compute_regime_a(base_reps)
    regime_a = regime_metrics.compute_regime_a(representatives)
    fragmentation = regime_metrics.compute_fragmentation_rows(reserves)
    regime_b = regime_metrics.compute_regime_b(
        list(selected), list(slots), representatives, reserves
    )

    validation = validate_case(
        target_total=target_total,
        selected=selected,
        scaled_skus=scaled_skus,
        base_reps=base_reps,
        representatives=representatives,
        reserves=reserves,
        shortages=shortages,
        base_regime_a=base_regime_a,
        regime_a=regime_a,
    )
    if validation["status"] != "passed":
        raise RuntimeError("occupancy validation failed: " + ", ".join(validation["failed_invariants"]))

    low_verification: dict[str, Any] = {"status": "not_requested"}
    if verify_low_root is not None:
        if occupancy_label != "low" or target_total != BASE_TOTAL:
            raise ValueError("--verify-low-against is valid only for Low=790")
        low_verification = verify_low_against(
            verify_low_root.resolve(),
            sku_columns=sku_columns,
            scaled_skus=scaled_skus,
            representatives=representatives,
            reserves=reserves,
            regime_a=regime_a,
            regime_b=regime_b,
            fragmentation=fragmentation,
        )
        if low_verification["status"] != "passed":
            raise RuntimeError("Low reproduction failed: " + ", ".join(low_verification["failed"]))

    current_hashes = {
        key: sha256_file(source / SOURCE_FILES[key]) for key in SOURCE_FILES
    }
    if current_hashes != source_hashes:
        raise RuntimeError("canonical source changed during occupancy computation")

    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "data"
    logs_root = output / "logs"
    scaled_columns = list(sku_columns) + [
        "base_pallets_per_sku",
        "occupancy_target_total",
        "occupancy_scaling_rule",
        "occupancy_tie_break_rule",
    ]
    write_csv(data_root / "sku_catalog_scaled.csv", scaled_columns, scaled_skus)
    write_csv(data_root / "representative_access_assignment.csv", assignment.REP_COLUMNS, representatives)
    write_csv(data_root / "reserve_pallet_assignment.csv", assignment.RESERVE_COLUMNS, reserves)
    write_csv(data_root / "regime_A_metrics.csv", regime_metrics.REGIME_A_COLUMNS, regime_a)
    write_csv(data_root / "regime_B_metrics.csv", regime_metrics.REGIME_B_COLUMNS, regime_b)
    write_csv(
        data_root / "reserve_fragmentation_summary.csv",
        regime_metrics.FRAGMENTATION_COLUMNS,
        fragmentation,
    )

    validation["low_reproduction"] = low_verification
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "module": "whl_experiments.build_operational_occupancy_sensitivity",
        "occupancy_label": occupancy_label,
        "target_inventory_total": target_total,
        "representative_total": EXPECTED_SKUS,
        "reserve_total": target_total - EXPECTED_SKUS,
        "source_data_root": str(source),
        "source_files": {
            SOURCE_FILES[key]: {"sha256": source_hashes[key]} for key in SOURCE_FILES
        },
        "scaling_rule": {
            "name": SCALING_RULE,
            "base_inventory_total": BASE_TOTAL,
            "tie_break_rule": TIE_BREAK,
        },
        "layout_capacities": EXPECTED_CAPACITIES,
        "utilization_by_layout": {
            label: target_total / capacity for label, capacity in EXPECTED_CAPACITIES.items()
        },
        "low_verification_status": low_verification["status"],
        "status": "complete",
    }
    (logs_root / "occupancy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one validated Section-7 occupancy-sensitivity case."
    )
    parser.add_argument(
        "--source-data-root",
        "--source-op-root",
        dest="source_data_root",
        type=Path,
        default=DEFAULT_SOURCE_DATA_ROOT,
        help="Canonical OPL data directory; defaults to data/operational_layer/paper_inputs.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--occupancy-label", choices=("low", "medium", "high", "custom"), required=True
    )
    parser.add_argument("--inventory-total", type=int, required=True)
    parser.add_argument("--verify-low-against", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_case(
            source_data_root=args.source_data_root,
            output_root=args.output_root,
            occupancy_label=args.occupancy_label,
            target_total=args.inventory_total,
            verify_low_root=args.verify_low_against,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "occupancy_label": result["occupancy_label"],
                "target_inventory_total": result["target_inventory_total"],
                "low_verification_status": result["low_verification_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
