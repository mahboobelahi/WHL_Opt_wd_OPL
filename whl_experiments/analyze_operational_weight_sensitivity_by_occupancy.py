"""Analyze Section-7 operational weight sensitivity across occupancy levels.

The module recomputes the existing reserve-access expression over already
validated fixed assignments. It does not regenerate structural layouts, slots,
SKU demand, assignments, or synthetic orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DATA_ROOT = (
    PROJECT_ROOT / "data" / "operational_layer" / "paper_inputs"
)

LAYOUTS = ("L1", "L2", "L3", "L4")
OCCUPANCY_TOTALS = {"low": 790, "medium": 2240, "high": 3584}
RESERVE_TOTALS = {"low": 690, "medium": 2140, "high": 3484}
EXPECTED_BASELINE_BEST = {"low": "L3", "medium": "L2", "high": "L2"}
ABS_TOLERANCE = 1e-9

WEIGHT_CASES = (
    ("B1", "B1_baseline", 0.10, 0.10),
    ("B2", "B2_higher_depth_penalty", 0.25, 0.10),
    ("B3", "B3_higher_level_penalty", 0.10, 0.25),
    ("B4", "B4_higher_depth_and_level_penalty", 0.25, 0.25),
)
SCENARIO_A_CASES = (
    "A2_lower_level_penalty",
    "A1_baseline",
    "A3_lower_depth_penalty",
    "A4_higher_depth_penalty",
    "A5_higher_level_penalty",
)

RESERVE_REL = Path("data") / "reserve_pallet_assignment.csv"
REP_REL = Path("data") / "representative_access_assignment.csv"
VALIDATION_REL = Path("logs") / "validation_summary.json"
CANONICAL_SUMMARY_REL = Path("sensitivity") / "lambda_sensitivity_summary.csv"
CANONICAL_BY_SEED_REL = Path("sensitivity") / "lambda_sensitivity_by_seed.csv"

REP_INVARIANT_COLUMNS = (
    "selection_label", "layout_signature", "sku_id", "sku_class",
    "global_sku_index", "demand_weight", "representative_access_pallets",
    "representative_slot_id", "row", "col", "level", "block_id", "block_size",
    "access_type", "effective_access_side", "effective_pick_face_row",
    "effective_pick_face_col", "effective_depth", "horizontal_access_distance",
    "normalized_distance", "normalized_depth", "normalized_level", "slot_cost",
)

SCENARIO_B_COLUMNS = (
    "occupancy_label", "inventory_total", "reserve_pallets_per_layout",
    "weight_case_id", "canonical_lambda_case_id", "omega_depth", "omega_level",
    "selection_label", "mean_normalized_horizontal_distance",
    "mean_normalized_depth", "mean_normalized_level", "reserve_access_cost",
    "rank_lower_is_better", "is_best_layout",
    "canonical_low_reserve_access_cost", "low_absolute_difference",
    "low_reproduction_tolerance", "low_reproduction_pass",
)
RANKING_COLUMNS = (
    "occupancy_label", "inventory_total", "weight_case_id",
    "canonical_lambda_case_id", "omega_depth", "omega_level",
    "ranking_lower_to_higher", "best_layout", "baseline_best_layout",
    "baseline_preference_preserved", "full_ranking_changed_vs_B1",
    "occupancy_robustness_classification",
)
SCENARIO_A_COLUMNS = (
    "canonical_lambda_case_id", "omega_depth", "omega_level",
    "representative_costs_lower_to_higher", "representative_best_layout",
    "synthetic_order_costs_lower_to_higher", "synthetic_order_best_layout",
    "representative_ranking_preserved_vs_A1",
    "synthetic_order_ranking_preserved_vs_A1",
    "applicable_occupancies", "applicability_basis",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column, "")) for column in columns})


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field!r}")
    return value


def mean(rows: Sequence[dict[str, str]], field: str) -> float:
    if not rows:
        raise ValueError(f"cannot calculate {field} over zero rows")
    return math.fsum(finite_float(row, field) for row in rows) / len(rows)


def load_case(label: str, root: Path) -> dict[str, Any]:
    root = root.resolve()
    reserve_path = root / RESERVE_REL
    rep_path = root / REP_REL
    validation_path = root / VALIDATION_REL
    for path in (reserve_path, rep_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: required source not found: {path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("failed_invariants"):
        raise RuntimeError(f"{label}: validated occupancy case did not pass")

    _, reserves = read_csv(reserve_path)
    _, representatives = read_csv(rep_path)
    if len(reserves) != RESERVE_TOTALS[label] * 4:
        raise RuntimeError(f"{label}: unexpected reserve row count")
    if len(representatives) != 400:
        raise RuntimeError(f"{label}: representative row count is not 400")
    for layout in LAYOUTS:
        layout_reserves = [row for row in reserves if row["selection_label"] == layout]
        if len(layout_reserves) != RESERVE_TOTALS[label]:
            raise RuntimeError(f"{label}/{layout}: reserve count mismatch")
        if any(
            row["reserve_assignment_status"].strip().lower() != "assigned"
            for row in layout_reserves
        ):
            raise RuntimeError(f"{label}/{layout}: incomplete reserve assignment")
    return {
        "label": label,
        "root": root,
        "reserves": reserves,
        "representatives": representatives,
        "validation_sha256": sha256_file(validation_path),
        "reserve_sha256": sha256_file(reserve_path),
        "representative_sha256": sha256_file(rep_path),
    }


def rep_projection(rows: Sequence[dict[str, Any]]):
    ordered = sorted(
        rows,
        key=lambda row: (
            row["selection_label"],
            int(float(row["global_sku_index"])),
        ),
    )
    return [tuple(str(row.get(column, "")) for column in REP_INVARIANT_COLUMNS) for row in ordered]


def validate_rep_invariance(cases: Sequence[dict[str, Any]]) -> bool:
    baseline = rep_projection(cases[0]["representatives"])
    result = all(rep_projection(case["representatives"]) == baseline for case in cases[1:])
    if not result:
        raise RuntimeError("representative assignment is not invariant across occupancies")
    return result


def load_canonical(root: Path) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / CANONICAL_SUMMARY_REL
    by_seed_path = root / CANONICAL_BY_SEED_REL
    if not summary_path.is_file() or not by_seed_path.is_file():
        raise FileNotFoundError("canonical lambda-sensitivity files are missing")
    _, summary = read_csv(summary_path)
    _, by_seed = read_csv(by_seed_path)
    scenario_a = [row for row in summary if row["regime"] == "A"]
    scenario_b = [row for row in summary if row["regime"] == "B"]
    if len(scenario_a) != 20 or len(scenario_b) != 16 or len(by_seed) != 600:
        raise RuntimeError("canonical sensitivity coverage is incomplete")
    if {row["lambda_case_id"] for row in scenario_a} != set(SCENARIO_A_CASES):
        raise RuntimeError("unexpected Scenario-A sensitivity cases")
    return {
        "root": root,
        "summary": summary,
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "by_seed": by_seed,
        "summary_path": summary_path,
        "by_seed_path": by_seed_path,
    }


def scenario_a_summary(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = {
        case_id: [row for row in rows if row["lambda_case_id"] == case_id]
        for case_id in SCENARIO_A_CASES
    }
    rep_orders = {
        case_id: [
            row["selection_label"]
            for row in sorted(
                case_rows,
                key=lambda row: finite_float(row, "representative_lambda_cost"),
            )
        ]
        for case_id, case_rows in grouped.items()
    }
    order_orders = {
        case_id: [
            row["selection_label"]
            for row in sorted(
                case_rows,
                key=lambda row: finite_float(
                    row, "mean_order_lambda_access_effort_sum"
                ),
            )
        ]
        for case_id, case_rows in grouped.items()
    }
    baseline_rep = rep_orders["A1_baseline"]
    baseline_order = order_orders["A1_baseline"]
    output = []
    for case_id in SCENARIO_A_CASES:
        case_rows = grouped[case_id]
        rep_sorted = sorted(
            case_rows, key=lambda row: finite_float(row, "representative_lambda_cost")
        )
        order_sorted = sorted(
            case_rows,
            key=lambda row: finite_float(row, "mean_order_lambda_access_effort_sum"),
        )
        output.append(
            {
                "canonical_lambda_case_id": case_id,
                "omega_depth": finite_float(case_rows[0], "lambda_depth"),
                "omega_level": finite_float(case_rows[0], "lambda_level"),
                "representative_costs_lower_to_higher": " < ".join(
                    f"{row['selection_label']}={finite_float(row, 'representative_lambda_cost'):.12g}"
                    for row in rep_sorted
                ),
                "representative_best_layout": rep_sorted[0]["selection_label"],
                "synthetic_order_costs_lower_to_higher": " < ".join(
                    f"{row['selection_label']}={finite_float(row, 'mean_order_lambda_access_effort_sum'):.12g}"
                    for row in order_sorted
                ),
                "synthetic_order_best_layout": order_sorted[0]["selection_label"],
                "representative_ranking_preserved_vs_A1": rep_orders[case_id] == baseline_rep,
                "synthetic_order_ranking_preserved_vs_A1": order_orders[case_id] == baseline_order,
                "applicable_occupancies": "Low / Medium / High",
                "applicability_basis": (
                    "representative assignments and demand weights are invariant"
                ),
            }
        )
    return output


def scenario_b_analysis(
    cases: Sequence[dict[str, Any]],
    canonical_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical_map = {
        (row["lambda_case_id"], row["selection_label"]): row for row in canonical_rows
    }
    output_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    low_checks: list[dict[str, Any]] = []

    for case in cases:
        components = {}
        for layout in LAYOUTS:
            rows = [row for row in case["reserves"] if row["selection_label"] == layout]
            components[layout] = {
                "horizontal": mean(rows, "normalized_distance"),
                "depth": mean(rows, "normalized_depth"),
                "level": mean(rows, "normalized_level"),
            }

        rankings: dict[str, list[str]] = {}
        best: dict[str, str] = {}
        for case_id, canonical_case_id, omega_depth, omega_level in WEIGHT_CASES:
            rows_for_case = []
            for layout in LAYOUTS:
                comp = components[layout]
                cost = (
                    comp["horizontal"]
                    + omega_depth * comp["depth"]
                    + omega_level * comp["level"]
                )
                row: dict[str, Any] = {
                    "occupancy_label": case["label"],
                    "inventory_total": OCCUPANCY_TOTALS[case["label"]],
                    "reserve_pallets_per_layout": RESERVE_TOTALS[case["label"]],
                    "weight_case_id": case_id,
                    "canonical_lambda_case_id": canonical_case_id,
                    "omega_depth": omega_depth,
                    "omega_level": omega_level,
                    "selection_label": layout,
                    "mean_normalized_horizontal_distance": comp["horizontal"],
                    "mean_normalized_depth": comp["depth"],
                    "mean_normalized_level": comp["level"],
                    "reserve_access_cost": cost,
                    "canonical_low_reserve_access_cost": "",
                    "low_absolute_difference": "",
                    "low_reproduction_tolerance": "",
                    "low_reproduction_pass": "",
                }
                if case["label"] == "low":
                    canonical = canonical_map[(canonical_case_id, layout)]
                    canonical_cost = finite_float(canonical, "reserve_lambda_access_cost")
                    difference = abs(cost - canonical_cost)
                    row.update(
                        {
                            "canonical_low_reserve_access_cost": canonical_cost,
                            "low_absolute_difference": difference,
                            "low_reproduction_tolerance": ABS_TOLERANCE,
                            "_canonical_rank": int(
                                canonical["rank_by_reserve_lambda_access_cost"]
                            ),
                        }
                    )
                rows_for_case.append(row)

            ordered = sorted(
                rows_for_case,
                key=lambda row: (float(row["reserve_access_cost"]), row["selection_label"]),
            )
            rankings[case_id] = [row["selection_label"] for row in ordered]
            best[case_id] = ordered[0]["selection_label"]
            for rank, row in enumerate(ordered, start=1):
                row["rank_lower_is_better"] = rank
                row["is_best_layout"] = rank == 1
                if case["label"] == "low":
                    canonical_rank = row.pop("_canonical_rank")
                    passed = (
                        float(row["low_absolute_difference"]) <= ABS_TOLERANCE
                        and rank == canonical_rank
                    )
                    row["low_reproduction_pass"] = passed
                    low_checks.append(
                        {
                            "case": row["weight_case_id"],
                            "layout": row["selection_label"],
                            "absolute_difference": row["low_absolute_difference"],
                            "rank_match": rank == canonical_rank,
                            "passed": passed,
                        }
                    )
            output_rows.extend(rows_for_case)

        baseline_best = best["B1"]
        if baseline_best != EXPECTED_BASELINE_BEST[case["label"]]:
            raise RuntimeError(
                f"{case['label']}: baseline best {baseline_best} does not match locked evidence"
            )
        baseline_ranking = rankings["B1"]
        preserved_count = sum(best[case_id] == baseline_best for case_id, *_ in WEIGHT_CASES)
        robustness = (
            "ranking robust"
            if preserved_count == len(WEIGHT_CASES)
            else "partially robust"
            if preserved_count >= 2
            else "not robust"
        )
        for case_id, canonical_case_id, omega_depth, omega_level in WEIGHT_CASES:
            ranking_rows.append(
                {
                    "occupancy_label": case["label"],
                    "inventory_total": OCCUPANCY_TOTALS[case["label"]],
                    "weight_case_id": case_id,
                    "canonical_lambda_case_id": canonical_case_id,
                    "omega_depth": omega_depth,
                    "omega_level": omega_level,
                    "ranking_lower_to_higher": " < ".join(rankings[case_id]),
                    "best_layout": best[case_id],
                    "baseline_best_layout": baseline_best,
                    "baseline_preference_preserved": best[case_id] == baseline_best,
                    "full_ranking_changed_vs_B1": rankings[case_id] != baseline_ranking,
                    "occupancy_robustness_classification": robustness,
                }
            )

    if not low_checks or not all(bool(row["passed"]) for row in low_checks):
        raise RuntimeError("Low Scenario-B sensitivity does not reproduce canonical evidence")
    return (
        output_rows,
        ranking_rows,
        {
            "status": "passed",
            "comparison_count": len(low_checks),
            "maximum_absolute_difference": max(
                float(row["absolute_difference"]) for row in low_checks
            ),
            "all_ranks_match": all(bool(row["rank_match"]) for row in low_checks),
        },
    )


def run_analysis(
    *,
    low_root: Path,
    medium_root: Path,
    high_root: Path,
    canonical_data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    roots = [low_root.resolve(), medium_root.resolve(), high_root.resolve()]
    output = output_root.resolve()
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("weight-sensitivity output must be isolated from case roots")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")

    cases = [
        load_case("low", roots[0]),
        load_case("medium", roots[1]),
        load_case("high", roots[2]),
    ]
    rep_invariant = validate_rep_invariance(cases)
    canonical = load_canonical(canonical_data_root)

    scenario_a = scenario_a_summary(canonical["scenario_a"])
    scenario_b, rankings, low_reproduction = scenario_b_analysis(
        cases, canonical["scenario_b"]
    )
    if len(scenario_a) != 5 or len(scenario_b) != 48 or len(rankings) != 12:
        raise RuntimeError("unexpected weight-sensitivity output row count")

    output.mkdir(parents=True, exist_ok=False)
    files = {
        "scenario_B_weight_sensitivity_by_occupancy.csv": (
            SCENARIO_B_COLUMNS,
            scenario_b,
        ),
        "scenario_B_weight_sensitivity_ranking.csv": (RANKING_COLUMNS, rankings),
        "scenario_A_existing_sensitivity_summary.csv": (SCENARIO_A_COLUMNS, scenario_a),
    }
    output_records = {}
    for filename, (columns, rows) in files.items():
        path = output / filename
        write_csv(path, columns, rows)
        output_records[filename] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    robustness = {
        label: next(
            row["occupancy_robustness_classification"]
            for row in rankings
            if row["occupancy_label"] == label
        )
        for label in OCCUPANCY_TOTALS
    }
    manifest = {
        "schema_version": 1,
        "module": "whl_experiments.analyze_operational_weight_sensitivity_by_occupancy",
        "scientific_scope": (
            "read-only recomputation of the existing normalized reserve-access "
            "expression over validated fixed assignments"
        ),
        "formula": (
            "mean(normalized_distance) + omega_depth*mean(normalized_depth) "
            "+ omega_level*mean(normalized_level)"
        ),
        "weight_cases": [
            {
                "weight_case_id": case_id,
                "canonical_lambda_case_id": canonical_case_id,
                "omega_depth": omega_depth,
                "omega_level": omega_level,
            }
            for case_id, canonical_case_id, omega_depth, omega_level in WEIGHT_CASES
        ],
        "representative_assignments_invariant": rep_invariant,
        "low_reproduction": low_reproduction,
        "ranking_robustness": robustness,
        "canonical_sources": {
            str(CANONICAL_SUMMARY_REL): sha256_file(canonical["summary_path"]),
            str(CANONICAL_BY_SEED_REL): sha256_file(canonical["by_seed_path"]),
        },
        "case_sources": {
            case["label"]: {
                "reserve_sha256": case["reserve_sha256"],
                "representative_sha256": case["representative_sha256"],
                "validation_sha256": case["validation_sha256"],
            }
            for case in cases
        },
        "outputs": output_records,
        "status": "complete",
    }
    (output / "weight_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Section-7 weight sensitivity across occupancy levels."
    )
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--medium-root", type=Path, required=True)
    parser.add_argument("--high-root", type=Path, required=True)
    parser.add_argument(
        "--canonical-data-root",
        "--canonical-op-root",
        "--scenario-a-op-root",
        dest="canonical_data_root",
        type=Path,
        default=DEFAULT_CANONICAL_DATA_ROOT,
        help="Canonical OPL data directory; defaults to data/operational_layer/paper_inputs.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_analysis(
            low_root=args.low_root,
            medium_root=args.medium_root,
            high_root=args.high_root,
            canonical_data_root=args.canonical_data_root,
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "low_reproduction": result["low_reproduction"],
                "ranking_robustness": result["ranking_robustness"],
                "outputs": result["outputs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
