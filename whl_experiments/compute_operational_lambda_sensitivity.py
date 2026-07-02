"""Compute lambda-sensitivity diagnostics from fixed operational-layer outputs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = PROJECT_ROOT / "data" / "operational_layer"
DATA_ROOT = OP_ROOT / "paper_inputs"
SENSITIVITY_ROOT = DATA_ROOT / "sensitivity"
LOG_ROOT = OP_ROOT / "paper_outputs" / "logs"
DOC_ROOT = PROJECT_ROOT / "docs"

SELECTED_LAYOUTS_CSV = DATA_ROOT / "selected_layouts.csv"
SKU_CATALOG_CSV = DATA_ROOT / "sku_catalog.csv"
SLOT_METRICS_CSV = DATA_ROOT / "slot_metrics_by_layout.csv"
REP_CSV = DATA_ROOT / "representative_access_assignment.csv"
RESERVE_CSV = DATA_ROOT / "reserve_pallet_assignment.csv"
SYNTHETIC_ORDERS_CSV = DATA_ROOT / "synthetic_orders.csv"
REGIME_A_CSV = DATA_ROOT / "regime_A_metrics.csv"
REGIME_B_CSV = DATA_ROOT / "regime_B_metrics.csv"
ORDER_EFFORT_SUMMARY_CSV = DATA_ROOT / "order_effort_summary.csv"
CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"
M8_SUMMARY_JSON = LOG_ROOT / "m8_order_proxy_summary.json"

SUMMARY_CSV = SENSITIVITY_ROOT / "lambda_sensitivity_summary.csv"
BY_SEED_CSV = SENSITIVITY_ROOT / "lambda_sensitivity_by_seed.csv"
SUMMARY_JSON = LOG_ROOT / "m9_lambda_sensitivity_summary.json"
REPORT_MD = DOC_ROOT / "104_operational_layer_lambda_sensitivity.md"

LAYOUTS = ["L1", "L2", "L3", "L4"]
REGIME_A_CASES = [
    ("A2_lower_level_penalty", 1.0, 0.5),
    ("A1_baseline", 1.0, 1.0),
    ("A3_lower_depth_penalty", 0.5, 1.0),
    ("A4_higher_depth_penalty", 2.0, 1.0),
    ("A5_higher_level_penalty", 1.0, 2.0),
]
REGIME_B_CASES = [
    ("B1_baseline", 0.1, 0.1),
    ("B2_higher_depth_penalty", 0.25, 0.1),
    ("B3_higher_level_penalty", 0.1, 0.25),
    ("B4_higher_depth_and_level_penalty", 0.25, 0.25),
]
BASELINE_A = (1.0, 1.0)
BASELINE_B = (0.1, 0.1)
WORKLOAD_SEED_COUNT = 30
ORDERS_PER_SEED = 1000
BASELINE_TOL = 1e-9

SUMMARY_COLUMNS = [
    "regime",
    "lambda_case_id",
    "lambda_depth",
    "lambda_level",
    "is_baseline_lambda",
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "representative_lambda_cost",
    "mean_order_lambda_access_effort_sum",
    "std_order_lambda_access_effort_sum_across_seeds",
    "ci95_order_lambda_access_effort_sum_halfwidth",
    "mean_line_lambda_access_effort",
    "reserve_lambda_access_cost",
    "interior_deep_slot_share",
    "upper_level_slot_share",
    "reserve_deep_pallet_share",
    "reserve_upper_level_pallet_share",
    "reserve_fragmentation_proxy",
    "same_block_reserve_share",
    "same_side_reserve_share",
    "rank_by_representative_lambda_cost",
    "rank_by_mean_order_lambda_access_effort_sum",
    "rank_by_mean_line_lambda_access_effort",
    "rank_by_reserve_lambda_access_cost",
    "rank_by_interior_deep_slot_share",
    "rank_by_reserve_fragmentation_proxy",
    "baseline_rank_reference",
    "ranking_changed_vs_baseline",
    "sensitivity_status",
    "sensitivity_warning",
]

BY_SEED_COLUMNS = [
    "regime",
    "lambda_case_id",
    "lambda_depth",
    "lambda_level",
    "selection_label",
    "workload_seed_id",
    "rng_seed",
    "orders_count",
    "order_lines_count",
    "mean_order_lambda_access_effort_sum",
    "mean_order_lambda_access_effort_mean",
    "mean_line_lambda_access_effort",
    "seed_status",
    "seed_warning",
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
    return float(row[field])


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def cost(row: dict[str, str], lambda_depth: float, lambda_level: float) -> float:
    return (
        to_float(row, "normalized_distance")
        + lambda_depth * to_float(row, "normalized_depth")
        + lambda_level * to_float(row, "normalized_level")
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_inputs() -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    required = [
        SELECTED_LAYOUTS_CSV,
        SKU_CATALOG_CSV,
        SLOT_METRICS_CSV,
        REP_CSV,
        RESERVE_CSV,
        SYNTHETIC_ORDERS_CSV,
        REGIME_A_CSV,
        REGIME_B_CSV,
        ORDER_EFFORT_SUMMARY_CSV,
        CONFIG_JSON,
        M8_SUMMARY_JSON,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required M9 input(s): " + ", ".join(as_posix(path) for path in missing))

    config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    selected = read_csv(SELECTED_LAYOUTS_CSV)
    reps = read_csv(REP_CSV)
    reserves = read_csv(RESERVE_CSV)
    orders = read_csv(SYNTHETIC_ORDERS_CSV)
    regime_a = read_csv(REGIME_A_CSV)
    regime_b = read_csv(REGIME_B_CSV)
    order_summary = read_csv(ORDER_EFFORT_SUMMARY_CSV)
    m8_summary = json.loads(M8_SUMMARY_JSON.read_text(encoding="utf-8"))

    if [row["selection_label"] for row in selected] != LAYOUTS:
        raise RuntimeError("selected_layouts.csv labels are not exactly L1-L4")
    if len({row["order_id"] for row in orders}) != 30000:
        raise RuntimeError("synthetic_orders.csv does not contain 30,000 fixed orders")
    if m8_summary.get("ready_for_milestone_9") is not True:
        raise RuntimeError("M8 summary does not report ready_for_milestone_9=true")
    for label in LAYOUTS:
        if sum(1 for row in reps if row["selection_label"] == label) != 100:
            raise RuntimeError(f"{label}: representative rows are not 100")
        if sum(1 for row in reserves if row["selection_label"] == label) != 690:
            raise RuntimeError(f"{label}: reserve rows are not 690")
    return config, selected, reps, reserves, orders, regime_a, regime_b, order_summary


def compute_a_seed_rows(
    reps: list[dict[str, str]],
    orders: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    reps_by_layout_sku = {(row["selection_label"], row["sku_id"]): row for row in reps}
    orders_by_seed_order: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in orders:
        orders_by_seed_order[(int(row["workload_seed_id"]), row["order_id"])].append(row)

    seed_rows: list[dict[str, Any]] = []
    summary_by_case_layout: dict[tuple[str, str], dict[str, float]] = {}
    for case_id, lambda_depth, lambda_level in REGIME_A_CASES:
        for label in LAYOUTS:
            per_seed_order_sum_means: list[float] = []
            per_seed_line_means: list[float] = []
            for seed_id in range(1, WORKLOAD_SEED_COUNT + 1):
                order_sums: list[float] = []
                order_means: list[float] = []
                line_costs: list[float] = []
                order_lines_count = 0
                orders_count = 0
                for (workload_seed_id, _order_id), lines in orders_by_seed_order.items():
                    if workload_seed_id != seed_id:
                        continue
                    orders_count += 1
                    costs = [
                        cost(reps_by_layout_sku[(label, line["sku_id"])], lambda_depth, lambda_level)
                        for line in lines
                    ]
                    order_lines_count += len(costs)
                    order_sums.append(sum(costs))
                    order_means.append(mean(costs))
                    line_costs.extend(costs)
                seed_mean_order_sum = mean(order_sums)
                seed_mean_line = mean(line_costs)
                per_seed_order_sum_means.append(seed_mean_order_sum)
                per_seed_line_means.append(seed_mean_line)
                seed_rows.append(
                    {
                        "regime": "A",
                        "lambda_case_id": case_id,
                        "lambda_depth": lambda_depth,
                        "lambda_level": lambda_level,
                        "selection_label": label,
                        "workload_seed_id": seed_id,
                        "rng_seed": 20260617 + seed_id,
                        "orders_count": orders_count,
                        "order_lines_count": order_lines_count,
                        "mean_order_lambda_access_effort_sum": seed_mean_order_sum,
                        "mean_order_lambda_access_effort_mean": mean(order_means),
                        "mean_line_lambda_access_effort": seed_mean_line,
                        "seed_status": "ok",
                        "seed_warning": "",
                    }
                )
            std = statistics.stdev(per_seed_order_sum_means) if len(per_seed_order_sum_means) > 1 else 0.0
            summary_by_case_layout[(case_id, label)] = {
                "mean_order_lambda_access_effort_sum": mean(per_seed_order_sum_means),
                "std_order_lambda_access_effort_sum_across_seeds": std,
                "ci95_order_lambda_access_effort_sum_halfwidth": 1.96 * std / math.sqrt(WORKLOAD_SEED_COUNT),
                "mean_line_lambda_access_effort": mean(per_seed_line_means),
            }
    return seed_rows, summary_by_case_layout


def rank_map(rows: list[dict[str, Any]], metric: str, reverse: bool = False) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)
    return {row["selection_label"]: index + 1 for index, row in enumerate(ordered)}


def compute_summary_rows(
    selected: list[dict[str, str]],
    reps: list[dict[str, str]],
    reserves: list[dict[str, str]],
    regime_b: list[dict[str, str]],
    a_seed_summary: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    selected_by_label = {row["selection_label"]: row for row in selected}
    b_by_label = {row["selection_label"]: row for row in regime_b}
    reps_by_label = {label: [row for row in reps if row["selection_label"] == label] for label in LAYOUTS}
    reserves_by_label = {label: [row for row in reserves if row["selection_label"] == label] for label in LAYOUTS}
    rows: list[dict[str, Any]] = []

    for case_id, lambda_depth, lambda_level in REGIME_A_CASES:
        case_rows: list[dict[str, Any]] = []
        for label in LAYOUTS:
            selected_row = selected_by_label[label]
            rep_cost = sum(
                to_float(row, "demand_weight") * cost(row, lambda_depth, lambda_level)
                for row in reps_by_label[label]
            )
            seed_stats = a_seed_summary[(case_id, label)]
            case_rows.append(
                {
                    "regime": "A",
                    "lambda_case_id": case_id,
                    "lambda_depth": lambda_depth,
                    "lambda_level": lambda_level,
                    "is_baseline_lambda": (lambda_depth, lambda_level) == BASELINE_A,
                    "selection_label": label,
                    "selection_type": selected_row["selection_type"],
                    "layout_signature": selected_row["layout_signature"],
                    "seed": selected_row["seed"],
                    "rank": selected_row["rank"],
                    "representative_lambda_cost": rep_cost,
                    **seed_stats,
                    "reserve_lambda_access_cost": "",
                    "interior_deep_slot_share": "",
                    "upper_level_slot_share": "",
                    "reserve_deep_pallet_share": "",
                    "reserve_upper_level_pallet_share": "",
                    "reserve_fragmentation_proxy": "",
                    "same_block_reserve_share": "",
                    "same_side_reserve_share": "",
                    "sensitivity_status": "ok",
                    "sensitivity_warning": "",
                }
            )
        rep_ranks = rank_map(case_rows, "representative_lambda_cost")
        order_ranks = rank_map(case_rows, "mean_order_lambda_access_effort_sum")
        line_ranks = rank_map(case_rows, "mean_line_lambda_access_effort")
        baseline_order_ranks = order_ranks if (lambda_depth, lambda_level) == BASELINE_A else None
        for row in case_rows:
            row["rank_by_representative_lambda_cost"] = rep_ranks[row["selection_label"]]
            row["rank_by_mean_order_lambda_access_effort_sum"] = order_ranks[row["selection_label"]]
            row["rank_by_mean_line_lambda_access_effort"] = line_ranks[row["selection_label"]]
            row["rank_by_reserve_lambda_access_cost"] = ""
            row["rank_by_interior_deep_slot_share"] = ""
            row["rank_by_reserve_fragmentation_proxy"] = ""
            row["baseline_rank_reference"] = ""
            row["ranking_changed_vs_baseline"] = ""
        rows.extend(case_rows)

    baseline_a_order = {
        row["selection_label"]: int(row["rank_by_mean_order_lambda_access_effort_sum"])
        for row in rows
        if row["regime"] == "A" and row["is_baseline_lambda"] is True
    }
    for row in rows:
        if row["regime"] == "A":
            base = baseline_a_order[row["selection_label"]]
            row["baseline_rank_reference"] = base
            row["ranking_changed_vs_baseline"] = int(row["rank_by_mean_order_lambda_access_effort_sum"]) != base

    for case_id, lambda_depth, lambda_level in REGIME_B_CASES:
        case_rows = []
        for label in LAYOUTS:
            selected_row = selected_by_label[label]
            b = b_by_label[label]
            reserve_cost = mean([cost(row, lambda_depth, lambda_level) for row in reserves_by_label[label]])
            case_rows.append(
                {
                    "regime": "B",
                    "lambda_case_id": case_id,
                    "lambda_depth": lambda_depth,
                    "lambda_level": lambda_level,
                    "is_baseline_lambda": (lambda_depth, lambda_level) == BASELINE_B,
                    "selection_label": label,
                    "selection_type": selected_row["selection_type"],
                    "layout_signature": selected_row["layout_signature"],
                    "seed": selected_row["seed"],
                    "rank": selected_row["rank"],
                    "representative_lambda_cost": "",
                    "mean_order_lambda_access_effort_sum": "",
                    "std_order_lambda_access_effort_sum_across_seeds": "",
                    "ci95_order_lambda_access_effort_sum_halfwidth": "",
                    "mean_line_lambda_access_effort": "",
                    "reserve_lambda_access_cost": reserve_cost,
                    "interior_deep_slot_share": b["interior_deep_slot_share"],
                    "upper_level_slot_share": b["upper_level_slot_share"],
                    "reserve_deep_pallet_share": b["reserve_deep_pallet_share"],
                    "reserve_upper_level_pallet_share": b["reserve_upper_level_pallet_share"],
                    "reserve_fragmentation_proxy": b["reserve_fragmentation_proxy"],
                    "same_block_reserve_share": b["same_block_reserve_share"],
                    "same_side_reserve_share": b["same_side_reserve_share"],
                    "sensitivity_status": "ok",
                    "sensitivity_warning": "",
                }
            )
        reserve_ranks = rank_map(case_rows, "reserve_lambda_access_cost")
        deep_ranks = rank_map(case_rows, "interior_deep_slot_share", reverse=True)
        frag_ranks = rank_map(case_rows, "reserve_fragmentation_proxy")
        for row in case_rows:
            row["rank_by_representative_lambda_cost"] = ""
            row["rank_by_mean_order_lambda_access_effort_sum"] = ""
            row["rank_by_mean_line_lambda_access_effort"] = ""
            row["rank_by_reserve_lambda_access_cost"] = reserve_ranks[row["selection_label"]]
            row["rank_by_interior_deep_slot_share"] = deep_ranks[row["selection_label"]]
            row["rank_by_reserve_fragmentation_proxy"] = frag_ranks[row["selection_label"]]
            row["baseline_rank_reference"] = ""
            row["ranking_changed_vs_baseline"] = ""
        rows.extend(case_rows)

    baseline_b_reserve = {
        row["selection_label"]: int(row["rank_by_reserve_lambda_access_cost"])
        for row in rows
        if row["regime"] == "B" and row["is_baseline_lambda"] is True
    }
    for row in rows:
        if row["regime"] == "B":
            base = baseline_b_reserve[row["selection_label"]]
            row["baseline_rank_reference"] = base
            row["ranking_changed_vs_baseline"] = int(row["rank_by_reserve_lambda_access_cost"]) != base
    return rows


def ranking(rows: list[dict[str, Any]], metric: str, regime: str, case_id: str | None = None, reverse: bool = False) -> list[dict[str, Any]]:
    scoped = [row for row in rows if row["regime"] == regime and (case_id is None or row["lambda_case_id"] == case_id)]
    scoped = [row for row in scoped if row.get(metric, "") != ""]
    ordered = sorted(scoped, key=lambda row: float(row[metric]), reverse=reverse)
    return [
        {"rank": index + 1, "selection_label": row["selection_label"], metric: float(row[metric])}
        for index, row in enumerate(ordered)
    ]


def validate(
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    regime_a: list[dict[str, str]],
    regime_b: list[dict[str, str]],
    order_summary: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    validation: dict[str, Any] = {
        "summary_rows_expected": 36,
        "summary_rows_written": len(summary_rows),
        "seed_rows_expected": 600,
        "seed_rows_written": len(seed_rows),
        "regime_A_case_count": len({row["lambda_case_id"] for row in summary_rows if row["regime"] == "A"}),
        "regime_B_case_count": len({row["lambda_case_id"] for row in summary_rows if row["regime"] == "B"}),
        "all_layouts_each_case": True,
        "all_lambda_metrics_finite": True,
        "ranking_columns_complete": True,
        "baseline_A_representative_matches_M7": True,
        "baseline_A_order_matches_M8": True,
        "baseline_B_reserve_matches_M7": True,
        "no_routing_or_simulation_fields_created": not any(
            "route" in col.lower()
            or "sequence" in col.lower()
            or "travel" in col.lower()
            or "picker" in col.lower()
            or "forklift" in col.lower()
            or "time" in col.lower()
            for col in SUMMARY_COLUMNS + BY_SEED_COLUMNS
        ),
    }
    if len(summary_rows) != 36:
        warnings.append(f"summary row count is {len(summary_rows)}, expected 36")
    if len(seed_rows) != 600:
        warnings.append(f"seed row count is {len(seed_rows)}, expected 600")
    for regime in ("A", "B"):
        for case_id in {row["lambda_case_id"] for row in summary_rows if row["regime"] == regime}:
            labels = sorted(row["selection_label"] for row in summary_rows if row["regime"] == regime and row["lambda_case_id"] == case_id)
            if labels != LAYOUTS:
                validation["all_layouts_each_case"] = False
                warnings.append(f"{regime}/{case_id} missing L1-L4 labels")
    finite_fields = [
        "representative_lambda_cost",
        "mean_order_lambda_access_effort_sum",
        "std_order_lambda_access_effort_sum_across_seeds",
        "ci95_order_lambda_access_effort_sum_halfwidth",
        "mean_line_lambda_access_effort",
        "reserve_lambda_access_cost",
    ]
    for row in summary_rows:
        for field in finite_fields:
            value = row.get(field, "")
            if value == "":
                continue
            if not math.isfinite(float(value)):
                validation["all_lambda_metrics_finite"] = False
                warnings.append(f"{row['regime']}/{row['lambda_case_id']}/{row['selection_label']} non-finite {field}")
    for row in summary_rows:
        if row["regime"] == "A":
            required = [
                "rank_by_representative_lambda_cost",
                "rank_by_mean_order_lambda_access_effort_sum",
                "rank_by_mean_line_lambda_access_effort",
                "baseline_rank_reference",
                "ranking_changed_vs_baseline",
            ]
        else:
            required = [
                "rank_by_reserve_lambda_access_cost",
                "rank_by_interior_deep_slot_share",
                "rank_by_reserve_fragmentation_proxy",
                "baseline_rank_reference",
                "ranking_changed_vs_baseline",
            ]
        if any(row.get(field, "") == "" for field in required):
            validation["ranking_columns_complete"] = False
            warnings.append(f"{row['regime']}/{row['lambda_case_id']}/{row['selection_label']} missing ranking field")

    a_by_label = {row["selection_label"]: row for row in regime_a}
    b_by_label = {row["selection_label"]: row for row in regime_b}
    order_by_label = {row["selection_label"]: row for row in order_summary}
    for row in summary_rows:
        if row["regime"] == "A" and row["lambda_case_id"] == "A1_baseline":
            label = row["selection_label"]
            if abs(float(row["representative_lambda_cost"]) - float(a_by_label[label]["weighted_access_cost"])) > BASELINE_TOL:
                validation["baseline_A_representative_matches_M7"] = False
                warnings.append(f"{label}: baseline representative lambda cost does not match M7")
            if abs(float(row["mean_order_lambda_access_effort_sum"]) - float(order_by_label[label]["mean_order_access_effort_sum_mean"])) > BASELINE_TOL:
                validation["baseline_A_order_matches_M8"] = False
                warnings.append(f"{label}: baseline order lambda effort does not match M8")
        if row["regime"] == "B" and row["lambda_case_id"] == "B1_baseline":
            label = row["selection_label"]
            if abs(float(row["reserve_lambda_access_cost"]) - float(b_by_label[label]["low_depth_level_weight_access_cost"])) > BASELINE_TOL:
                validation["baseline_B_reserve_matches_M7"] = False
                warnings.append(f"{label}: baseline reserve lambda access cost does not match M7")
    if not validation["no_routing_or_simulation_fields_created"]:
        warnings.append("routing/sequencing/simulation fields were created")
    return validation, warnings


def stability(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a_best_by_case = {}
    a_worst_by_case = {}
    a_top_competitive = {}
    a_l2_worst = {}
    for case_id, _lambda_depth, _lambda_level in REGIME_A_CASES:
        order_rank = ranking(summary_rows, "mean_order_lambda_access_effort_sum", "A", case_id)
        a_best_by_case[case_id] = order_rank[0]["selection_label"]
        a_worst_by_case[case_id] = order_rank[-1]["selection_label"]
        positions = {item["selection_label"]: item["rank"] for item in order_rank}
        a_top_competitive[case_id] = positions["L1"] <= 2 and positions["L3"] <= 2
        a_l2_worst[case_id] = positions["L2"] >= 3

    b_best_reserve = {}
    for case_id, _lambda_depth, _lambda_level in REGIME_B_CASES:
        b_best_reserve[case_id] = ranking(summary_rows, "reserve_lambda_access_cost", "B", case_id)[0]["selection_label"]
    deep_rank = ranking(summary_rows, "interior_deep_slot_share", "B", "B1_baseline", reverse=True)
    frag_rank = ranking(summary_rows, "reserve_fragmentation_proxy", "B", "B1_baseline")
    return {
        "regime_A_best_layouts_by_case": a_best_by_case,
        "regime_A_worst_layouts_by_case": a_worst_by_case,
        "regime_A_L1_L3_top_competitive_check": a_top_competitive,
        "regime_A_L2_worst_or_near_worst_check": a_l2_worst,
        "regime_B_best_reserve_access_layouts_by_case": b_best_reserve,
        "regime_B_L2_deep_capacity_stability": deep_rank[0]["selection_label"] == "L2",
        "regime_B_L2_fragmentation_stability": frag_rank[0]["selection_label"] == "L2",
    }


def update_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["lambda_sensitivity"] = {
        "supplement_only": True,
        "regime_A_lambda_cases": [[ld, ll] for _cid, ld, ll in REGIME_A_CASES],
        "regime_B_lambda_cases": [[ld, ll] for _cid, ld, ll in REGIME_B_CASES],
        "uses_fixed_assignments": True,
        "uses_fixed_synthetic_orders": True,
        "does_not_modify_main_results": True,
        "outputs": {
            "lambda_sensitivity_summary_csv": as_posix(SUMMARY_CSV),
            "lambda_sensitivity_by_seed_csv": as_posix(BY_SEED_CSV),
            "m9_summary_json": as_posix(SUMMARY_JSON),
        },
    }
    completed = list(updated.get("milestones_completed", []))
    if "M9" not in completed:
        completed.append("M9")
    updated["milestones_completed"] = completed
    return updated


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    baseline_rows = [
        row for row in summary_rows if row["lambda_case_id"] in {"A1_baseline", "B1_baseline"}
    ]
    report = f"""# Operational-layer lambda sensitivity

## Inputs

{chr(10).join(f"- `{path}`" for path in summary['input_files'].values())}

## Lambda settings

- Scenario A cases: `{summary['regime_A_lambda_cases']}`
- Scenario B cases: `{summary['regime_B_lambda_cases']}`

## Baseline rankings

`{json.dumps(summary['baseline_rankings'], sort_keys=True)}`

## Baseline rows

{table(baseline_rows, ['regime', 'lambda_case_id', 'selection_label', 'representative_lambda_cost', 'mean_order_lambda_access_effort_sum', 'reserve_lambda_access_cost', 'rank_by_mean_order_lambda_access_effort_sum', 'rank_by_reserve_lambda_access_cost'])}

## Ranking stability

`{json.dumps(summary['ranking_stability'], sort_keys=True)}`

## Validation summary

- Summary rows written: `{summary['summary_rows_written']}`
- Seed rows written: `{summary['seed_rows_written']}`
- Baseline A representative matches M7: `{summary['validation']['baseline_A_representative_matches_M7']}`
- Baseline A order matches M8: `{summary['validation']['baseline_A_order_matches_M8']}`
- Baseline B reserve matches M7: `{summary['validation']['baseline_B_reserve_matches_M7']}`
- Ready for Milestone 10: `{summary['ready_for_milestone_10']}`

## Output files

- Sensitivity summary: `{summary['lambda_sensitivity_summary_csv']}`
- Sensitivity by seed: `{summary['lambda_sensitivity_by_seed_csv']}`
- Summary JSON: `{as_posix(SUMMARY_JSON)}`
- Report: `{as_posix(REPORT_MD)}`

## Warnings

{warnings}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    config, selected, reps, reserves, orders, regime_a, regime_b, order_summary = load_inputs()
    seed_rows, a_seed_summary = compute_a_seed_rows(reps, orders)
    summary_rows = compute_summary_rows(selected, reps, reserves, regime_b, a_seed_summary)
    validation, warnings = validate(summary_rows, seed_rows, regime_a, regime_b, order_summary)
    if warnings:
        raise RuntimeError("M9 validation failed before writing outputs: " + "; ".join(warnings))

    serial_summary_rows = [{key: fmt(value) for key, value in row.items()} for row in summary_rows]
    serial_seed_rows = [{key: fmt(value) for key, value in row.items()} for row in seed_rows]
    write_csv(SUMMARY_CSV, serial_summary_rows, SUMMARY_COLUMNS)
    write_csv(BY_SEED_CSV, serial_seed_rows, BY_SEED_COLUMNS)
    updated_config = update_config(config)
    CONFIG_JSON.write_text(json.dumps(updated_config, indent=2) + "\n", encoding="utf-8")

    validation, warnings = validate(summary_rows, seed_rows, regime_a, regime_b, order_summary)
    stability_summary = stability(summary_rows)
    baseline_rankings = {
        "regime_A_by_representative_lambda_cost": ranking(summary_rows, "representative_lambda_cost", "A", "A1_baseline"),
        "regime_A_by_order_lambda_effort": ranking(summary_rows, "mean_order_lambda_access_effort_sum", "A", "A1_baseline"),
        "regime_B_by_reserve_lambda_access_cost": ranking(summary_rows, "reserve_lambda_access_cost", "B", "B1_baseline"),
        "regime_B_by_interior_deep_slot_share": ranking(summary_rows, "interior_deep_slot_share", "B", "B1_baseline", reverse=True),
        "regime_B_by_reserve_fragmentation_proxy": ranking(summary_rows, "reserve_fragmentation_proxy", "B", "B1_baseline"),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "selected_layouts_csv": as_posix(SELECTED_LAYOUTS_CSV),
            "sku_catalog_csv": as_posix(SKU_CATALOG_CSV),
            "slot_metrics_by_layout_csv": as_posix(SLOT_METRICS_CSV),
            "representative_access_assignment_csv": as_posix(REP_CSV),
            "reserve_pallet_assignment_csv": as_posix(RESERVE_CSV),
            "synthetic_orders_csv": as_posix(SYNTHETIC_ORDERS_CSV),
            "regime_A_metrics_csv": as_posix(REGIME_A_CSV),
            "regime_B_metrics_csv": as_posix(REGIME_B_CSV),
            "order_effort_summary_csv": as_posix(ORDER_EFFORT_SUMMARY_CSV),
        },
        "lambda_sensitivity_summary_csv": as_posix(SUMMARY_CSV),
        "lambda_sensitivity_by_seed_csv": as_posix(BY_SEED_CSV),
        "regime_A_lambda_cases": [[ld, ll] for _cid, ld, ll in REGIME_A_CASES],
        "regime_B_lambda_cases": [[ld, ll] for _cid, ld, ll in REGIME_B_CASES],
        "summary_rows_expected": 36,
        "summary_rows_written": len(summary_rows),
        "seed_rows_expected": 600,
        "seed_rows_written": len(seed_rows),
        "baseline_rankings": baseline_rankings,
        "ranking_stability": stability_summary,
        "validation": validation,
        "warnings": warnings,
        "blockers_or_warnings": warnings,
        "ready_for_milestone_10": not warnings,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, serial_summary_rows)
    if warnings:
        raise RuntimeError("M9 validation failed after writing outputs: " + "; ".join(warnings))

    print("Milestone 9 lambda sensitivity complete.")
    print(f"lambda_sensitivity_summary.csv: {rel_posix(SUMMARY_CSV)}")
    print(f"lambda_sensitivity_by_seed.csv: {rel_posix(BY_SEED_CSV)}")
    print(f"summary JSON: {rel_posix(SUMMARY_JSON)}")
    print(f"Markdown report: {rel_posix(REPORT_MD)}")
    print(f"summary rows written: {len(summary_rows)}")
    print(f"seed rows written: {len(seed_rows)}")
    print(f"Regime A ranking stability: {json.dumps(stability_summary['regime_A_best_layouts_by_case'], sort_keys=True)}; worst={json.dumps(stability_summary['regime_A_worst_layouts_by_case'], sort_keys=True)}")
    print(f"Regime B ranking stability: reserve_access_best={json.dumps(stability_summary['regime_B_best_reserve_access_layouts_by_case'], sort_keys=True)}, L2_deep={stability_summary['regime_B_L2_deep_capacity_stability']}, L2_fragmentation={stability_summary['regime_B_L2_fragmentation_stability']}")
    print(
        "baseline matching result: "
        + json.dumps(
            {
                "A_representative_matches_M7": validation["baseline_A_representative_matches_M7"],
                "A_order_matches_M8": validation["baseline_A_order_matches_M8"],
                "B_reserve_matches_M7": validation["baseline_B_reserve_matches_M7"],
            },
            sort_keys=True,
        )
    )
    print("warnings or blockers: none")
    print(f"ready_for_milestone_10: {summary['ready_for_milestone_10']}")


if __name__ == "__main__":
    main()
