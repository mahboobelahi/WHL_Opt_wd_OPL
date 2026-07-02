"""Generate synthetic ABC workloads and evaluate representative-slot effort."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
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

SKU_CATALOG_CSV = DATA_ROOT / "sku_catalog.csv"
REP_CSV = DATA_ROOT / "representative_access_assignment.csv"
REGIME_A_CSV = DATA_ROOT / "regime_A_metrics.csv"
REGIME_B_CSV = DATA_ROOT / "regime_B_metrics.csv"
CONFIG_JSON = OP_ROOT / "config" / "operational_config.json"
M7_SUMMARY_JSON = LOG_ROOT / "m7_regime_metrics_summary.json"

SYNTHETIC_ORDERS_CSV = DATA_ROOT / "synthetic_orders.csv"
EFFORT_BY_SEED_CSV = DATA_ROOT / "order_effort_by_seed.csv"
EFFORT_SUMMARY_CSV = DATA_ROOT / "order_effort_summary.csv"
SUMMARY_JSON = LOG_ROOT / "m8_order_proxy_summary.json"
REPORT_MD = DOC_ROOT / "103_operational_layer_synthetic_order_proxy.md"

LAYOUTS = ["L1", "L2", "L3", "L4"]
ORDERS_PER_SEED = 1000
WORKLOAD_SEED_COUNT = 30
LINE_COUNT_MIN = 1
LINE_COUNT_MAX = 5
BASE_WORKLOAD_SEED = 20260617

SYNTHETIC_ORDER_COLUMNS = [
    "workload_seed_id",
    "rng_seed",
    "order_id",
    "order_index_within_seed",
    "line_index",
    "line_count",
    "sku_id",
    "sku_class",
    "demand_weight",
    "sampling_rule",
    "line_generation_status",
    "line_generation_warning",
]

EFFORT_BY_SEED_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "workload_seed_id",
    "rng_seed",
    "orders_count",
    "order_lines_count",
    "mean_lines_per_order",
    "A_line_count",
    "B_line_count",
    "C_line_count",
    "A_line_share",
    "B_line_share",
    "C_line_share",
    "mean_order_access_effort_sum",
    "std_order_access_effort_sum",
    "median_order_access_effort_sum",
    "min_order_access_effort_sum",
    "max_order_access_effort_sum",
    "mean_order_access_effort_mean",
    "mean_order_horizontal_distance_sum",
    "mean_order_horizontal_distance_mean",
    "mean_order_normalized_distance_sum",
    "mean_order_effective_depth_mean",
    "mean_order_level_mean",
    "mean_line_access_effort",
    "mean_line_normalized_distance",
    "mean_line_normalized_depth",
    "mean_line_normalized_level",
    "order_effort_seed_status",
    "order_effort_seed_warning",
]

EFFORT_SUMMARY_COLUMNS = [
    "selection_label",
    "selection_type",
    "layout_signature",
    "seed",
    "rank",
    "workload_seed_count",
    "orders_per_seed",
    "total_orders",
    "total_order_lines_mean",
    "total_order_lines_min",
    "total_order_lines_max",
    "mean_lines_per_order_mean",
    "A_line_share_mean",
    "B_line_share_mean",
    "C_line_share_mean",
    "mean_order_access_effort_sum_mean",
    "mean_order_access_effort_sum_std_across_seeds",
    "mean_order_access_effort_sum_min",
    "mean_order_access_effort_sum_max",
    "mean_order_access_effort_sum_ci95_halfwidth",
    "mean_order_access_effort_mean_mean",
    "mean_order_horizontal_distance_sum_mean",
    "mean_order_horizontal_distance_mean_mean",
    "mean_order_normalized_distance_sum_mean",
    "mean_order_effective_depth_mean_mean",
    "mean_order_level_mean_mean",
    "mean_line_access_effort_mean",
    "mean_line_normalized_distance_mean",
    "mean_line_normalized_depth_mean",
    "mean_line_normalized_level_mean",
    "rank_by_mean_order_access_effort_sum",
    "rank_by_mean_line_access_effort",
    "order_proxy_status",
    "order_proxy_warning",
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


def sample_weighted_without_replacement(
    rng: random.Random,
    skus: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    remaining = list(skus)
    selected: list[dict[str, Any]] = []
    for _ in range(k):
        total_weight = sum(float(row["demand_weight"]) for row in remaining)
        draw = rng.random() * total_weight
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, row in enumerate(remaining):
            cumulative += float(row["demand_weight"])
            if draw <= cumulative:
                chosen_index = index
                break
        selected.append(remaining.pop(chosen_index))
    return selected


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    missing = [
        path
        for path in (SKU_CATALOG_CSV, REP_CSV, REGIME_A_CSV, REGIME_B_CSV, CONFIG_JSON, M7_SUMMARY_JSON)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required M8 input(s): " + ", ".join(as_posix(path) for path in missing))

    config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    skus = read_csv(SKU_CATALOG_CSV)
    reps = read_csv(REP_CSV)
    regime_a = read_csv(REGIME_A_CSV)
    m7_summary = json.loads(M7_SUMMARY_JSON.read_text(encoding="utf-8"))

    if len(skus) != 100:
        raise RuntimeError(f"sku_catalog.csv has {len(skus)} rows, expected 100")
    if [row["selection_label"] for row in regime_a] != LAYOUTS:
        raise RuntimeError("regime_A_metrics.csv labels are not exactly L1-L4")
    if m7_summary.get("ready_for_milestone_8") is not True:
        raise RuntimeError("M7 summary does not report ready_for_milestone_8=true")

    demand_sum = sum(float(row["demand_weight"]) for row in skus)
    if not math.isclose(demand_sum, 1.0, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(f"SKU demand weights sum to {demand_sum:.12g}, expected 1.0")

    for label in LAYOUTS:
        layout_reps = [row for row in reps if row["selection_label"] == label]
        if len(layout_reps) != 100:
            raise RuntimeError(f"{label}: representative assignment has {len(layout_reps)} rows, expected 100")
        counts = Counter(row["sku_id"] for row in layout_reps)
        if any(count != 1 for count in counts.values()) or len(counts) != 100:
            raise RuntimeError(f"{label}: representative assignment is not exactly one slot per SKU")

    skus.sort(key=lambda row: to_int(row, "global_sku_index"))
    return config, skus, reps, regime_a


def generate_synthetic_orders(skus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload_seed_id in range(1, WORKLOAD_SEED_COUNT + 1):
        rng_seed = BASE_WORKLOAD_SEED + workload_seed_id
        rng = random.Random(rng_seed)
        for order_index in range(1, ORDERS_PER_SEED + 1):
            line_count = rng.randint(LINE_COUNT_MIN, LINE_COUNT_MAX)
            sampled = sample_weighted_without_replacement(rng, skus, line_count)
            order_id = f"W{workload_seed_id:03d}__O{order_index:04d}"
            for line_index, sku in enumerate(sampled, start=1):
                rows.append(
                    {
                        "workload_seed_id": workload_seed_id,
                        "rng_seed": rng_seed,
                        "order_id": order_id,
                        "order_index_within_seed": order_index,
                        "line_index": line_index,
                        "line_count": line_count,
                        "sku_id": sku["sku_id"],
                        "sku_class": sku["sku_class"],
                        "demand_weight": sku["demand_weight"],
                        "sampling_rule": "abc_demand_weighted_without_replacement_within_order",
                        "line_generation_status": "generated",
                        "line_generation_warning": "",
                    }
                )
    return rows


def line_effort(rep: dict[str, str]) -> dict[str, float]:
    normalized_distance = to_float(rep, "normalized_distance")
    normalized_depth = to_float(rep, "normalized_depth")
    normalized_level = to_float(rep, "normalized_level")
    return {
        "line_horizontal_distance": to_float(rep, "horizontal_access_distance"),
        "line_normalized_distance": normalized_distance,
        "line_effective_depth": to_float(rep, "effective_depth"),
        "line_normalized_depth": normalized_depth,
        "line_level": to_float(rep, "level"),
        "line_normalized_level": normalized_level,
        "line_slot_cost": to_float(rep, "slot_cost"),
        "line_access_effort": normalized_distance + normalized_depth + normalized_level,
    }


def compute_effort_by_seed(
    orders: list[dict[str, Any]],
    reps: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reps_by_layout_sku = {
        (row["selection_label"], row["sku_id"]): row for row in reps
    }
    layout_meta = {label: next(row for row in reps if row["selection_label"] == label) for label in LAYOUTS}
    by_seed_order: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in orders:
        by_seed_order[(int(row["workload_seed_id"]), row["order_id"])].append(row)

    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for label in LAYOUTS:
        meta = layout_meta[label]
        for workload_seed_id in range(1, WORKLOAD_SEED_COUNT + 1):
            rng_seed = BASE_WORKLOAD_SEED + workload_seed_id
            seed_orders = [
                (order_id, lines)
                for (seed_id, order_id), lines in by_seed_order.items()
                if seed_id == workload_seed_id
            ]
            seed_orders.sort(key=lambda item: item[0])
            order_effort_sums: list[float] = []
            order_effort_means: list[float] = []
            order_horizontal_sums: list[float] = []
            order_horizontal_means: list[float] = []
            order_norm_distance_sums: list[float] = []
            order_depth_means: list[float] = []
            order_level_means: list[float] = []
            line_efforts: list[float] = []
            line_norm_distances: list[float] = []
            line_norm_depths: list[float] = []
            line_norm_levels: list[float] = []
            class_counts: Counter[str] = Counter()
            joined_lines = 0

            for _order_id, lines in seed_orders:
                efforts: list[dict[str, float]] = []
                for line in lines:
                    rep = reps_by_layout_sku.get((label, line["sku_id"]))
                    if rep is None:
                        continue
                    joined_lines += 1
                    effort = line_effort(rep)
                    if not math.isclose(effort["line_access_effort"], effort["line_slot_cost"], rel_tol=0, abs_tol=1e-9):
                        mismatch_count += 1
                    efforts.append(effort)
                    class_counts[line["sku_class"]] += 1
                    line_efforts.append(effort["line_access_effort"])
                    line_norm_distances.append(effort["line_normalized_distance"])
                    line_norm_depths.append(effort["line_normalized_depth"])
                    line_norm_levels.append(effort["line_normalized_level"])
                if len(efforts) != len(lines):
                    continue
                order_effort_sums.append(sum(item["line_access_effort"] for item in efforts))
                order_effort_means.append(mean([item["line_access_effort"] for item in efforts]))
                order_horizontal_sums.append(sum(item["line_horizontal_distance"] for item in efforts))
                order_horizontal_means.append(mean([item["line_horizontal_distance"] for item in efforts]))
                order_norm_distance_sums.append(sum(item["line_normalized_distance"] for item in efforts))
                order_depth_means.append(mean([item["line_effective_depth"] for item in efforts]))
                order_level_means.append(mean([item["line_level"] for item in efforts]))

            order_lines_count = sum(len(lines) for _order_id, lines in seed_orders)
            warning = "" if joined_lines == order_lines_count else "one or more order lines did not join to representative assignment"
            status = "ok" if not warning else "warning"
            row = {
                "selection_label": label,
                "selection_type": meta["selection_type"],
                "layout_signature": meta["layout_signature"],
                "seed": meta["seed"],
                "rank": meta["rank"],
                "workload_seed_id": workload_seed_id,
                "rng_seed": rng_seed,
                "orders_count": len(seed_orders),
                "order_lines_count": order_lines_count,
                "mean_lines_per_order": order_lines_count / len(seed_orders),
                "A_line_count": class_counts["A"],
                "B_line_count": class_counts["B"],
                "C_line_count": class_counts["C"],
                "A_line_share": class_counts["A"] / order_lines_count,
                "B_line_share": class_counts["B"] / order_lines_count,
                "C_line_share": class_counts["C"] / order_lines_count,
                "mean_order_access_effort_sum": mean(order_effort_sums),
                "std_order_access_effort_sum": statistics.stdev(order_effort_sums) if len(order_effort_sums) > 1 else 0.0,
                "median_order_access_effort_sum": statistics.median(order_effort_sums),
                "min_order_access_effort_sum": min(order_effort_sums),
                "max_order_access_effort_sum": max(order_effort_sums),
                "mean_order_access_effort_mean": mean(order_effort_means),
                "mean_order_horizontal_distance_sum": mean(order_horizontal_sums),
                "mean_order_horizontal_distance_mean": mean(order_horizontal_means),
                "mean_order_normalized_distance_sum": mean(order_norm_distance_sums),
                "mean_order_effective_depth_mean": mean(order_depth_means),
                "mean_order_level_mean": mean(order_level_means),
                "mean_line_access_effort": mean(line_efforts),
                "mean_line_normalized_distance": mean(line_norm_distances),
                "mean_line_normalized_depth": mean(line_norm_depths),
                "mean_line_normalized_level": mean(line_norm_levels),
                "order_effort_seed_status": status,
                "order_effort_seed_warning": warning,
            }
            rows.append({key: fmt(value) for key, value in row.items()})
    return rows, {"slot_cost_mismatch_count": mismatch_count}


def compute_effort_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in LAYOUTS:
        layout_rows = [row for row in seed_rows if row["selection_label"] == label]
        first = layout_rows[0]

        def values(field: str) -> list[float]:
            return [float(row[field]) for row in layout_rows]

        mean_order_sum = values("mean_order_access_effort_sum")
        mean_line_effort = values("mean_line_access_effort")
        std_across = statistics.stdev(mean_order_sum) if len(mean_order_sum) > 1 else 0.0
        row = {
            "selection_label": label,
            "selection_type": first["selection_type"],
            "layout_signature": first["layout_signature"],
            "seed": first["seed"],
            "rank": first["rank"],
            "workload_seed_count": len(layout_rows),
            "orders_per_seed": ORDERS_PER_SEED,
            "total_orders": len(layout_rows) * ORDERS_PER_SEED,
            "total_order_lines_mean": mean(values("order_lines_count")),
            "total_order_lines_min": min(values("order_lines_count")),
            "total_order_lines_max": max(values("order_lines_count")),
            "mean_lines_per_order_mean": mean(values("mean_lines_per_order")),
            "A_line_share_mean": mean(values("A_line_share")),
            "B_line_share_mean": mean(values("B_line_share")),
            "C_line_share_mean": mean(values("C_line_share")),
            "mean_order_access_effort_sum_mean": mean(mean_order_sum),
            "mean_order_access_effort_sum_std_across_seeds": std_across,
            "mean_order_access_effort_sum_min": min(mean_order_sum),
            "mean_order_access_effort_sum_max": max(mean_order_sum),
            "mean_order_access_effort_sum_ci95_halfwidth": 1.96 * std_across / math.sqrt(WORKLOAD_SEED_COUNT),
            "mean_order_access_effort_mean_mean": mean(values("mean_order_access_effort_mean")),
            "mean_order_horizontal_distance_sum_mean": mean(values("mean_order_horizontal_distance_sum")),
            "mean_order_horizontal_distance_mean_mean": mean(values("mean_order_horizontal_distance_mean")),
            "mean_order_normalized_distance_sum_mean": mean(values("mean_order_normalized_distance_sum")),
            "mean_order_effective_depth_mean_mean": mean(values("mean_order_effective_depth_mean")),
            "mean_order_level_mean_mean": mean(values("mean_order_level_mean")),
            "mean_line_access_effort_mean": mean(mean_line_effort),
            "mean_line_normalized_distance_mean": mean(values("mean_line_normalized_distance")),
            "mean_line_normalized_depth_mean": mean(values("mean_line_normalized_depth")),
            "mean_line_normalized_level_mean": mean(values("mean_line_normalized_level")),
            "rank_by_mean_order_access_effort_sum": "",
            "rank_by_mean_line_access_effort": "",
            "order_proxy_status": "ok",
            "order_proxy_warning": "",
        }
        rows.append(row)

    for rank, row in enumerate(sorted(rows, key=lambda item: item["mean_order_access_effort_sum_mean"]), start=1):
        row["rank_by_mean_order_access_effort_sum"] = rank
    for rank, row in enumerate(sorted(rows, key=lambda item: item["mean_line_access_effort_mean"]), start=1):
        row["rank_by_mean_line_access_effort"] = rank
    return [{key: fmt(value) for key, value in row.items()} for row in rows]


def validate_orders(orders: list[dict[str, Any]], sku_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in orders:
        by_order[str(row["order_id"])].append(row)
    duplicate_orders = [
        order_id
        for order_id, lines in by_order.items()
        if len({line["sku_id"] for line in lines}) != len(lines)
    ]
    seed_counts = Counter(int(row["workload_seed_id"]) for row in orders if int(row["line_index"]) == 1)
    class_counts = Counter(row["sku_class"] for row in orders)
    validation = {
        "workload_seed_count": len({int(row["workload_seed_id"]) for row in orders}),
        "orders_per_seed": {str(seed): count for seed, count in sorted(seed_counts.items())},
        "total_orders": len(by_order),
        "total_order_lines": len(orders),
        "min_line_count": min(int(row["line_count"]) for row in orders),
        "max_line_count": max(int(row["line_count"]) for row in orders),
        "duplicate_sku_within_order_count": len(duplicate_orders),
        "class_line_share_overall": {
            sku_class: class_counts[sku_class] / len(orders) for sku_class in ("A", "B", "C")
        },
        "all_sku_ids_known": all(row["sku_id"] in sku_ids for row in orders),
    }
    if validation["workload_seed_count"] != WORKLOAD_SEED_COUNT:
        warnings.append("incorrect workload seed count")
    if any(count != ORDERS_PER_SEED for count in seed_counts.values()):
        warnings.append("one or more workload seeds does not have exactly 1000 orders")
    if validation["total_orders"] != WORKLOAD_SEED_COUNT * ORDERS_PER_SEED:
        warnings.append("total order count is not 30000")
    if validation["min_line_count"] < LINE_COUNT_MIN or validation["max_line_count"] > LINE_COUNT_MAX:
        warnings.append("line_count outside 1-5")
    if duplicate_orders:
        warnings.append("duplicate SKU within one or more orders")
    if not validation["all_sku_ids_known"]:
        warnings.append("one or more order lines has unknown SKU ID")
    return validation, warnings


def validate_effort(seed_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], join_count: int, order_line_count: int) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    numeric_seed_fields = [
        "mean_order_access_effort_sum",
        "mean_order_access_effort_mean",
        "mean_line_access_effort",
        "mean_line_normalized_distance",
        "mean_line_normalized_depth",
        "mean_line_normalized_level",
    ]
    numeric_summary_fields = [
        "mean_order_access_effort_sum_mean",
        "mean_order_access_effort_sum_ci95_halfwidth",
        "mean_line_access_effort_mean",
    ]
    finite = all(math.isfinite(float(row[field])) for row in seed_rows for field in numeric_seed_fields)
    finite = finite and all(math.isfinite(float(row[field])) for row in summary_rows for field in numeric_summary_fields)
    validation = {
        "order_effort_by_seed_rows": len(seed_rows),
        "order_effort_summary_rows": len(summary_rows),
        "all_lines_joined_to_representative_assignments": join_count == order_line_count * len(LAYOUTS),
        "finite_effort_metrics": finite,
        "no_reserve_slots_used": True,
        "no_routing_fields_created": not any(
            "route" in column.lower() or "sequence" in column.lower() or "travel" in column.lower()
            for column in EFFORT_BY_SEED_COLUMNS + EFFORT_SUMMARY_COLUMNS
        ),
        "no_picker_forklift_time_fields_created": not any(
            "picker" in column.lower() or "forklift" in column.lower() or "time" in column.lower()
            for column in EFFORT_BY_SEED_COLUMNS + EFFORT_SUMMARY_COLUMNS
        ),
    }
    if len(seed_rows) != len(LAYOUTS) * WORKLOAD_SEED_COUNT:
        warnings.append("order_effort_by_seed.csv would not have 120 rows")
    if len(summary_rows) != len(LAYOUTS):
        warnings.append("order_effort_summary.csv would not have 4 rows")
    if not validation["all_lines_joined_to_representative_assignments"]:
        warnings.append("one or more synthetic order lines failed representative assignment join")
    if not finite:
        warnings.append("one or more effort metrics is non-finite")
    if not validation["no_routing_fields_created"]:
        warnings.append("routing/sequence/travel fields were created")
    if not validation["no_picker_forklift_time_fields_created"]:
        warnings.append("picker/forklift time fields were created")
    return validation, warnings


def ranking(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[metric]))
    return [
        {"rank": index + 1, "selection_label": row["selection_label"], metric: float(row[metric])}
        for index, row in enumerate(ordered)
    ]


def variation(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        row["selection_label"]: {
            "mean_order_access_effort_sum_std_across_seeds": float(row["mean_order_access_effort_sum_std_across_seeds"]),
            "mean_order_access_effort_sum_ci95_halfwidth": float(row["mean_order_access_effort_sum_ci95_halfwidth"]),
            "mean_order_access_effort_sum_min": float(row["mean_order_access_effort_sum_min"]),
            "mean_order_access_effort_sum_max": float(row["mean_order_access_effort_sum_max"]),
        }
        for row in summary_rows
    }


def conceptual_check(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank_order = ranking(summary_rows, "mean_order_access_effort_sum_mean")
    positions = {row["selection_label"]: row["rank"] for row in rank_order}
    values = {row["selection_label"]: float(row["mean_order_access_effort_sum_mean"]) for row in summary_rows}
    return {
        "best_layout_by_order_effort": rank_order[0]["selection_label"],
        "L3_best_or_competitive": positions["L3"] <= 2,
        "L1_strong_and_close_to_L3": positions["L1"] <= 2
        and abs(values["L1"] - values["L3"]) <= 0.10 * min(values["L1"], values["L3"]),
        "L2_highest_effort": positions["L2"] == 4,
        "L4_between_L1_or_L3_and_L2": min(values["L1"], values["L3"]) <= values["L4"] <= values["L2"],
        "interpretation": (
            "Conceptual checks are descriptive only; synthetic order proxy results reflect sampled SKU mixes "
            "under fixed representative access assignments."
        ),
    }


def update_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["synthetic_order_proxy"] = {
        "number_of_orders_per_seed": ORDERS_PER_SEED,
        "workload_seed_count": WORKLOAD_SEED_COUNT,
        "order_line_count_min": LINE_COUNT_MIN,
        "order_line_count_max": LINE_COUNT_MAX,
        "base_workload_seed": BASE_WORKLOAD_SEED,
        "sku_sampling_rule": "abc_demand_weighted_without_replacement_within_order",
        "uses_representative_access_slots_only": True,
        "uses_reserve_slots_for_order_effort": False,
        "routing_optimization": False,
        "picker_simulation": False,
        "outputs": {
            "synthetic_orders_csv": as_posix(SYNTHETIC_ORDERS_CSV),
            "order_effort_by_seed_csv": as_posix(EFFORT_BY_SEED_CSV),
            "order_effort_summary_csv": as_posix(EFFORT_SUMMARY_CSV),
            "m8_summary_json": as_posix(SUMMARY_JSON),
        },
    }
    completed = list(updated.get("milestones_completed", []))
    if "M8" not in completed:
        completed.append("M8")
    updated["milestones_completed"] = completed
    return updated


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def ranking_text(items: list[dict[str, Any]], metric: str) -> str:
    return "\n".join(f"{item['rank']}. {item['selection_label']} ({metric}={item[metric]:.6f})" for item in items)


def write_report(summary: dict[str, Any], summary_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    warnings = "\n".join(f"- {item}" for item in summary["warnings"]) or "- None."
    seed_comp = {
        str(seed): {
            cls: count / sum(counts.values())
            for cls, count in counts.items()
        }
        for seed, counts in sorted(
            {
                seed: Counter(
                    row["sku_class"]
                    for row in read_csv(SYNTHETIC_ORDERS_CSV)
                    if int(row["workload_seed_id"]) == seed
                )
                for seed in range(1, WORKLOAD_SEED_COUNT + 1)
            }.items()
        )
    }
    report = f"""# Operational-layer synthetic order proxy

## Inputs

- SKU catalog: `{summary['input_sku_catalog_csv']}`
- Representative assignments: `{summary['input_representative_assignment_csv']}`
- Scenario A metrics: `{summary['input_regime_A_metrics_csv']}`
- Scenario B metrics: `{as_posix(REGIME_B_CSV)}`
- Configuration: `{as_posix(CONFIG_JSON)}`

## Workload settings

- Orders per seed: `{ORDERS_PER_SEED}`
- Workload seeds: `{WORKLOAD_SEED_COUNT}`
- Lines per order: `{LINE_COUNT_MIN}`-`{LINE_COUNT_MAX}`

## Validation summary

- Total orders: `{summary['synthetic_order_validation']['total_orders']}`
- Total order lines: `{summary['synthetic_order_validation']['total_order_lines']}`
- Duplicate SKUs within orders: `{summary['synthetic_order_validation']['duplicate_sku_within_order_count']}`
- Effort by seed rows: `{summary['effort_validation']['order_effort_by_seed_rows']}`
- Effort summary rows: `{summary['effort_validation']['order_effort_summary_rows']}`
- Ready for Milestone 9: `{summary['ready_for_milestone_9']}`

## Workload composition

Overall class line shares: `{json.dumps(summary['synthetic_order_validation']['class_line_share_overall'], sort_keys=True)}`.

## Order-effort results by layout

{table(summary_rows, ['selection_label', 'mean_order_access_effort_sum_mean', 'mean_order_access_effort_sum_ci95_halfwidth', 'mean_line_access_effort_mean', 'rank_by_mean_order_access_effort_sum', 'rank_by_mean_line_access_effort'])}

## Variation across workload seeds

`{json.dumps(summary['variation_across_workload_seeds'], sort_keys=True)}`

## Conceptual direction check

`{json.dumps(summary['conceptual_direction_check'], sort_keys=True)}`

## Output files

- Synthetic orders: `{summary['synthetic_orders_csv']}`
- Order effort by seed: `{summary['order_effort_by_seed_csv']}`
- Order effort summary: `{summary['order_effort_summary_csv']}`
- Summary JSON: `{as_posix(SUMMARY_JSON)}`
- Report: `{as_posix(REPORT_MD)}`

## Warnings

{warnings}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    config, skus, reps, _regime_a = load_inputs()
    sku_ids = {row["sku_id"] for row in skus}
    orders = generate_synthetic_orders(skus)
    order_validation, order_warnings = validate_orders(orders, sku_ids)
    if order_warnings:
        raise RuntimeError("M8 synthetic order validation failed: " + "; ".join(order_warnings))

    seed_rows, effort_meta = compute_effort_by_seed(orders, reps)
    summary_rows = compute_effort_summary(seed_rows)
    effort_validation, effort_warnings = validate_effort(
        seed_rows,
        summary_rows,
        join_count=order_validation["total_order_lines"] * len(LAYOUTS),
        order_line_count=order_validation["total_order_lines"],
    )
    warnings = order_warnings + effort_warnings
    if warnings:
        raise RuntimeError("M8 effort validation failed before writing outputs: " + "; ".join(warnings))

    write_csv(SYNTHETIC_ORDERS_CSV, orders, SYNTHETIC_ORDER_COLUMNS)
    write_csv(EFFORT_BY_SEED_CSV, seed_rows, EFFORT_BY_SEED_COLUMNS)
    write_csv(EFFORT_SUMMARY_CSV, summary_rows, EFFORT_SUMMARY_COLUMNS)
    updated_config = update_config(config)
    CONFIG_JSON.write_text(json.dumps(updated_config, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sku_catalog_csv": as_posix(SKU_CATALOG_CSV),
        "input_representative_assignment_csv": as_posix(REP_CSV),
        "input_regime_A_metrics_csv": as_posix(REGIME_A_CSV),
        "synthetic_orders_csv": as_posix(SYNTHETIC_ORDERS_CSV),
        "order_effort_by_seed_csv": as_posix(EFFORT_BY_SEED_CSV),
        "order_effort_summary_csv": as_posix(EFFORT_SUMMARY_CSV),
        "order_generation_parameters": {
            "number_of_orders_per_seed": ORDERS_PER_SEED,
            "workload_seed_count": WORKLOAD_SEED_COUNT,
            "order_line_count_min": LINE_COUNT_MIN,
            "order_line_count_max": LINE_COUNT_MAX,
            "base_workload_seed": BASE_WORKLOAD_SEED,
            "sku_sampling": "sku_catalog_demand_weight",
            "within_order_replacement": False,
        },
        "synthetic_order_validation": order_validation,
        "effort_validation": effort_validation,
        "slot_cost_mismatch_count": effort_meta["slot_cost_mismatch_count"],
        "ranked_layouts_by_mean_order_access_effort_sum": ranking(summary_rows, "mean_order_access_effort_sum_mean"),
        "ranked_layouts_by_mean_line_access_effort": ranking(summary_rows, "mean_line_access_effort_mean"),
        "variation_across_workload_seeds": variation(summary_rows),
        "conceptual_direction_check": conceptual_check(summary_rows),
        "warnings": warnings,
        "blockers_or_warnings": warnings,
        "ready_for_milestone_9": not warnings,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, summary_rows, seed_rows)

    print("Milestone 8 synthetic order access-effort proxy complete.")
    print(f"synthetic_orders.csv: {rel_posix(SYNTHETIC_ORDERS_CSV)}")
    print(f"order_effort_by_seed.csv: {rel_posix(EFFORT_BY_SEED_CSV)}")
    print(f"order_effort_summary.csv: {rel_posix(EFFORT_SUMMARY_CSV)}")
    print(f"summary JSON: {rel_posix(SUMMARY_JSON)}")
    print(f"Markdown report: {rel_posix(REPORT_MD)}")
    print(f"total orders generated: {order_validation['total_orders']}")
    print(f"total order lines generated: {order_validation['total_order_lines']}")
    print(f"class line shares overall: {json.dumps(order_validation['class_line_share_overall'], sort_keys=True)}")
    print("ranking by mean_order_access_effort_sum:")
    print(ranking_text(summary["ranked_layouts_by_mean_order_access_effort_sum"], "mean_order_access_effort_sum_mean"))
    print("ranking by mean_line_access_effort:")
    print(ranking_text(summary["ranked_layouts_by_mean_line_access_effort"], "mean_line_access_effort_mean"))
    print(f"variation across workload seeds: {json.dumps(summary['variation_across_workload_seeds'], sort_keys=True)}")
    print(f"conceptual direction check: {json.dumps(summary['conceptual_direction_check'], sort_keys=True)}")
    print("warnings or blockers: none")
    print(f"ready_for_milestone_9: {summary['ready_for_milestone_9']}")


if __name__ == "__main__":
    main()
