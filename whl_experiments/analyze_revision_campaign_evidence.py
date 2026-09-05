"""Analyze saved revision-campaign evidence without rerunning optimizers.

This public post-processing tool implements the Section-5 indicator/statistical
protocol used by the IJPR manuscript. Phase-12B V0 is intentionally reused
from the completed Phase-11 proposed NSGA-II+BS runs; no V0 rerun is expected
under ``p12b``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


SEEDS = tuple(range(101, 131))
INSTANCES = (
    "AT_S_comercial_layout_AW_3",
    "demo_layout_door_left_AW_2",
    "Gyorgy-KOVACS_WH_Narrow_AW_4",
    "Gyorgy-KOVACS_WH_Wide_AW_5",
)
HV_REFERENCE = np.asarray([1.1, 1.1, 1.1], dtype=float)

PHASE11_METHODS = (
    "proposed_nsga2_bs",
    "bs_only_direct",
    "random_restart_bs",
)
PHASE12B_VARIANTS = (
    "V0_full_proposed",
    "V1_fixed_sorting",
    "V2_fixed_weights",
    "V3_uniform_mutation",
    "V4_no_symmetry_breaking",
    "V5_random_feasible_start_spacing",
)
SUMMARY_METRICS = (
    "hypervolume",
    "igd_plus",
    "osd",
    "nondominated_solution_count",
    "archive_size_rank_0_3",
    "unique_layout_signature_count_rank_0_3",
    "runtime_seconds",
)


@dataclass(frozen=True)
class SourceSpec:
    logical_label: str
    relative_root: str
    manifest_phase: str
    manifest_label: str


PHASE_SOURCES: dict[str, tuple[SourceSpec, ...]] = {
    "phase11": (
        SourceSpec("proposed_nsga2_bs", "p11/nsga2", "phase11", "proposed_nsga2_bs"),
        SourceSpec("bs_only_direct", "p11/bsonly", "phase11", "bs_only"),
        SourceSpec("random_restart_bs", "p11/rrbs", "phase11", "random_restart_bs"),
    ),
    "phase12b": (
        SourceSpec("V0_full_proposed", "p11/nsga2", "phase11", "proposed_nsga2_bs"),
        SourceSpec("V1_fixed_sorting", "p12b/V1_fixsort", "phase12b", "V1_fixed_sorting"),
        SourceSpec("V2_fixed_weights", "p12b/V2_fixw", "phase12b", "V2_fixed_weights"),
        SourceSpec("V3_uniform_mutation", "p12b/V3_um", "phase12b", "V3_uniform_mutation"),
        SourceSpec("V4_no_symmetry_breaking", "p12b/V4_nsb", "phase12b", "V4_no_symmetry_breaking"),
        SourceSpec(
            "V5_random_feasible_start_spacing",
            "p12b/V5_rfs",
            "phase12b",
            "V5_random_feasible_start_spacing",
        ),
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_array_hash(points: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    payload = f"shape={array.shape};dtype={array.dtype};".encode("ascii") + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def metric(entry: dict[str, Any], name: str) -> Any:
    value = entry.get(name)
    if value not in (None, ""):
        return value
    nested = entry.get("metrics")
    return nested.get(name) if isinstance(nested, dict) else None


def objective(entry: dict[str, Any]) -> tuple[float, float, float]:
    """Return manuscript-order minimization vector (N_locked, -N_pf, R_p)."""
    return (
        float(metric(entry, "interior_storage")),
        -float(metric(entry, "pick_faces")),
        float(metric(entry, "retrieval_penalty")),
    )


def final_entry_is_feasible(entry: dict[str, Any]) -> bool:
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return True
    checks = (
        "exact_width_ok",
        "has_access_anchor_reachable_aisle_network",
        "has_access_anchor_connected_aisle",
    )
    return all(bool(metrics[name]) for name in checks if name in metrics)


def unique_rows(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    return np.unique(points.reshape(-1, points.shape[-1]), axis=0)


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mask = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        mask[i] = not np.any(
            np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        )
    return mask


def nondominated_points(points: np.ndarray) -> np.ndarray:
    points = unique_rows(points)
    return points[nondominated_mask(points)] if len(points) else points


def normalize(points: np.ndarray, minima: np.ndarray, maxima: np.ndarray) -> np.ndarray:
    ranges = np.asarray(maxima, dtype=float) - np.asarray(minima, dtype=float)
    safe = np.where(ranges == 0.0, 1.0, ranges)
    return (np.asarray(points, dtype=float) - minima) / safe


def hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    if points.size == 0:
        return 0.0
    points = points.reshape(-1, reference_point.shape[0])
    points = points[np.all(points < reference_point, axis=1)]
    if points.size == 0:
        return 0.0
    return float(_hypervolume_recursive(nondominated_points(points), reference_point))


def _hypervolume_recursive(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = _as_2d(points)
    if points.size == 0:
        return 0.0
    if reference_point.shape[0] == 1:
        return max(0.0, float(reference_point[0] - np.min(points[:, 0])))
    breaks = sorted(set(float(value) for value in points[:, 0]) | {float(reference_point[0])})
    total = 0.0
    for left, right in zip(breaks, breaks[1:], strict=False):
        width = right - left
        if width <= 0.0:
            continue
        active = points[points[:, 0] <= left]
        if active.size:
            total += width * _hypervolume_recursive(active[:, 1:], reference_point[1:])
    return total


def igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    approximation = _as_2d(approximation)
    reference_front = _as_2d(reference_front)
    if approximation.size == 0 or reference_front.size == 0:
        return math.nan
    minima = []
    for ref in reference_front:
        diff = np.maximum(approximation - ref, 0.0)
        minima.append(float(np.min(np.linalg.norm(diff, axis=1))))
    return float(np.mean(minima))


def osd(points: np.ndarray) -> float:
    points = _as_2d(points)
    if points.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    upper = distances[np.triu_indices(points.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _as_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        return points.reshape(1, -1)
    return points


def quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {"n": 0, "mean": "", "std": "", "median": "", "iqr": "", "min": "", "max": ""}
    return {
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def load_completed_manifest(
    results_root: Path,
) -> tuple[dict[tuple[str, str, str, int], dict[str, str]], Path]:
    path = results_root / "manifests" / "batch_manifest.csv"
    rows = read_csv(path)
    mapping: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        try:
            return_code = int(row.get("return_code", ""))
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        if return_code != 0:
            continue
        key = (row["phase"], row["method_or_variant"], row["instance"], seed)
        if key in mapping:
            raise ValueError(f"duplicate completed manifest key: {key}")
        mapping[key] = row
    if not mapping:
        raise ValueError(f"no completed runs found in {path}")
    return mapping, path


def discover_indexes(source_root: Path) -> dict[tuple[str, int], Path]:
    mapping: dict[tuple[str, int], Path] = {}
    for path in sorted(source_root.rglob("final_ranked_layouts_index.json")):
        seed_dir = path.parent
        instance_dir = seed_dir.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        key = (instance_dir.name, int(seed_dir.name.removeprefix("seed_")))
        if key in mapping:
            raise ValueError(f"duplicate final-ranked index under {source_root}: {key}")
        mapping[key] = path
    expected = {(instance, seed) for instance in INSTANCES for seed in SEEDS}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(
            f"archive coverage mismatch under {source_root}; missing={missing[:10]}, extra={extra[:10]}"
        )
    return mapping


def load_phase_records(
    results_root: Path,
    phase: str,
    manifest: dict[tuple[str, str, str, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[Path]]:
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    input_files: set[Path] = set()
    for spec in PHASE_SOURCES[phase]:
        source_root = results_root / spec.relative_root
        index_map = discover_indexes(source_root)
        for instance in INSTANCES:
            for seed in SEEDS:
                manifest_key = (spec.manifest_phase, spec.manifest_label, instance, seed)
                if manifest_key not in manifest:
                    raise KeyError(f"completed manifest row missing: {manifest_key}")
                manifest_row = manifest[manifest_key]
                runtime = float(manifest_row["runtime_seconds"])
                index_path = index_map[(instance, seed)]
                with index_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, list):
                    raise ValueError(f"final-ranked index is not a list: {index_path}")
                archive = [
                    item
                    for item in data
                    if int(item.get("rank", 999)) <= 3 and final_entry_is_feasible(item)
                ]
                if not archive:
                    raise ValueError(f"empty feasible rank-0--3 archive: {index_path}")
                signatures = []
                for item in archive:
                    signature = str(item.get("layout_signature") or "")
                    if not signature:
                        raise ValueError(f"blank layout_signature in {index_path}")
                    signatures.append(signature)
                    records.append(
                        {
                            "comparison_group": phase,
                            "method_or_variant": spec.logical_label,
                            "source_phase": spec.manifest_phase,
                            "source_manifest_label": spec.manifest_label,
                            "instance": instance,
                            "seed": seed,
                            "rank": int(item["rank"]),
                            "layout_signature": signature,
                            "archive_key": item.get("archive_key", ""),
                            "candidate_id": item.get("candidate_id", ""),
                            "objectives": objective(item),
                            "runtime_seconds": runtime,
                            "source_path": index_path.as_posix(),
                        }
                    )
                provenance.append(
                    {
                        "comparison_group": phase,
                        "method_or_variant": spec.logical_label,
                        "source_phase": spec.manifest_phase,
                        "source_manifest_label": spec.manifest_label,
                        "source_relative_root": spec.relative_root,
                        "instance": instance,
                        "seed": seed,
                        "archive_scope": "final feasible rank 0-3",
                        "archive_record_count": len(archive),
                        "unique_signature_count": len(set(signatures)),
                        "runtime_seconds": runtime,
                        "archive_index_path": index_path.as_posix(),
                        "manifest_output_dir": manifest_row.get("output_dir", ""),
                    }
                )
                input_files.add(index_path)
    return records, provenance, input_files


def compute_indicators(
    rows: list[dict[str, Any]],
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    labels = [spec.logical_label for spec in PHASE_SOURCES[phase]]
    for instance in INSTANCES:
        subset = [row for row in rows if row["instance"] == instance]
        union = np.asarray([row["objectives"] for row in subset], dtype=float)
        minima, maxima = union.min(axis=0), union.max(axis=0)
        normalized_union = normalize(union, minima, maxima)
        reference_front = nondominated_points(normalized_union)
        ranges = maxima - minima
        metadata_rows.append(
            {
                "comparison_group": phase,
                "instance": instance,
                "objective_vector": "(N_locked, -N_pf, R_p); minimization",
                "archive_scope": "final feasible rank 0-3",
                "normalization_scope": "full instance/comparison-group union",
                "source_record_count": len(union),
                "source_unique_objective_count": len(unique_rows(union)),
                "min_N_locked": minima[0],
                "max_N_locked": maxima[0],
                "range_N_locked": ranges[0],
                "min_negative_N_pf": minima[1],
                "max_negative_N_pf": maxima[1],
                "range_negative_N_pf": ranges[1],
                "min_R_p": minima[2],
                "max_R_p": maxima[2],
                "range_R_p": ranges[2],
                "zero_range_replacement": 1,
                "reference_front_size": len(reference_front),
                "hv_reference_point": "(1.1,1.1,1.1)",
                "union_objectives_sha256": stable_array_hash(union),
                "reference_front_sha256": stable_array_hash(reference_front),
                "comparability_note": "Indicators are comparable within this comparison group only.",
            }
        )
        for label in labels:
            for seed in SEEDS:
                archive = [
                    row
                    for row in subset
                    if row["method_or_variant"] == label and row["seed"] == seed
                ]
                if not archive:
                    raise ValueError(f"no archive rows for {phase}/{label}/{instance}/seed_{seed}")
                raw_points = np.asarray([row["objectives"] for row in archive], dtype=float)
                normalized = normalize(raw_points, minima, maxima)
                normalized_nd = nondominated_points(normalized)
                raw_nd = nondominated_points(raw_points)
                signatures = {row["layout_signature"] for row in archive}
                seed_rows.append(
                    {
                        "comparison_group": phase,
                        "instance": instance,
                        "method_or_variant": label,
                        "seed": seed,
                        "hypervolume": hypervolume(normalized_nd, HV_REFERENCE),
                        "igd_plus": igd_plus(normalized_nd, reference_front),
                        "osd": osd(normalized_nd),
                        "nondominated_solution_count": len(raw_nd),
                        "archive_size_rank_0_3": len(archive),
                        "unique_layout_signature_count_rank_0_3": len(signatures),
                        "runtime_seconds": archive[0]["runtime_seconds"],
                        "reference_front_size": len(reference_front),
                        "normalization_scope": "per instance and comparison group",
                        "archive_scope": "final feasible rank 0-3",
                        "hv_reference_point": "(1.1,1.1,1.1)",
                    }
                )
    return seed_rows, metadata_rows


def summarize_seed_rows(
    seed_rows: list[dict[str, Any]], *, by_instance: bool
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = (row["comparison_group"], row["method_or_variant"])
        if by_instance:
            key += (row["instance"],)
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        phase, label = key[:2]
        instance = key[2] if by_instance else "ALL_FIXED_BLOCKS"
        for metric_name in SUMMARY_METRICS:
            output.append(
                {
                    "comparison_group": phase,
                    "method_or_variant": label,
                    "instance": instance,
                    "metric": metric_name,
                    **quantile_summary([float(row[metric_name]) for row in rows]),
                    "sampling_unit": (
                        "seed within one instance"
                        if by_instance
                        else "fixed block (instance, seed)"
                    ),
                    "independence_note": (
                        ""
                        if by_instance
                        else "120 observations are 4 instances x 30 seeds, not 120 independent warehouses"
                    ),
                }
            )
    return output


def signature_summary(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    output = []
    for spec in PHASE_SOURCES[phase]:
        per_instance_rows = []
        for instance in INSTANCES:
            subset = [
                row
                for row in rows
                if row["method_or_variant"] == spec.logical_label and row["instance"] == instance
            ]
            signature_seeds: dict[str, set[int]] = defaultdict(set)
            for row in subset:
                signature_seeds[row["layout_signature"]].add(int(row["seed"]))
            repeated = [seeds for seeds in signature_seeds.values() if len(seeds) > 1]
            item = {
                "comparison_group": phase,
                "method_or_variant": spec.logical_label,
                "scope": "per_instance",
                "instance": instance,
                "total_archive_entries_rank_0_3": len(subset),
                "unique_signatures_cross_seed_rank_0_3": len(signature_seeds),
                "repeated_signatures_cross_seed_rank_0_3": len(repeated),
                "cross_seed_rediscovery_ratio_rank_0_3": 1.0 - len(signature_seeds) / len(subset),
                "mean_seeds_per_repeated_signature": mean(map(len, repeated)) if repeated else 0.0,
                "max_seed_overlap": max(map(len, repeated), default=0),
                "inference_scope": "descriptive aggregate; not a Wilcoxon observation",
            }
            output.append(item)
            per_instance_rows.append(item)
        total = sum(row["total_archive_entries_rank_0_3"] for row in per_instance_rows)
        unique = sum(row["unique_signatures_cross_seed_rank_0_3"] for row in per_instance_rows)
        repeated_count = sum(row["repeated_signatures_cross_seed_rank_0_3"] for row in per_instance_rows)
        weighted_seed_total = sum(
            row["mean_seeds_per_repeated_signature"] * row["repeated_signatures_cross_seed_rank_0_3"]
            for row in per_instance_rows
        )
        output.append(
            {
                "comparison_group": phase,
                "method_or_variant": spec.logical_label,
                "scope": "aggregate_sum_of_per_instance_counts",
                "instance": "ALL_FIXED_BLOCKS",
                "total_archive_entries_rank_0_3": total,
                "unique_signatures_cross_seed_rank_0_3": unique,
                "repeated_signatures_cross_seed_rank_0_3": repeated_count,
                "cross_seed_rediscovery_ratio_rank_0_3": 1.0 - unique / total,
                "mean_seeds_per_repeated_signature": (
                    weighted_seed_total / repeated_count if repeated_count else 0.0
                ),
                "max_seed_overlap": max(row["max_seed_overlap"] for row in per_instance_rows),
                "inference_scope": "descriptive aggregate; not a Wilcoxon observation",
            }
        )
    return output


def phase11_seed_novelty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label in PHASE11_METHODS:
        for instance in INSTANCES:
            archives: dict[int, set[str]] = defaultdict(set)
            for row in rows:
                if row["method_or_variant"] == label and row["instance"] == instance:
                    archives[int(row["seed"])].add(row["layout_signature"])
            occurrences = Counter(sig for signatures in archives.values() for sig in signatures)
            for seed in SEEDS:
                signatures = archives[seed]
                if not signatures:
                    raise ValueError(f"empty signature archive for {label}/{instance}/seed_{seed}")
                exclusive = sum(occurrences[sig] == 1 for sig in signatures)
                rediscovered = len(signatures) - exclusive
                output.append(
                    {
                        "comparison_group": "phase11",
                        "instance": instance,
                        "method_or_variant": label,
                        "seed": seed,
                        "seed_archive_signature_count": len(signatures),
                        "seed_exclusive_signature_count": exclusive,
                        "seed_rediscovered_signature_count": rediscovered,
                        "seed_rediscovered_signature_ratio": rediscovered / len(signatures),
                        "definition": "order-invariant within method and instance across seeds 101-130",
                    }
                )
    return output


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted_sorted = []
    running = 0.0
    m = len(p)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * float(p[index])))
        adjusted_sorted.append(running)
    result = np.empty(m, dtype=float)
    for index, adjusted in zip(order, adjusted_sorted, strict=True):
        result[index] = adjusted
    return result.tolist()


def preferred_direction(metric_name: str) -> str:
    if metric_name in {"igd_plus", "runtime_seconds", "seed_rediscovered_signature_ratio"}:
        return "lower"
    if metric_name == "osd":
        return "descriptive_no_preferred_direction"
    return "higher"


def paired_test(first: Sequence[float], second: Sequence[float], metric_name: str) -> dict[str, Any]:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal lengths")
    if np.all(a == b):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    direction = preferred_direction(metric_name)
    raw = a - b
    oriented = -raw if direction == "lower" else raw.copy()
    if direction == "descriptive_no_preferred_direction":
        oriented = raw.copy()
    nonzero = oriented[oriented != 0.0]
    if len(nonzero):
        ranks = rankdata(np.abs(nonzero), method="average")
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        rank_biserial = 0.0
    return {
        "wilcoxon_statistic": statistic,
        "p_value_raw": p_value,
        "zero_method": "wilcox",
        "alternative": "two-sided",
        "preferred_direction": direction,
        "rank_biserial_method1_preferred": (
            "" if direction == "descriptive_no_preferred_direction" else rank_biserial
        ),
        "method1_wins": int(np.sum(oriented > 0)),
        "ties": int(np.sum(oriented == 0)),
        "method1_losses": int(np.sum(oriented < 0)),
        "method1_win_proportion": float(np.mean(oriented > 0)),
        "tie_proportion": float(np.mean(oriented == 0)),
        "method1_loss_proportion": float(np.mean(oriented < 0)),
        "raw_mean_difference_method1_minus_method2": float(np.mean(raw)),
    }


def phase11_statistics(
    seed_rows: list[dict[str, Any]],
    novelty_rows: list[dict[str, Any]],
    *,
    pooled: bool,
) -> list[dict[str, Any]]:
    combined: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        key = (row["instance"], int(row["seed"]), row["method_or_variant"])
        combined[key].update({name: float(row[name]) for name in SUMMARY_METRICS})
    for row in novelty_rows:
        key = (row["instance"], int(row["seed"]), row["method_or_variant"])
        combined[key].update(
            {
                "seed_archive_signature_count": float(row["seed_archive_signature_count"]),
                "seed_exclusive_signature_count": float(row["seed_exclusive_signature_count"]),
                "seed_rediscovered_signature_ratio": float(row["seed_rediscovered_signature_ratio"]),
            }
        )
    metric_names = (
        "hypervolume",
        "igd_plus",
        "osd",
        "unique_layout_signature_count_rank_0_3",
        "runtime_seconds",
        "seed_exclusive_signature_count",
        "seed_rediscovered_signature_ratio",
    )
    pairs = (
        ("proposed_nsga2_bs", "bs_only_direct"),
        ("proposed_nsga2_bs", "random_restart_bs"),
        ("random_restart_bs", "bs_only_direct"),
    )
    scopes = (
        [("ALL_FIXED_BLOCKS", INSTANCES)]
        if pooled
        else [(instance, (instance,)) for instance in INSTANCES]
    )
    output = []
    for scope_name, instances in scopes:
        for metric_name in metric_names:
            family = []
            for first_method, second_method in pairs:
                block_keys = [(instance, seed) for instance in instances for seed in SEEDS]
                first = [combined[(instance, seed, first_method)][metric_name] for instance, seed in block_keys]
                second = [combined[(instance, seed, second_method)][metric_name] for instance, seed in block_keys]
                family.append(
                    {
                        "scope": "pooled_secondary_fixed_block" if pooled else "per_instance_primary",
                        "instance": scope_name,
                        "metric": metric_name,
                        "method1": first_method,
                        "method2": second_method,
                        "pairing_key": "(instance, seed)" if pooled else "seed",
                        "n_pairs": len(first),
                        "method1_mean": float(np.mean(first)),
                        "method2_mean": float(np.mean(second)),
                        **paired_test(first, second, metric_name),
                        "holm_family": (
                            f"three Phase-11 method pairs within metric={metric_name}, instance={scope_name}"
                        ),
                        "independence_note": (
                            "120 observations are 4 instances x 30 seeds, not 120 independent warehouses"
                            if pooled
                            else "30 matched seeds within one warehouse instance"
                        ),
                    }
                )
            adjusted = holm_adjust([row["p_value_raw"] for row in family])
            for row, p_adjusted in zip(family, adjusted, strict=True):
                row["p_value_holm"] = p_adjusted
                output.append(row)
    return output


def phase12b_v0_pairwise_statistics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supporting V0-vs-ablation tests; Table 7 itself remains descriptive."""
    by_key = {
        (row["instance"], int(row["seed"]), row["method_or_variant"]): row
        for row in seed_rows
    }
    output = []
    for instance in INSTANCES:
        for metric_name in (
            "hypervolume",
            "igd_plus",
            "osd",
            "unique_layout_signature_count_rank_0_3",
            "runtime_seconds",
        ):
            family = []
            for variant in PHASE12B_VARIANTS[1:]:
                first = [
                    float(by_key[(instance, seed, "V0_full_proposed")][metric_name])
                    for seed in SEEDS
                ]
                second = [float(by_key[(instance, seed, variant)][metric_name]) for seed in SEEDS]
                family.append(
                    {
                        "scope": "per_instance_supporting",
                        "instance": instance,
                        "metric": metric_name,
                        "method1": "V0_full_proposed",
                        "method2": variant,
                        "n_pairs": 30,
                        **paired_test(first, second, metric_name),
                        "holm_family": (
                            f"five V0-vs-ablation comparisons within metric={metric_name}, instance={instance}"
                        ),
                    }
                )
            adjusted = holm_adjust([row["p_value_raw"] for row in family])
            for row, p_adjusted in zip(family, adjusted, strict=True):
                row["p_value_holm"] = p_adjusted
                output.append(row)
    return output


def friedman_instance_mean_check(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric_name in (
        "hypervolume",
        "igd_plus",
        "unique_layout_signature_count_rank_0_3",
        "runtime_seconds",
    ):
        instance_means: dict[str, list[float]] = {method: [] for method in PHASE11_METHODS}
        for instance in INSTANCES:
            for method in PHASE11_METHODS:
                values = [
                    float(row[metric_name])
                    for row in seed_rows
                    if row["instance"] == instance and row["method_or_variant"] == method
                ]
                instance_means[method].append(float(np.mean(values)))
        result = friedmanchisquare(*(instance_means[method] for method in PHASE11_METHODS))
        output.append(
            {
                "metric": metric_name,
                "instance_blocks": len(INSTANCES),
                "method_count": len(PHASE11_METHODS),
                "methods": ";".join(PHASE11_METHODS),
                "friedman_chi_square": float(result.statistic),
                "p_value": float(result.pvalue),
                "interpretation_scope": "descriptive corroboration; only four warehouse-instance blocks",
            }
        )
    return output


def manuscript_summary_rows(
    overall: list[dict[str, Any]],
    signatures: list[dict[str, Any]],
    labels: Sequence[str],
    label_field: str,
) -> list[dict[str, Any]]:
    summary_map = {(row["method_or_variant"], row["metric"]): row for row in overall}
    sig_map = {
        row["method_or_variant"]: row
        for row in signatures
        if row["scope"] == "aggregate_sum_of_per_instance_counts"
    }
    output = []
    for label in labels:
        sig = sig_map[label]
        row = {
            label_field: label,
            "hv_mean": summary_map[(label, "hypervolume")]["mean"],
            "hv_std": summary_map[(label, "hypervolume")]["std"],
            "igd_plus_mean": summary_map[(label, "igd_plus")]["mean"],
            "igd_plus_std": summary_map[(label, "igd_plus")]["std"],
            "osd_mean": summary_map[(label, "osd")]["mean"],
            "osd_std": summary_map[(label, "osd")]["std"],
            "unique_per_run_mean": summary_map[
                (label, "unique_layout_signature_count_rank_0_3")
            ]["mean"],
            "unique_per_run_std": summary_map[
                (label, "unique_layout_signature_count_rank_0_3")
            ]["std"],
            "total_final_rank_0_3_archive_entries": sig["total_archive_entries_rank_0_3"],
            "unique_signatures_cross_seed": sig["unique_signatures_cross_seed_rank_0_3"],
            "cross_seed_rediscovery_ratio": sig["cross_seed_rediscovery_ratio_rank_0_3"],
            "runtime_mean_seconds": summary_map[(label, "runtime_seconds")]["mean"],
            "runtime_std_seconds": summary_map[(label, "runtime_seconds")]["std"],
        }
        if label == "V0_full_proposed":
            row["v0_source_note"] = "reused Phase-11 proposed raw archives"
        output.append(row)
    return output


def input_hash_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    output = []
    for path in sorted({path.resolve() for path in paths}, key=lambda item: str(item).lower()):
        output.append(
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return output


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_readme(results_root: Path, phases: Sequence[str]) -> str:
    return f"""# Revision structural evidence

Generated by `python -m whl_experiments.analyze_revision_campaign_evidence`.

Source campaign: `{results_root.as_posix()}`

Selected phases: {", ".join(phases)}.

## Indicator protocol

- Final feasible Pareto-rank 0--3 archive records only.
- Minimization vector `(N_locked, -N_pf, R_p)`.
- Min--max normalization is separate for each fixed instance and comparison group.
- A zero observed range is replaced by one.
- The empirical reference front is the unique nondominated normalized union.
- HV reference point: `(1.1,1.1,1.1)`.
- IGD+: positive-part minimization distance.
- OSD: mean pairwise Euclidean distance among normalized nondominated points;
  descriptive, with no preferred direction.
- Structural identity uses the saved exact-grid `layout_signature`.

Phase-11 and Phase-12B indicators are comparable within, not across, groups.

## Phase-12B V0

`V0_full_proposed` is not rerun. The analyzer reuses completed Phase-11
`proposed_nsga2_bs` raw final archives and recomputes their indicators inside
the V0--V5 comparison-specific union.

## Statistical protocol

Phase-11 primary tests are per instance with 30 matched seeds. The three method
pairs form the Holm family within each metric/instance. Two-sided Wilcoxon
signed-rank uses `zero_method=wilcox`; paired rank-biserial effects are oriented
to method 1. The 4 x 30 pooled analysis is secondary fixed-block evidence only.
The four-instance Friedman result is descriptive corroboration.

`phase12b_v0_pairwise_stats.csv` is supporting evidence only; Table 7 remains a
descriptive ablation summary unless the manuscript explicitly cites these tests.
"""


def analyze(results_root: Path, output_dir: Path, phases: Sequence[str]) -> dict[str, Any]:
    prepare_output(output_dir)
    manifest, manifest_path = load_completed_manifest(results_root)
    input_files: set[Path] = {manifest_path}
    all_provenance = []
    all_reference_metadata = []
    summary: dict[str, Any] = {
        "results_root": str(results_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "phases": list(phases),
    }
    for phase in phases:
        records, provenance, phase_inputs = load_phase_records(results_root, phase, manifest)
        input_files.update(phase_inputs)
        all_provenance.extend(provenance)
        seed_rows, reference_metadata = compute_indicators(records, phase)
        all_reference_metadata.extend(reference_metadata)
        by_instance = summarize_seed_rows(seed_rows, by_instance=True)
        overall = summarize_seed_rows(seed_rows, by_instance=False)
        signatures = signature_summary(records, phase)
        write_csv(output_dir / f"{phase}_seed_level.csv", seed_rows)
        write_csv(output_dir / f"{phase}_summary_by_instance.csv", by_instance)
        write_csv(output_dir / f"{phase}_summary_overall.csv", overall)
        write_csv(output_dir / f"{phase}_signature_summary.csv", signatures)
        phase_summary = {
            "logical_runs": len(seed_rows),
            "archive_records": len(records),
            "methods_or_variants": len({row["method_or_variant"] for row in seed_rows}),
        }
        if phase == "phase11":
            novelty = phase11_seed_novelty(records)
            stats_by_instance = phase11_statistics(seed_rows, novelty, pooled=False)
            stats_pooled = phase11_statistics(seed_rows, novelty, pooled=True)
            friedman = friedman_instance_mean_check(seed_rows)
            table5 = manuscript_summary_rows(
                overall, signatures, PHASE11_METHODS, "method"
            )
            write_csv(output_dir / "phase11_seed_novelty.csv", novelty)
            write_csv(output_dir / "phase11_stats_by_instance.csv", stats_by_instance)
            write_csv(output_dir / "phase11_stats_pooled.csv", stats_pooled)
            write_csv(output_dir / "phase11_friedman_instance_means.csv", friedman)
            write_csv(output_dir / "table5_phase11_manuscript_values.csv", table5)
            phase_summary["primary_statistical_rows"] = len(stats_by_instance)
        if phase == "phase12b":
            pairwise = phase12b_v0_pairwise_statistics(seed_rows)
            table7 = manuscript_summary_rows(
                overall, signatures, PHASE12B_VARIANTS, "variant"
            )
            write_csv(output_dir / "phase12b_v0_pairwise_stats.csv", pairwise)
            write_csv(output_dir / "table7_phase12b_manuscript_values.csv", table7)
            phase_summary["v0_source"] = "phase11/proposed_nsga2_bs"
        summary[phase] = phase_summary
    write_csv(output_dir / "input_manifest.csv", all_provenance)
    write_csv(output_dir / "indicator_reference_metadata.csv", all_reference_metadata)
    write_csv(output_dir / "input_hashes.csv", input_hash_rows(input_files))
    (output_dir / "README.md").write_text(
        build_readme(results_root, phases), encoding="utf-8"
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/revision_final_30seed_nofg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reproducibility/revision_final_30seed_nofg/structural"),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=tuple(PHASE_SOURCES),
        default=["phase11", "phase12b"],
        help="Completed comparison groups to analyze. Default: phase11 phase12b.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.results_root, args.output_dir, args.phases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
