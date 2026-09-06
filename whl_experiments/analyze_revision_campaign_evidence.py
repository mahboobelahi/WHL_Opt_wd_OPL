"""Analyze saved final-clean revision evidence without rerunning optimizers.

This public post-processing tool implements the manuscript-facing structural
indicator/statistical protocol for the final-clean IJPR revision campaigns.

Comparison families are intentionally separated:

* Phase 11: Proposed / BS-only / RRBS.
* Phase 12B: V0--V5, where V0 reuses the Phase-11 Proposed raw archives.
* Phase 12C: V0 / V6 / V7, where V0 again reuses the same Phase-11 raw
  archives but indicators are recomputed inside the V0/V6/V7 union.
* V6b: a separate Demo-1-w2 matched V0-versus-V6b binding-depth diagnostic
  with its own two-condition normalization/reference front.

The module is post-processing only. It never invokes an optimizer.
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

from whl_algorithms.beam_node import layout_signature


SEEDS = tuple(range(101, 131))
INSTANCES = (
    "AT_S_comercial_layout_AW_3",
    "demo_layout_door_left_AW_2",
    "Gyorgy-KOVACS_WH_Narrow_AW_4",
    "Gyorgy-KOVACS_WH_Wide_AW_5",
)
V6B_INSTANCE = "demo_layout_door_left_AW_2"
V6B_VARIANT = "V6b_binding_depth10"
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
PHASE12C_VARIANTS = (
    "V0_full_proposed",
    "V6_depth15_beam_default",
    "V7_beam_plus1_depth_default",
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

V6B_EXPECTED_MAX_DEPTH = {
    "V0_full_proposed": 28,
    V6B_VARIANT: 10,
}
V6B_EXPECTED_AUTO_PARAMETERS = {
    "auto_population_size": 10,
    "auto_generations": 15,
    "auto_beam_width": 3,
    "auto_max_depth": 28,
    "auto_decode_budget": 150,
}
V6B_CONFIG_FIELDS = (
    "method",
    "instance",
    "seed",
    "aisle_width",
    "budget_policy",
    "auto_population_size",
    "auto_generations",
    "auto_beam_width",
    "auto_max_depth",
    "auto_decode_budget",
    "sorting_rule_mode",
    "sorting_rule_pool_path",
    "adaptive_weight_mode",
    "fixed_w1",
    "fixed_w2",
    "lambda",
    "mutation_mode",
    "initialization_spacing_mode",
    "adaptive_spacing_mode",
    "adaptive_spacing_alpha",
    "adaptive_spacing_bf",
    "symmetry_breaking_enabled",
    "archive_layouts",
    "archive_rank_max",
    "profile_light",
    "save_generation_objectives",
    "objective_keys",
    "objective_directions",
    "input_mask_path",
    "beta_h",
    "beta_v",
    "min_fragment_size",
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
    "phase12c": (
        SourceSpec("V0_full_proposed", "p11/nsga2", "phase11", "proposed_nsga2_bs"),
        SourceSpec(
            "V6_depth15_beam_default",
            "p12c/V6_d15",
            "phase12c",
            "V6_depth15_beam_default",
        ),
        SourceSpec(
            "V7_beam_plus1_depth_default",
            "p12c/V7_bw1",
            "phase12c",
            "V7_beam_plus1_depth_default",
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


def exact_grid_signature(grid: np.ndarray) -> str:
    """Return the repository's exact-grid SHA-1 structural signature."""
    return hashlib.sha1(layout_signature(np.asarray(grid))).hexdigest()


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
    breaks = sorted(
        set(float(value) for value in points[:, 0]) | {float(reference_point[0])}
    )
    total = 0.0
    for left, right in zip(breaks, breaks[1:], strict=False):
        width = right - left
        if width <= 0.0:
            continue
        active = points[points[:, 0] <= left]
        if active.size:
            total += width * _hypervolume_recursive(
                active[:, 1:], reference_point[1:]
            )
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
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "median": "",
            "iqr": "",
            "min": "",
            "max": "",
        }
    return {
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def discover_indexes_for(
    source_root: Path,
    instances: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], Path]:
    mapping: dict[tuple[str, int], Path] = {}
    for path in sorted(source_root.rglob("final_ranked_layouts_index.json")):
        seed_dir = path.parent
        instance_dir = seed_dir.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        key = (instance_dir.name, seed)
        if key in mapping:
            raise ValueError(f"duplicate final-ranked index under {source_root}: {key}")
        mapping[key] = path
    expected = {(instance, seed) for instance in instances for seed in seeds}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(
            f"archive coverage mismatch under {source_root}; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return mapping


def discover_indexes(source_root: Path) -> dict[tuple[str, int], Path]:
    return discover_indexes_for(source_root, INSTANCES, SEEDS)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_metadata_path(index_path: Path) -> Path:
    return index_path.parent / "run_metadata.json"


def experiment_metadata_path(index_path: Path) -> Path:
    # .../<experiment>/runs/<instance>/seed_<n>/final_ranked_layouts_index.json
    return index_path.parent.parents[2] / "experiment_metadata.json"


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
                manifest_key = (
                    spec.manifest_phase,
                    spec.manifest_label,
                    instance,
                    seed,
                )
                if manifest_key not in manifest:
                    raise KeyError(f"completed manifest row missing: {manifest_key}")
                manifest_row = manifest[manifest_key]
                runtime = float(manifest_row["runtime_seconds"])
                index_path = index_map[(instance, seed)]
                metadata_path = run_metadata_path(index_path)
                data = load_json(index_path)
                run_meta = load_json(metadata_path)
                if not isinstance(data, list):
                    raise ValueError(f"final-ranked index is not a list: {index_path}")
                if run_meta.get("status") != "completed":
                    raise ValueError(f"run metadata not completed: {metadata_path}")
                configured_max_depth = int(
                    run_meta.get("actual_max_depth", run_meta.get("max_depth"))
                )
                archive = [
                    item
                    for item in data
                    if int(item.get("rank", 999)) <= 3
                    and final_entry_is_feasible(item)
                ]
                if not archive:
                    raise ValueError(f"empty feasible rank-0--3 archive: {index_path}")
                signatures = []
                for item in archive:
                    signature = str(item.get("layout_signature") or "")
                    if not signature:
                        raise ValueError(f"blank layout_signature in {index_path}")
                    signatures.append(signature)
                    depth_raw = item.get("depth")
                    depth = int(depth_raw) if depth_raw not in (None, "") else None
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
                            "depth": depth,
                            "configured_max_depth": configured_max_depth,
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
                        "configured_max_depth": configured_max_depth,
                        "runtime_seconds": runtime,
                        "archive_index_path": index_path.as_posix(),
                        "run_metadata_path": metadata_path.as_posix(),
                        "manifest_output_dir": manifest_row.get("output_dir", ""),
                    }
                )
                input_files.update({index_path, metadata_path})
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
                "comparability_note": (
                    "Indicators are comparable within this comparison group only."
                ),
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
                    raise ValueError(
                        f"no archive rows for {phase}/{label}/{instance}/seed_{seed}"
                    )
                raw_points = np.asarray(
                    [row["objectives"] for row in archive], dtype=float
                )
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
                        "hypervolume": hypervolume(
                            normalized_nd, HV_REFERENCE
                        ),
                        "igd_plus": igd_plus(
                            normalized_nd, reference_front
                        ),
                        "osd": osd(normalized_nd),
                        "nondominated_solution_count": len(raw_nd),
                        "archive_size_rank_0_3": len(archive),
                        "unique_layout_signature_count_rank_0_3": len(signatures),
                        "runtime_seconds": archive[0]["runtime_seconds"],
                        "reference_front_size": len(reference_front),
                        "normalization_scope": (
                            "per instance and comparison group"
                        ),
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
                    **quantile_summary(
                        [float(row[metric_name]) for row in rows]
                    ),
                    "sampling_unit": (
                        "seed within one instance"
                        if by_instance
                        else "fixed block (instance, seed)"
                    ),
                    "independence_note": (
                        ""
                        if by_instance
                        else (
                            "120 observations are 4 instances x 30 seeds, "
                            "not 120 independent warehouses"
                        )
                    ),
                }
            )
    return output


def signature_summary(
    rows: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    output = []
    for spec in PHASE_SOURCES[phase]:
        per_instance_rows = []
        for instance in INSTANCES:
            subset = [
                row
                for row in rows
                if row["method_or_variant"] == spec.logical_label
                and row["instance"] == instance
            ]
            signature_seeds: dict[str, set[int]] = defaultdict(set)
            for row in subset:
                signature_seeds[row["layout_signature"]].add(int(row["seed"]))
            repeated = [
                seeds for seeds in signature_seeds.values() if len(seeds) > 1
            ]
            item = {
                "comparison_group": phase,
                "method_or_variant": spec.logical_label,
                "scope": "per_instance",
                "instance": instance,
                "total_archive_entries_rank_0_3": len(subset),
                "unique_signatures_cross_seed_rank_0_3": len(signature_seeds),
                "repeated_signatures_cross_seed_rank_0_3": len(repeated),
                "cross_seed_rediscovery_ratio_rank_0_3": (
                    1.0 - len(signature_seeds) / len(subset)
                ),
                "mean_seeds_per_repeated_signature": (
                    mean(map(len, repeated)) if repeated else 0.0
                ),
                "max_seed_overlap": max(map(len, repeated), default=0),
                "inference_scope": (
                    "descriptive aggregate; not a Wilcoxon observation"
                ),
            }
            output.append(item)
            per_instance_rows.append(item)
        total = sum(
            row["total_archive_entries_rank_0_3"]
            for row in per_instance_rows
        )
        unique = sum(
            row["unique_signatures_cross_seed_rank_0_3"]
            for row in per_instance_rows
        )
        repeated_count = sum(
            row["repeated_signatures_cross_seed_rank_0_3"]
            for row in per_instance_rows
        )
        weighted_seed_total = sum(
            row["mean_seeds_per_repeated_signature"]
            * row["repeated_signatures_cross_seed_rank_0_3"]
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
                    weighted_seed_total / repeated_count
                    if repeated_count
                    else 0.0
                ),
                "max_seed_overlap": max(
                    row["max_seed_overlap"] for row in per_instance_rows
                ),
                "inference_scope": (
                    "descriptive aggregate; not a Wilcoxon observation"
                ),
            }
        )
    return output


def phase11_seed_novelty(
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for label in PHASE11_METHODS:
        for instance in INSTANCES:
            archives: dict[int, set[str]] = defaultdict(set)
            for row in rows:
                if (
                    row["method_or_variant"] == label
                    and row["instance"] == instance
                ):
                    archives[int(row["seed"])].add(row["layout_signature"])
            occurrences = Counter(
                sig for signatures in archives.values() for sig in signatures
            )
            for seed in SEEDS:
                signatures = archives[seed]
                if not signatures:
                    raise ValueError(
                        f"empty signature archive for "
                        f"{label}/{instance}/seed_{seed}"
                    )
                exclusive = sum(
                    occurrences[sig] == 1 for sig in signatures
                )
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
                        "seed_rediscovered_signature_ratio": (
                            rediscovered / len(signatures)
                        ),
                        "definition": (
                            "order-invariant within method and instance "
                            "across seeds 101-130"
                        ),
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
        running = max(
            running, min(1.0, (m - position) * float(p[index]))
        )
        adjusted_sorted.append(running)
    result = np.empty(m, dtype=float)
    for index, adjusted in zip(order, adjusted_sorted, strict=True):
        result[index] = adjusted
    return result.tolist()


def preferred_direction(metric_name: str) -> str:
    if metric_name in {
        "igd_plus",
        "runtime_seconds",
        "seed_rediscovered_signature_ratio",
    }:
        return "lower"
    if metric_name == "osd":
        return "descriptive_no_preferred_direction"
    return "higher"


def paired_test(
    first: Sequence[float],
    second: Sequence[float],
    metric_name: str,
) -> dict[str, Any]:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal lengths")
    if np.all(a == b):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
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
            ""
            if direction == "descriptive_no_preferred_direction"
            else rank_biserial
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
        key = (
            row["instance"],
            int(row["seed"]),
            row["method_or_variant"],
        )
        combined[key].update(
            {name: float(row[name]) for name in SUMMARY_METRICS}
        )
    for row in novelty_rows:
        key = (
            row["instance"],
            int(row["seed"]),
            row["method_or_variant"],
        )
        combined[key].update(
            {
                "seed_archive_signature_count": float(
                    row["seed_archive_signature_count"]
                ),
                "seed_exclusive_signature_count": float(
                    row["seed_exclusive_signature_count"]
                ),
                "seed_rediscovered_signature_ratio": float(
                    row["seed_rediscovered_signature_ratio"]
                ),
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
                block_keys = [
                    (instance, seed)
                    for instance in instances
                    for seed in SEEDS
                ]
                first = [
                    combined[(instance, seed, first_method)][metric_name]
                    for instance, seed in block_keys
                ]
                second = [
                    combined[(instance, seed, second_method)][metric_name]
                    for instance, seed in block_keys
                ]
                family.append(
                    {
                        "scope": (
                            "pooled_secondary_fixed_block"
                            if pooled
                            else "per_instance_primary"
                        ),
                        "instance": scope_name,
                        "metric": metric_name,
                        "method1": first_method,
                        "method2": second_method,
                        "pairing_key": (
                            "(instance, seed)" if pooled else "seed"
                        ),
                        "n_pairs": len(first),
                        "method1_mean": float(np.mean(first)),
                        "method2_mean": float(np.mean(second)),
                        **paired_test(first, second, metric_name),
                        "holm_family": (
                            "three Phase-11 method pairs within "
                            f"metric={metric_name}, instance={scope_name}"
                        ),
                        "independence_note": (
                            "120 observations are 4 instances x 30 seeds, "
                            "not 120 independent warehouses"
                            if pooled
                            else "30 matched seeds within one warehouse instance"
                        ),
                    }
                )
            adjusted = holm_adjust(
                [row["p_value_raw"] for row in family]
            )
            for row, p_adjusted in zip(
                family, adjusted, strict=True
            ):
                row["p_value_holm"] = p_adjusted
                output.append(row)
    return output


def v0_pairwise_statistics(
    seed_rows: list[dict[str, Any]],
    variants: Sequence[str],
    family_name: str,
) -> list[dict[str, Any]]:
    """Supporting V0-versus-variant tests; manuscript summary stays descriptive."""
    by_key = {
        (
            row["instance"],
            int(row["seed"]),
            row["method_or_variant"],
        ): row
        for row in seed_rows
    }
    output = []
    comparisons = tuple(variants[1:])
    for instance in INSTANCES:
        for metric_name in (
            "hypervolume",
            "igd_plus",
            "osd",
            "unique_layout_signature_count_rank_0_3",
            "runtime_seconds",
        ):
            family = []
            for variant in comparisons:
                first = [
                    float(
                        by_key[
                            (instance, seed, "V0_full_proposed")
                        ][metric_name]
                    )
                    for seed in SEEDS
                ]
                second = [
                    float(
                        by_key[(instance, seed, variant)][metric_name]
                    )
                    for seed in SEEDS
                ]
                family.append(
                    {
                        "scope": "per_instance_supporting",
                        "comparison_group": family_name,
                        "instance": instance,
                        "metric": metric_name,
                        "method1": "V0_full_proposed",
                        "method2": variant,
                        "n_pairs": len(SEEDS),
                        **paired_test(first, second, metric_name),
                        "holm_family": (
                            f"{len(comparisons)} V0-vs-variant "
                            f"comparisons within metric={metric_name}, "
                            f"instance={instance}"
                        ),
                    }
                )
            adjusted = holm_adjust(
                [row["p_value_raw"] for row in family]
            )
            for row, p_adjusted in zip(
                family, adjusted, strict=True
            ):
                row["p_value_holm"] = p_adjusted
                output.append(row)
    return output


def friedman_instance_mean_check(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for metric_name in (
        "hypervolume",
        "igd_plus",
        "unique_layout_signature_count_rank_0_3",
        "runtime_seconds",
    ):
        instance_means: dict[str, list[float]] = {
            method: [] for method in PHASE11_METHODS
        }
        for instance in INSTANCES:
            for method in PHASE11_METHODS:
                values = [
                    float(row[metric_name])
                    for row in seed_rows
                    if row["instance"] == instance
                    and row["method_or_variant"] == method
                ]
                instance_means[method].append(float(np.mean(values)))
        result = friedmanchisquare(
            *(instance_means[method] for method in PHASE11_METHODS)
        )
        output.append(
            {
                "metric": metric_name,
                "instance_blocks": len(INSTANCES),
                "method_count": len(PHASE11_METHODS),
                "methods": ";".join(PHASE11_METHODS),
                "friedman_chi_square": float(result.statistic),
                "p_value": float(result.pvalue),
                "interpretation_scope": (
                    "descriptive corroboration; only four warehouse-instance blocks"
                ),
            }
        )
    return output


def manuscript_summary_rows(
    overall: list[dict[str, Any]],
    signatures: list[dict[str, Any]],
    labels: Sequence[str],
    label_field: str,
) -> list[dict[str, Any]]:
    summary_map = {
        (row["method_or_variant"], row["metric"]): row
        for row in overall
    }
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
                (
                    label,
                    "unique_layout_signature_count_rank_0_3",
                )
            ]["mean"],
            "unique_per_run_std": summary_map[
                (
                    label,
                    "unique_layout_signature_count_rank_0_3",
                )
            ]["std"],
            "total_final_rank_0_3_archive_entries": sig[
                "total_archive_entries_rank_0_3"
            ],
            "unique_signatures_cross_seed": sig[
                "unique_signatures_cross_seed_rank_0_3"
            ],
            "cross_seed_rediscovery_ratio": sig[
                "cross_seed_rediscovery_ratio_rank_0_3"
            ],
            "runtime_mean_seconds": summary_map[
                (label, "runtime_seconds")
            ]["mean"],
            "runtime_std_seconds": summary_map[
                (label, "runtime_seconds")
            ]["std"],
        }
        if label == "V0_full_proposed":
            row["v0_source_note"] = (
                "reused Phase-11 proposed raw archives"
            )
        output.append(row)
    return output


def phase12c_depth_evidence(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize retained structural depths for V0/V6/V7.

    This is a final-ranked-archive diagnostic. It does not claim exact Beam
    Search stop reasons or frontier-level cap-hit counts.
    """
    seed_rows: list[dict[str, Any]] = []
    for label in PHASE12C_VARIANTS:
        for instance in INSTANCES:
            for seed in SEEDS:
                archive = [
                    row
                    for row in rows
                    if row["method_or_variant"] == label
                    and row["instance"] == instance
                    and row["seed"] == seed
                ]
                depths = [
                    int(row["depth"])
                    for row in archive
                    if row["depth"] is not None
                ]
                if not depths:
                    raise ValueError(
                        f"no retained depth values for "
                        f"{label}/{instance}/seed_{seed}"
                    )
                caps = {
                    int(row["configured_max_depth"])
                    for row in archive
                }
                if len(caps) != 1:
                    raise ValueError(
                        f"inconsistent max-depth metadata for "
                        f"{label}/{instance}/seed_{seed}: {caps}"
                    )
                cap = next(iter(caps))
                seed_rows.append(
                    {
                        "comparison_group": "phase12c",
                        "method_or_variant": label,
                        "instance": instance,
                        "seed": seed,
                        "configured_max_depth": cap,
                        "retained_archive_count": len(depths),
                        "mean_retained_depth": float(np.mean(depths)),
                        "median_retained_depth": float(np.median(depths)),
                        "min_retained_depth": min(depths),
                        "max_retained_depth": max(depths),
                        "retained_layouts_reaching_cap": sum(
                            depth == cap for depth in depths
                        ),
                        "run_reaches_configured_cap": max(depths) >= cap,
                        "scope_note": (
                            "final feasible rank 0-3 archive only; "
                            "not Beam Search stop-reason evidence"
                        ),
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    for label in PHASE12C_VARIANTS:
        for instance in INSTANCES:
            group = [
                row
                for row in seed_rows
                if row["method_or_variant"] == label
                and row["instance"] == instance
            ]
            caps = {int(row["configured_max_depth"]) for row in group}
            if len(caps) != 1:
                raise ValueError(
                    f"inconsistent Phase12C cap for {label}/{instance}"
                )
            max_depths = [
                float(row["max_retained_depth"]) for row in group
            ]
            summary_rows.append(
                {
                    "comparison_group": "phase12c",
                    "method_or_variant": label,
                    "instance": instance,
                    "configured_max_depth": next(iter(caps)),
                    "n_runs": len(group),
                    "mean_seed_max_retained_depth": float(
                        np.mean(max_depths)
                    ),
                    "median_seed_max_retained_depth": float(
                        np.median(max_depths)
                    ),
                    "min_seed_max_retained_depth": int(min(max_depths)),
                    "max_seed_max_retained_depth": int(max(max_depths)),
                    "runs_reaching_configured_cap": sum(
                        parse_bool(row["run_reaches_configured_cap"])
                        for row in group
                    ),
                    "scope_note": (
                        "final feasible rank 0-3 archive only; "
                        "exact Beam Search stop reasons are not logged here"
                    ),
                }
            )
    return seed_rows, summary_rows


def _canonical_config_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def configuration_snapshot(run_meta: dict[str, Any]) -> dict[str, str]:
    return {
        field: _canonical_config_value(run_meta.get(field))
        for field in V6B_CONFIG_FIELDS
    }


def candidate_cap_counts(
    path: Path,
    cap: int,
) -> dict[str, int]:
    rows = read_csv(path)
    at_cap = 0
    above_cap = 0
    safety_cap = 0
    for row in rows:
        depth_raw = row.get("depth", "")
        if depth_raw not in (None, ""):
            depth = int(float(depth_raw))
            at_cap += depth == cap
            above_cap += depth > cap
        safety_cap += parse_bool(row.get("safety_max_depth_reached"))
    return {
        "candidate_row_count": len(rows),
        "candidate_rows_at_cap": at_cap,
        "candidate_rows_above_cap": above_cap,
        "candidate_rows_safety_max_depth_reached": safety_cap,
    }


def verify_archive_grid_signatures(
    index_path: Path,
    archive_items: Sequence[dict[str, Any]],
    npz_path: Path,
) -> None:
    """Verify saved layout_signature against the exact archived 2-D grids."""
    with np.load(npz_path, allow_pickle=False) as layouts:
        available = set(layouts.files)
        for item in archive_items:
            key = str(
                item.get("layout_key")
                or item.get("archive_key")
                or ""
            )
            if not key or key not in available:
                raise KeyError(
                    f"layout key {key!r} from {index_path} "
                    f"not found in {npz_path}"
                )
            observed = exact_grid_signature(layouts[key])
            saved = str(item.get("layout_signature") or "")
            if observed != saved:
                raise ValueError(
                    f"exact-grid signature mismatch in {index_path}: "
                    f"saved={saved}, recomputed={observed}"
                )


def _validate_expected_run_parameters(
    label: str,
    run_meta: dict[str, Any],
    experiment_meta: dict[str, Any],
    source: Path,
) -> None:
    expected_depth = V6B_EXPECTED_MAX_DEPTH[label]
    actual = {
        "actual_population_size": int(
            run_meta.get("actual_population_size")
        ),
        "actual_generations": int(run_meta.get("actual_generations")),
        "actual_beam_width": int(run_meta.get("actual_beam_width")),
        "actual_max_depth": int(run_meta.get("actual_max_depth")),
    }
    expected_actual = {
        "actual_population_size": 10,
        "actual_generations": 15,
        "actual_beam_width": 3,
        "actual_max_depth": expected_depth,
    }
    if actual != expected_actual:
        raise ValueError(
            f"unexpected {label} parameters in {source}: "
            f"{actual} != {expected_actual}"
        )
    accepted_variants = (
        {None, "", "none", "V0_full_proposed"}
        if label == "V0_full_proposed"
        else {V6B_VARIANT}
    )
    if run_meta.get("ablation_variant") not in accepted_variants:
        raise ValueError(
            f"unexpected {label} ablation_variant in {source}: "
            f"{run_meta.get('ablation_variant')!r}"
        )
    for field, expected in V6B_EXPECTED_AUTO_PARAMETERS.items():
        if int(run_meta.get(field)) != expected:
            raise ValueError(
                f"unexpected {label} {field} in {source}: "
                f"{run_meta.get(field)!r} != {expected}"
            )
    exp_default = experiment_meta.get("default_parameters")
    if isinstance(exp_default, dict):
        if int(exp_default.get("max_depth")) != expected_depth:
            raise ValueError(
                f"experiment metadata max_depth mismatch for "
                f"{label}: {source}"
            )


def load_v6b_condition(
    label: str,
    results_root: Path,
    source_relative_root: str,
    manifest: dict[tuple[str, str, str, int], dict[str, str]],
    manifest_phase: str,
    manifest_label: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, str]],
    set[Path],
]:
    index_map = discover_indexes_for(
        results_root / source_relative_root,
        (V6B_INSTANCE,),
        SEEDS,
    )
    archive_records: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    configs: dict[int, dict[str, str]] = {}
    input_files: set[Path] = set()
    cap = V6B_EXPECTED_MAX_DEPTH[label]
    for seed in SEEDS:
        manifest_key = (
            manifest_phase,
            manifest_label,
            V6B_INSTANCE,
            seed,
        )
        if manifest_key not in manifest:
            raise KeyError(
                f"completed V6b diagnostic manifest row missing: "
                f"{manifest_key}"
            )
        manifest_row = manifest[manifest_key]
        runtime = float(manifest_row["runtime_seconds"])
        index_path = index_map[(V6B_INSTANCE, seed)]
        seed_dir = index_path.parent
        run_meta_path = seed_dir / "run_metadata.json"
        exp_meta_path = experiment_metadata_path(index_path)
        candidates_path = seed_dir / "candidates.csv"
        npz_path = seed_dir / "final_ranked_layouts.npz"
        required = (
            index_path,
            run_meta_path,
            exp_meta_path,
            candidates_path,
            npz_path,
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing V6b diagnostic inputs for {label}/seed_{seed}: "
                f"{missing}"
            )
        items = load_json(index_path)
        run_meta = load_json(run_meta_path)
        exp_meta = load_json(exp_meta_path)
        if run_meta.get("status") != "completed":
            raise ValueError(
                f"run metadata not completed: {run_meta_path}"
            )
        _validate_expected_run_parameters(
            label, run_meta, exp_meta, run_meta_path
        )
        configs[seed] = configuration_snapshot(run_meta)
        archive = [
            item
            for item in items
            if int(item.get("rank", 999)) <= 3
            and final_entry_is_feasible(item)
        ]
        if not archive:
            raise ValueError(
                f"empty V6b diagnostic archive: {index_path}"
            )
        verify_archive_grid_signatures(
            index_path, archive, npz_path
        )
        depths = []
        signatures = set()
        for item in archive:
            depth = int(item["depth"])
            if depth > cap:
                raise ValueError(
                    f"retained depth {depth} exceeds configured "
                    f"cap {cap}: {index_path}"
                )
            signature = str(item.get("layout_signature") or "")
            if not signature:
                raise ValueError(
                    f"blank layout signature: {index_path}"
                )
            depths.append(depth)
            signatures.add(signature)
            archive_records.append(
                {
                    "comparison_group": "v6b_demo_matched",
                    "method_or_variant": label,
                    "instance": V6B_INSTANCE,
                    "seed": seed,
                    "rank": int(item["rank"]),
                    "layout_signature": signature,
                    "objectives": objective(item),
                    "depth": depth,
                    "configured_max_depth": cap,
                    "runtime_seconds": runtime,
                    "source_path": index_path.as_posix(),
                }
            )
        cap_counts = candidate_cap_counts(candidates_path, cap)
        if cap_counts["candidate_rows_above_cap"]:
            raise ValueError(
                f"candidate depth above configured cap {cap}: "
                f"{candidates_path}"
            )
        binding_rows.append(
            {
                "comparison_group": "v6b_demo_matched",
                "method_or_variant": label,
                "instance": V6B_INSTANCE,
                "seed": seed,
                "configured_max_depth": cap,
                "observed_min_structural_depth": min(depths),
                "observed_max_structural_depth": max(depths),
                "retained_archive_count": len(archive),
                "retained_unique_signature_count": len(signatures),
                "retained_layouts_reaching_cap": sum(
                    depth == cap for depth in depths
                ),
                "run_reaches_configured_cap": max(depths) >= cap,
                "runtime_seconds": runtime,
                **cap_counts,
                "archive_index_path": index_path.as_posix(),
            }
        )
        input_files.update(required)
    return archive_records, binding_rows, configs, input_files


def validate_v6b_matched_configuration(
    v0_configs: dict[int, dict[str, str]],
    v6b_configs: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        first = v0_configs[seed]
        second = v6b_configs[seed]
        differing = [
            field
            for field in V6B_CONFIG_FIELDS
            if first[field] != second[field]
        ]
        matched = not differing
        rows.append(
            {
                "seed": seed,
                "matched_on_all_non_depth_scientific_fields": matched,
                "differing_fields": ";".join(differing),
                "v0_configured_max_depth": 28,
                "v6b_configured_max_depth": 10,
                "intended_difference": (
                    "configured max depth / ablation label only"
                ),
            }
        )
        if not matched:
            raise ValueError(
                f"V0/V6b configuration mismatch for seed {seed}: "
                f"{differing}"
            )
    return rows


def v6b_binding_summary(
    binding_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in binding_rows:
        by_label[row["method_or_variant"]].append(row)
    conditions: dict[str, Any] = {}
    for label in ("V0_full_proposed", V6B_VARIANT):
        rows = sorted(by_label[label], key=lambda item: int(item["seed"]))
        maxima = np.asarray(
            [float(row["observed_max_structural_depth"]) for row in rows]
        )
        conditions[label] = {
            "configured_max_depth": V6B_EXPECTED_MAX_DEPTH[label],
            "n_runs": len(rows),
            "mean_observed_max_structural_depth": float(np.mean(maxima)),
            "median_observed_max_structural_depth": float(np.median(maxima)),
            "min_observed_max_structural_depth": int(np.min(maxima)),
            "max_observed_max_structural_depth": int(np.max(maxima)),
            "runs_reaching_configured_cap": int(
                sum(
                    parse_bool(row["run_reaches_configured_cap"])
                    for row in rows
                )
            ),
            "candidate_rows_at_configured_cap": int(
                sum(int(row["candidate_rows_at_cap"]) for row in rows)
            ),
            "candidate_rows_above_configured_cap": int(
                sum(int(row["candidate_rows_above_cap"]) for row in rows)
            ),
        }
    by_seed = {
        (
            row["method_or_variant"],
            int(row["seed"]),
        ): row
        for row in binding_rows
    }
    v0_exceeds_ten = 0
    v6b_reaches_ten = 0
    paired_binding = 0
    for seed in SEEDS:
        v0_max = int(
            by_seed[
                ("V0_full_proposed", seed)
            ]["observed_max_structural_depth"]
        )
        v6b_max = int(
            by_seed[
                (V6B_VARIANT, seed)
            ]["observed_max_structural_depth"]
        )
        v0_gt = v0_max > 10
        v6b_hit = v6b_max >= 10
        v0_exceeds_ten += v0_gt
        v6b_reaches_ten += v6b_hit
        paired_binding += v0_gt and v6b_hit
    return {
        "instance": V6B_INSTANCE,
        "seeds": "101-130",
        "conditions": conditions,
        "paired_binding_evidence": {
            "v0_runs_with_retained_depth_above_10": v0_exceeds_ten,
            "v6b_runs_with_retained_depth_reaching_10": v6b_reaches_ten,
            "paired_seeds_v0_above_10_and_v6b_reaches_10": paired_binding,
            "binding_demonstrated": paired_binding > 0,
        },
        "scope_note": (
            "Demo-1-w2 matched diagnostic only. Binding is established "
            "from observed retained/candidate depth evidence, not from the "
            "configured Dmax value alone."
        ),
    }


def compute_v6b_indicators(
    archive_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    union = np.asarray(
        [row["objectives"] for row in archive_records], dtype=float
    )
    minima, maxima = union.min(axis=0), union.max(axis=0)
    normalized_union = normalize(union, minima, maxima)
    reference_front = nondominated_points(normalized_union)
    ranges = maxima - minima
    protocol = {
        "comparison_group": "v6b_demo_matched",
        "instance": V6B_INSTANCE,
        "conditions": ["V0_full_proposed", V6B_VARIANT],
        "objective_vector": "(N_locked, -N_pf, R_p); minimization",
        "archive_scope": "final feasible rank 0-3",
        "normalization_scope": (
            "Demo-1-w2 V0+V6b union only"
        ),
        "minima": minima.tolist(),
        "maxima": maxima.tolist(),
        "ranges": ranges.tolist(),
        "zero_range_replacement": 1,
        "reference_front_size": len(reference_front),
        "hv_reference_point": [1.1, 1.1, 1.1],
        "union_objectives_sha256": stable_array_hash(union),
        "reference_front_sha256": stable_array_hash(reference_front),
        "comparability_note": (
            "V6b indicators are comparable only within this "
            "V0-versus-V6b Demo comparison."
        ),
    }
    seed_rows = []
    for label in ("V0_full_proposed", V6B_VARIANT):
        for seed in SEEDS:
            archive = [
                row
                for row in archive_records
                if row["method_or_variant"] == label
                and row["seed"] == seed
            ]
            if not archive:
                raise ValueError(
                    f"missing V6b indicator archive: "
                    f"{label}/seed_{seed}"
                )
            points = np.asarray(
                [row["objectives"] for row in archive], dtype=float
            )
            normalized = normalize(points, minima, maxima)
            normalized_nd = nondominated_points(normalized)
            raw_nd = nondominated_points(points)
            signatures = {
                row["layout_signature"] for row in archive
            }
            seed_rows.append(
                {
                    "comparison_group": "v6b_demo_matched",
                    "instance": V6B_INSTANCE,
                    "method_or_variant": label,
                    "seed": seed,
                    "hypervolume": hypervolume(
                        normalized_nd, HV_REFERENCE
                    ),
                    "igd_plus": igd_plus(
                        normalized_nd, reference_front
                    ),
                    "osd": osd(normalized_nd),
                    "nondominated_solution_count": len(raw_nd),
                    "archive_size_rank_0_3": len(archive),
                    "unique_layout_signature_count_rank_0_3": len(
                        signatures
                    ),
                    "runtime_seconds": archive[0]["runtime_seconds"],
                }
            )
    summary_rows = []
    for label in ("V0_full_proposed", V6B_VARIANT):
        rows = [
            row
            for row in seed_rows
            if row["method_or_variant"] == label
        ]
        for metric_name in SUMMARY_METRICS:
            summary_rows.append(
                {
                    "comparison_group": "v6b_demo_matched",
                    "instance": V6B_INSTANCE,
                    "method_or_variant": label,
                    "metric": metric_name,
                    **quantile_summary(
                        [float(row[metric_name]) for row in rows]
                    ),
                    "sampling_unit": "matched seed",
                }
            )
    return seed_rows, summary_rows, protocol


def v6b_paired_statistics(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["method_or_variant"],
            int(row["seed"]),
        ): row
        for row in seed_rows
    }
    output = []
    for metric_name in (
        "hypervolume",
        "igd_plus",
        "osd",
        "unique_layout_signature_count_rank_0_3",
        "runtime_seconds",
    ):
        first = [
            float(
                by_key[("V0_full_proposed", seed)][metric_name]
            )
            for seed in SEEDS
        ]
        second = [
            float(by_key[(V6B_VARIANT, seed)][metric_name])
            for seed in SEEDS
        ]
        row = {
            "comparison_group": "v6b_demo_matched",
            "instance": V6B_INSTANCE,
            "metric": metric_name,
            "method1": "V0_full_proposed",
            "method2": V6B_VARIANT,
            "n_pairs": len(SEEDS),
            "multiplicity_note": (
                "single prespecified V0-vs-V6b contrast within this "
                "metric; no across-metric multiplicity adjustment"
            ),
        }
        if metric_name == "osd":
            row.update(
                {
                    "method1_mean": float(np.mean(first)),
                    "method2_mean": float(np.mean(second)),
                    "wilcoxon_statistic": "",
                    "p_value_raw": "",
                    "zero_method": "",
                    "alternative": "",
                    "preferred_direction": (
                        "descriptive_no_preferred_direction"
                    ),
                    "rank_biserial_method1_preferred": "",
                    "method1_wins": "",
                    "ties": "",
                    "method1_losses": "",
                    "method1_win_proportion": "",
                    "tie_proportion": "",
                    "method1_loss_proportion": "",
                    "raw_mean_difference_method1_minus_method2": (
                        float(np.mean(first) - np.mean(second))
                    ),
                    "test_status": "descriptive_only",
                }
            )
        else:
            row.update(
                {
                    "method1_mean": float(np.mean(first)),
                    "method2_mean": float(np.mean(second)),
                    **paired_test(first, second, metric_name),
                    "test_status": "paired_wilcoxon",
                }
            )
        output.append(row)
    return output


def analyze_v6b(
    main_results_root: Path,
    v6b_results_root: Path,
    output_dir: Path,
    main_manifest: dict[tuple[str, str, str, int], dict[str, str]],
    input_files: set[Path],
    all_provenance: list[dict[str, Any]],
    all_reference_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    v6b_manifest, v6b_manifest_path = load_completed_manifest(
        v6b_results_root
    )
    input_files.add(v6b_manifest_path)
    v0_records, v0_binding, v0_configs, v0_inputs = (
        load_v6b_condition(
            "V0_full_proposed",
            main_results_root,
            "p11/nsga2",
            main_manifest,
            "phase11",
            "proposed_nsga2_bs",
        )
    )
    v6b_records, v6b_binding, v6b_configs, v6b_inputs = (
        load_v6b_condition(
            V6B_VARIANT,
            v6b_results_root,
            "p12c/V6b_d10",
            v6b_manifest,
            "phase12c",
            V6B_VARIANT,
        )
    )
    input_files.update(v0_inputs)
    input_files.update(v6b_inputs)
    config_rows = validate_v6b_matched_configuration(
        v0_configs, v6b_configs
    )
    binding_rows = v0_binding + v6b_binding
    binding_summary = v6b_binding_summary(binding_rows)
    archive_records = v0_records + v6b_records
    seed_rows, summary_rows, protocol = compute_v6b_indicators(
        archive_records
    )
    paired = v6b_paired_statistics(seed_rows)

    write_csv(output_dir / "v6b_binding_by_seed.csv", binding_rows)
    write_csv(
        output_dir / "v6b_configuration_match_by_seed.csv",
        config_rows,
    )
    write_csv(
        output_dir / "v6b_indicator_seed_level.csv",
        seed_rows,
    )
    write_csv(
        output_dir / "v6b_indicator_summary.csv",
        summary_rows,
    )
    write_csv(
        output_dir / "v6b_paired_statistics.csv",
        paired,
    )
    (output_dir / "v6b_binding_summary.json").write_text(
        json.dumps(binding_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "v6b_indicator_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for label, root, binding in (
        ("V0_full_proposed", main_results_root, v0_binding),
        (V6B_VARIANT, v6b_results_root, v6b_binding),
    ):
        for row in binding:
            all_provenance.append(
                {
                    "comparison_group": "v6b_demo_matched",
                    "method_or_variant": label,
                    "source_phase": (
                        "phase11"
                        if label == "V0_full_proposed"
                        else "phase12c"
                    ),
                    "source_manifest_label": (
                        "proposed_nsga2_bs"
                        if label == "V0_full_proposed"
                        else V6B_VARIANT
                    ),
                    "source_relative_root": (
                        "p11/nsga2"
                        if label == "V0_full_proposed"
                        else "p12c/V6b_d10"
                    ),
                    "instance": V6B_INSTANCE,
                    "seed": row["seed"],
                    "archive_scope": "final feasible rank 0-3",
                    "archive_record_count": row[
                        "retained_archive_count"
                    ],
                    "unique_signature_count": row[
                        "retained_unique_signature_count"
                    ],
                    "configured_max_depth": row[
                        "configured_max_depth"
                    ],
                    "runtime_seconds": row["runtime_seconds"],
                    "archive_index_path": row[
                        "archive_index_path"
                    ],
                    "manifest_output_dir": "",
                    "source_campaign_root": str(root.resolve()),
                }
            )

    all_reference_metadata.append(
        {
            "comparison_group": "v6b_demo_matched",
            "instance": V6B_INSTANCE,
            "objective_vector": protocol["objective_vector"],
            "archive_scope": protocol["archive_scope"],
            "normalization_scope": protocol[
                "normalization_scope"
            ],
            "source_record_count": len(archive_records),
            "source_unique_objective_count": len(
                unique_rows(
                    np.asarray(
                        [
                            row["objectives"]
                            for row in archive_records
                        ],
                        dtype=float,
                    )
                )
            ),
            "min_N_locked": protocol["minima"][0],
            "max_N_locked": protocol["maxima"][0],
            "range_N_locked": protocol["ranges"][0],
            "min_negative_N_pf": protocol["minima"][1],
            "max_negative_N_pf": protocol["maxima"][1],
            "range_negative_N_pf": protocol["ranges"][1],
            "min_R_p": protocol["minima"][2],
            "max_R_p": protocol["maxima"][2],
            "range_R_p": protocol["ranges"][2],
            "zero_range_replacement": 1,
            "reference_front_size": protocol[
                "reference_front_size"
            ],
            "hv_reference_point": "(1.1,1.1,1.1)",
            "union_objectives_sha256": protocol[
                "union_objectives_sha256"
            ],
            "reference_front_sha256": protocol[
                "reference_front_sha256"
            ],
            "comparability_note": protocol[
                "comparability_note"
            ],
        }
    )

    return {
        "instance": V6B_INSTANCE,
        "matched_runs": len(SEEDS),
        "archive_records": len(archive_records),
        "configuration_match_rows": len(config_rows),
        "binding_demonstrated": binding_summary[
            "paired_binding_evidence"
        ]["binding_demonstrated"],
        "comparison_specific_reference_front_size": protocol[
            "reference_front_size"
        ],
    }


def input_hash_rows(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    output = []
    for path in sorted(
        {path.resolve() for path in paths},
        key=lambda item: str(item).lower(),
    ):
        output.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return output


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory must be absent or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_readme(
    results_root: Path,
    v6b_results_root: Path | None,
    phases: Sequence[str],
) -> str:
    v6b_note = (
        f"`{v6b_results_root.as_posix()}`"
        if v6b_results_root is not None
        else "not selected"
    )
    return f"""# Revision structural evidence

Generated by `python -m whl_experiments.analyze_revision_campaign_evidence`.

Main source campaign: `{results_root.as_posix()}`

V6b source campaign: {v6b_note}

Selected main comparison groups: {", ".join(phases)}.

## Comparison families

- Phase 11: Proposed / BS-only / RRBS.
- Phase 12B: V0--V5. V0 reuses the completed Phase-11 Proposed raw archives.
- Phase 12C: V0 / V6 / V7. V0 again reuses the same raw Phase-11 Proposed
  archives, but all indicators are recomputed inside the separate V0/V6/V7
  normalization/reference union.
- V6b: Demo-1-w2 only, matched V0-versus-V6b diagnostic. It uses a separate
  V0+V6b normalization/reference union and is not numerically comparable with
  the Phase-11, Phase-12B, or Phase-12C indicator values.

## Indicator protocol

- Final feasible Pareto-rank 0--3 archive records only.
- Minimization vector `(N_locked, -N_pf, R_p)`.
- Min--max normalization is separate for each fixed instance and comparison
  group. A zero observed range is replaced by one.
- The empirical reference front is the unique nondominated normalized union.
- HV reference point: `(1.1,1.1,1.1)`.
- IGD+: positive-part minimization distance.
- OSD: mean pairwise Euclidean distance among normalized nondominated points;
  descriptive, with no preferred direction.
- Structural identity uses the saved exact-grid `layout_signature`.

## Statistical protocol

Phase-11 primary tests are per instance with 30 matched seeds. The three method
pairs form the Holm family within each metric/instance. Two-sided Wilcoxon
signed-rank uses `zero_method=wilcox`; paired rank-biserial effects are oriented
to method 1. The 4 x 30 pooled analysis is secondary fixed-block evidence only.
The four-instance Friedman result is descriptive corroboration.

Phase-12B and Phase-12C V0-versus-variant test files are supporting evidence;
their manuscript summary tables remain descriptive unless explicitly cited.

V6b is a single prespecified matched Demo-1-w2 contrast. OSD is descriptive;
the other V0-versus-V6b metrics use paired two-sided Wilcoxon tests. The V6b
configuration audit requires the two conditions to match on the stored
non-depth scientific configuration fields. Exact archived grids are also
rehashed and checked against their saved structural signatures.

## Depth diagnostics

`phase12c_depth_*` and `v6b_binding_*` use retained final feasible rank-0--3
depths. V6b additionally checks all candidate rows for depths above the
configured cap. These files do not claim Beam Search stop reasons that were not
logged.

## Safety

This analyzer performs post-processing only and never invokes an optimizer.
The output directory must be absent or empty.
"""


def analyze(
    results_root: Path,
    output_dir: Path,
    phases: Sequence[str],
    *,
    v6b_results_root: Path | None = None,
) -> dict[str, Any]:
    prepare_output(output_dir)
    manifest, manifest_path = load_completed_manifest(results_root)
    input_files: set[Path] = {manifest_path}
    all_provenance: list[dict[str, Any]] = []
    all_reference_metadata: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "results_root": str(results_root.resolve()),
        "v6b_results_root": (
            str(v6b_results_root.resolve())
            if v6b_results_root is not None
            else None
        ),
        "output_dir": str(output_dir.resolve()),
        "phases": list(phases),
        "post_processing_only": True,
    }
    for phase in phases:
        records, provenance, phase_inputs = load_phase_records(
            results_root, phase, manifest
        )
        input_files.update(phase_inputs)
        all_provenance.extend(provenance)
        seed_rows, reference_metadata = compute_indicators(
            records, phase
        )
        all_reference_metadata.extend(reference_metadata)
        by_instance = summarize_seed_rows(
            seed_rows, by_instance=True
        )
        overall = summarize_seed_rows(
            seed_rows, by_instance=False
        )
        signatures = signature_summary(records, phase)
        write_csv(
            output_dir / f"{phase}_seed_level.csv", seed_rows
        )
        write_csv(
            output_dir / f"{phase}_summary_by_instance.csv",
            by_instance,
        )
        write_csv(
            output_dir / f"{phase}_summary_overall.csv", overall
        )
        write_csv(
            output_dir / f"{phase}_signature_summary.csv",
            signatures,
        )
        phase_summary: dict[str, Any] = {
            "logical_runs": len(seed_rows),
            "archive_records": len(records),
            "methods_or_variants": len(
                {row["method_or_variant"] for row in seed_rows}
            ),
        }
        if phase == "phase11":
            novelty = phase11_seed_novelty(records)
            stats_by_instance = phase11_statistics(
                seed_rows, novelty, pooled=False
            )
            stats_pooled = phase11_statistics(
                seed_rows, novelty, pooled=True
            )
            friedman = friedman_instance_mean_check(seed_rows)
            table5 = manuscript_summary_rows(
                overall,
                signatures,
                PHASE11_METHODS,
                "method",
            )
            write_csv(
                output_dir / "phase11_seed_novelty.csv",
                novelty,
            )
            write_csv(
                output_dir / "phase11_stats_by_instance.csv",
                stats_by_instance,
            )
            write_csv(
                output_dir / "phase11_stats_pooled.csv",
                stats_pooled,
            )
            write_csv(
                output_dir / "phase11_friedman_instance_means.csv",
                friedman,
            )
            write_csv(
                output_dir / "table5_phase11_manuscript_values.csv",
                table5,
            )
            phase_summary["primary_statistical_rows"] = len(
                stats_by_instance
            )
        elif phase == "phase12b":
            pairwise = v0_pairwise_statistics(
                seed_rows,
                PHASE12B_VARIANTS,
                "phase12b",
            )
            table7 = manuscript_summary_rows(
                overall,
                signatures,
                PHASE12B_VARIANTS,
                "variant",
            )
            write_csv(
                output_dir / "phase12b_v0_pairwise_stats.csv",
                pairwise,
            )
            write_csv(
                output_dir / "table7_phase12b_manuscript_values.csv",
                table7,
            )
            phase_summary["v0_source"] = (
                "phase11/proposed_nsga2_bs"
            )
        elif phase == "phase12c":
            pairwise = v0_pairwise_statistics(
                seed_rows,
                PHASE12C_VARIANTS,
                "phase12c",
            )
            table8 = manuscript_summary_rows(
                overall,
                signatures,
                PHASE12C_VARIANTS,
                "variant",
            )
            depth_seed, depth_summary = phase12c_depth_evidence(
                records
            )
            write_csv(
                output_dir / "phase12c_v0_pairwise_stats.csv",
                pairwise,
            )
            write_csv(
                output_dir / "table8_phase12c_manuscript_values.csv",
                table8,
            )
            write_csv(
                output_dir / "phase12c_depth_by_seed.csv",
                depth_seed,
            )
            write_csv(
                output_dir / "phase12c_depth_summary.csv",
                depth_summary,
            )
            phase_summary["v0_source"] = (
                "phase11/proposed_nsga2_bs"
            )
            phase_summary["depth_diagnostic_rows"] = len(
                depth_seed
            )
        summary[phase] = phase_summary

    if v6b_results_root is not None:
        summary["v6b_demo_matched"] = analyze_v6b(
            results_root,
            v6b_results_root,
            output_dir,
            manifest,
            input_files,
            all_provenance,
            all_reference_metadata,
        )

    write_csv(
        output_dir / "input_manifest.csv", all_provenance
    )
    write_csv(
        output_dir / "indicator_reference_metadata.csv",
        all_reference_metadata,
    )
    write_csv(
        output_dir / "input_hashes.csv",
        input_hash_rows(input_files),
    )
    (output_dir / "README.md").write_text(
        build_readme(
            results_root, v6b_results_root, phases
        ),
        encoding="utf-8",
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/revision_final_30seed_nofg"),
    )
    parser.add_argument(
        "--v6b-results-root",
        type=Path,
        default=Path(
            "results/revision_final_30seed_nofg_v6b"
        ),
        help=(
            "Separate completed V6b campaign root. "
            "Use --skip-v6b to omit the matched diagnostic."
        ),
    )
    parser.add_argument(
        "--skip-v6b",
        action="store_true",
        help="Skip the separate matched V0-versus-V6b Demo diagnostic.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/reproducibility/"
            "revision_final_30seed_nofg/structural"
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=tuple(PHASE_SOURCES),
        default=["phase11", "phase12b", "phase12c"],
        help=(
            "Completed main comparison groups to analyze. "
            "Default: phase11 phase12b phase12c."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        args.results_root,
        args.output_dir,
        args.phases,
        v6b_results_root=(
            None if args.skip_v6b else args.v6b_results_root
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
