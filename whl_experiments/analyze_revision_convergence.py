"""Regenerate final-clean IJPR convergence evidence from saved generation_objectives.csv.

This module is POST-PROCESSING ONLY. It never invokes an optimizer.

It reconstructs cumulative best-so-far hypervolume trajectories for the two
representative manuscript instances using the SAME comparison-specific
normalization/reference point as the final structural evidence:

* Phase 11  -> Proposed / BS-only / RRBS
* Phase 12B -> V0--V5
* Phase 12C -> V0 / V6 / V7

The normalization bounds are read from the already-generated
indicator_reference_metadata.csv. This prevents accidental cross-campaign
normalization and keeps Figure 7 / Table 11 consistent with Tables 5--8.

V6b is intentionally excluded: it is a separate Demo-only binding-depth
diagnostic and is not part of Figure 7 / Table 11.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from whl_experiments.analyze_revision_campaign_evidence import (
    HV_REFERENCE,
    PHASE_SOURCES,
    SEEDS,
    hypervolume,
    nondominated_points,
    normalize,
    read_csv,
    write_csv,
)

REPRESENTATIVE_INSTANCES = (
    "AT_S_comercial_layout_AW_3",
    "Gyorgy-KOVACS_WH_Narrow_AW_4",
)

INSTANCE_LABELS = {
    "AT_S_comercial_layout_AW_3": "Atefeh",
    "Gyorgy-KOVACS_WH_Narrow_AW_4": "Kov-1-O-w4",
}

METHOD_LABELS = {
    "proposed_nsga2_bs": "Prop. (Reference)",
    "bs_only_direct": "BS-only",
    "random_restart_bs": "RRBS",
    "V0_full_proposed": "Prop. (Reference)",
    "V1_fixed_sorting": "V1 fixed sorting",
    "V2_fixed_weights": "V2 fixed weights",
    "V3_uniform_mutation": "V3 uniform mutation",
    "V4_no_symmetry_breaking": "V4 no symmetry breaking",
    "V5_random_feasible_start_spacing": "V5 random feasible start",
    "V6_depth15_beam_default": "Depth 15",
    "V7_beam_plus1_depth_default": "Beam +1",
}

TABLE11_PHASES = ("phase11", "phase12c")
VALIDATION_TOLERANCE = 1e-8


def _parse_int(value: Any) -> int:
    if value in (None, ""):
        raise ValueError("missing integer value")
    return int(float(str(value).strip()))


def _parse_float(value: Any) -> float:
    if value in (None, ""):
        raise ValueError("missing float value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def _objective_from_generation_row(row: dict[str, Any]) -> tuple[float, float, float]:
    """Return minimization vector (N_locked, -N_pf, R_p)."""
    return (
        _parse_float(row["interior_storage"]),
        -_parse_float(row["pick_faces"]),
        _parse_float(row["retrieval_penalty"]),
    )


def _discover_generation_files(
    source_root: Path,
    instance: str,
    seeds: Sequence[int] = SEEDS,
) -> dict[int, Path]:
    """Find exactly one generation_objectives.csv for each requested seed."""
    mapping: dict[int, Path] = {}
    for path in sorted(source_root.rglob("generation_objectives.csv")):
        seed_dir = path.parent
        instance_dir = seed_dir.parent
        if instance_dir.name != instance:
            continue
        if not seed_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed not in seeds:
            continue
        if seed in mapping:
            raise ValueError(
                f"duplicate generation_objectives.csv under {source_root} "
                f"for {instance}, seed {seed}: {mapping[seed]} and {path}"
            )
        mapping[seed] = path

    expected = set(int(seed) for seed in seeds)
    found = set(mapping)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f"generation-objective coverage mismatch under {source_root} "
            f"for {instance}; missing={missing}, extra={extra}"
        )
    return mapping


def _load_reference_metadata(
    structural_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    path = structural_dir / "indicator_reference_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the unified structural evidence analyzer first."
        )

    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv(path):
        group = row.get("comparison_group", "")
        instance = row.get("instance", "")
        if group not in {"phase11", "phase12b", "phase12c"}:
            continue
        if instance not in REPRESENTATIVE_INSTANCES:
            continue
        key = (group, instance)
        if key in mapping:
            raise ValueError(f"duplicate reference metadata row: {key}")

        minima = np.asarray(
            [
                _parse_float(row["min_N_locked"]),
                _parse_float(row["min_negative_N_pf"]),
                _parse_float(row["min_R_p"]),
            ],
            dtype=float,
        )
        maxima = np.asarray(
            [
                _parse_float(row["max_N_locked"]),
                _parse_float(row["max_negative_N_pf"]),
                _parse_float(row["max_R_p"]),
            ],
            dtype=float,
        )
        if np.any(maxima < minima):
            raise ValueError(f"invalid normalization bounds for {key}")

        mapping[key] = {
            "minima": minima,
            "maxima": maxima,
            "normalization_scope": row.get("normalization_scope", ""),
            "hv_reference_point": row.get("hv_reference_point", ""),
            "union_objectives_sha256": row.get("union_objectives_sha256", ""),
            "reference_front_sha256": row.get("reference_front_sha256", ""),
        }

    expected = {
        (phase, instance)
        for phase in ("phase11", "phase12b", "phase12c")
        for instance in REPRESENTATIVE_INSTANCES
    }
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        raise ValueError(f"reference metadata coverage incomplete: missing={missing}")
    return mapping


def _load_seed_level_hv(
    structural_dir: Path,
    phase: str,
) -> dict[tuple[str, str, int], float]:
    path = structural_dir / f"{phase}_seed_level.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing structural seed-level evidence: {path}")
    mapping: dict[tuple[str, str, int], float] = {}
    for row in read_csv(path):
        instance = row.get("instance", "")
        if instance not in REPRESENTATIVE_INSTANCES:
            continue
        key = (
            row["method_or_variant"],
            instance,
            _parse_int(row["seed"]),
        )
        if key in mapping:
            raise ValueError(f"duplicate seed-level HV row in {path}: {key}")
        mapping[key] = _parse_float(row["hypervolume"])
    return mapping


def _trajectory_from_file(
    path: Path,
    minima: np.ndarray,
    maxima: np.ndarray,
) -> list[tuple[int, float, int]]:
    """Build cumulative best-so-far HV from logged rank-0/selected objective rows."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty generation objectives: {path}")

    points_by_index: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        generation = _parse_int(row["generation"])
        points_by_index[generation].append(_objective_from_generation_row(row))

    if not points_by_index:
        raise ValueError(f"no objective rows found in {path}")

    cumulative: list[tuple[float, float, float]] = []
    trajectory: list[tuple[int, float, int]] = []
    for generation in sorted(points_by_index):
        cumulative.extend(points_by_index[generation])
        raw_front = nondominated_points(np.asarray(cumulative, dtype=float))
        norm_front = normalize(raw_front, minima, maxima)
        hv = hypervolume(norm_front, HV_REFERENCE)
        trajectory.append((generation, float(hv), int(len(raw_front))))
    return trajectory


def _carry_forward(
    raw_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align each method/instance trajectory over its observed index range.

    If a saved trajectory terminates earlier than another seed, its final best-so-far
    HV is carried forward. This keeps the plotted mean based on all 30 seeds.
    """
    grouped: dict[tuple[str, str, str], dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    meta: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}

    for row in raw_rows:
        key = (row["phase"], row["method_or_variant"], row["instance"])
        seed = int(row["seed"])
        index = int(row["index"])
        grouped[key][seed][index] = float(row["hypervolume"])
        meta[(key[0], key[1], key[2], seed, index)] = row

    aligned: list[dict[str, Any]] = []
    for key, seed_maps in sorted(grouped.items()):
        phase, method, instance = key
        if set(seed_maps) != set(SEEDS):
            raise ValueError(
                f"incomplete seed trajectories for {key}: "
                f"{sorted(set(SEEDS) - set(seed_maps))}"
            )
        max_index = max(max(values) for values in seed_maps.values())
        min_index = min(min(values) for values in seed_maps.values())

        for seed in SEEDS:
            series = seed_maps[seed]
            first = min(series)
            last_hv: float | None = None
            for index in range(min_index, max_index + 1):
                if index in series:
                    last_hv = series[index]
                if index < first or last_hv is None:
                    continue
                aligned.append(
                    {
                        "phase": phase,
                        "method_or_variant": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "instance": instance,
                        "instance_label": INSTANCE_LABELS[instance],
                        "seed": int(seed),
                        "index": int(index),
                        "hypervolume": float(last_hv),
                        "carried_forward": bool(index not in series),
                    }
                )
    return aligned


def _summary_rows(aligned_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    carried: dict[tuple[str, str, str, int], int] = defaultdict(int)

    for row in aligned_rows:
        key = (
            row["phase"],
            row["method_or_variant"],
            row["instance"],
            int(row["index"]),
        )
        grouped[key].append(float(row["hypervolume"]))
        carried[key] += int(bool(row["carried_forward"]))

    rows: list[dict[str, Any]] = []
    for (phase, method, instance, index), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "phase": phase,
                "method_or_variant": method,
                "method_label": METHOD_LABELS.get(method, method),
                "instance": instance,
                "instance_label": INSTANCE_LABELS[instance],
                "index": int(index),
                "n": int(len(arr)),
                "hv_mean": float(np.mean(arr)),
                "hv_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "hv_median": float(np.median(arr)),
                "hv_q25": float(np.quantile(arr, 0.25)),
                "hv_q75": float(np.quantile(arr, 0.75)),
                "hv_min": float(np.min(arr)),
                "hv_max": float(np.max(arr)),
                "n_carried_forward": int(carried[(phase, method, instance, index)]),
            }
        )
    return rows


def _table11_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["phase"] not in TABLE11_PHASES:
            continue
        grouped[(row["phase"], row["method_or_variant"], row["instance"])].append(row)

    output: list[dict[str, Any]] = []
    for (phase, method, instance), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["index"]))
        initial = rows[0]
        final = rows[-1]
        output.append(
            {
                "section": (
                    "Baseline comparison" if phase == "phase11" else "Sensitivity"
                ),
                "comparison_group": phase,
                "instance": INSTANCE_LABELS[instance],
                "instance_internal": instance,
                "method_or_variant": method,
                "method_label": METHOD_LABELS.get(method, method),
                "initial_hv_mean": float(initial["hv_mean"]),
                "initial_hv_std": float(initial["hv_std"]),
                "final_hv_mean": float(final["hv_mean"]),
                "final_hv_std": float(final["hv_std"]),
                "hv_gain_mean": float(final["hv_mean"]) - float(initial["hv_mean"]),
                "final_index": int(final["index"]),
                "n_final": int(final["n"]),
                "shaded_band": "mean ± 1 SD across seeds",
            }
        )
    return output


def _validate_final_hv(
    aligned_rows: Sequence[dict[str, Any]],
    structural_dir: Path,
) -> list[dict[str, Any]]:
    expected_by_phase = {
        phase: _load_seed_level_hv(structural_dir, phase)
        for phase in ("phase11", "phase12b", "phase12c")
    }

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aligned_rows:
        grouped[
            (
                row["phase"],
                row["method_or_variant"],
                row["instance"],
                int(row["seed"]),
            )
        ].append(row)

    checks: list[dict[str, Any]] = []
    for (phase, method, instance, seed), rows in sorted(grouped.items()):
        final = max(rows, key=lambda row: int(row["index"]))
        expected_key = (method, instance, seed)
        if expected_key not in expected_by_phase[phase]:
            raise ValueError(
                f"missing final structural HV for {phase}/{method}/{instance}/seed_{seed}"
            )
        expected = float(expected_by_phase[phase][expected_key])
        observed = float(final["hypervolume"])
        delta = observed - expected
        checks.append(
            {
                "phase": phase,
                "method_or_variant": method,
                "instance": instance,
                "seed": seed,
                "final_index": int(final["index"]),
                "convergence_final_hv": observed,
                "structural_final_hv": expected,
                "difference": delta,
                "abs_difference": abs(delta),
                "within_tolerance": abs(delta) <= VALIDATION_TOLERANCE,
            }
        )
    return checks


def _write_plot_script_hint(output_dir: Path) -> None:
    text = """# Figure 7 plotting note

Use `convergence_summary.csv` to rebuild Figure 7.

- Panel (a): `phase11`
- Panel (b): `phase12b`
- Panel (c): `phase12c`
- Use only the two representative instances:
  `AT_S_comercial_layout_AW_3` (Atefeh) and
  `Gyorgy-KOVACS_WH_Narrow_AW_4` (Kov-1-O-w4).
- Plot `hv_mean` against `index`.
- Shade `hv_mean ± hv_std` (one standard deviation across the 30 seeds).
- Do NOT compare numerical HV values across Phase11/Phase12B/Phase12C:
  each panel uses its own comparison-specific normalization.
- BS-only has only a single direct-search archive point and therefore no
  iterative outer-search trajectory.
"""
    (output_dir / "FIGURE7_README.md").write_text(text, encoding="utf-8")


def analyze(
    *,
    results_root: Path,
    structural_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    results_root = results_root.resolve()
    structural_dir = structural_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_reference_metadata(structural_dir)

    raw_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for phase in ("phase11", "phase12b", "phase12c"):
        for source in PHASE_SOURCES[phase]:
            source_root = results_root / source.relative_root
            if not source_root.exists():
                raise FileNotFoundError(
                    f"Missing raw campaign root for {phase}/{source.logical_label}: "
                    f"{source_root}"
                )

            for instance in REPRESENTATIVE_INSTANCES:
                ref = metadata[(phase, instance)]
                files = _discover_generation_files(source_root, instance)

                for seed in SEEDS:
                    path = files[seed]
                    trajectory = _trajectory_from_file(
                        path,
                        ref["minima"],
                        ref["maxima"],
                    )
                    source_files.append(
                        {
                            "phase": phase,
                            "method_or_variant": source.logical_label,
                            "instance": instance,
                            "seed": int(seed),
                            "generation_objectives_csv": str(path),
                            "row_count": len(read_csv(path)),
                        }
                    )
                    for index, hv, front_size in trajectory:
                        raw_rows.append(
                            {
                                "phase": phase,
                                "method_or_variant": source.logical_label,
                                "method_label": METHOD_LABELS.get(
                                    source.logical_label, source.logical_label
                                ),
                                "instance": instance,
                                "instance_label": INSTANCE_LABELS[instance],
                                "seed": int(seed),
                                "index": int(index),
                                "hypervolume": float(hv),
                                "cumulative_nondominated_objectives": int(front_size),
                                "normalization_scope": ref["normalization_scope"],
                                "hv_reference_point": "(1.1,1.1,1.1)",
                                "source_csv": str(path),
                            }
                        )

    aligned_rows = _carry_forward(raw_rows)
    summary_rows = _summary_rows(aligned_rows)
    table11_rows = _table11_rows(summary_rows)
    validation_rows = _validate_final_hv(aligned_rows, structural_dir)

    failures = [row for row in validation_rows if not row["within_tolerance"]]

    write_csv(output_dir / "convergence_run_hv_raw.csv", raw_rows)
    write_csv(output_dir / "convergence_run_hv_aligned.csv", aligned_rows)
    write_csv(output_dir / "convergence_summary.csv", summary_rows)
    write_csv(output_dir / "table11_convergence_summary.csv", table11_rows)
    write_csv(output_dir / "convergence_final_hv_validation.csv", validation_rows)
    write_csv(output_dir / "convergence_input_files.csv", source_files)

    reference_rows = []
    for (phase, instance), ref in sorted(metadata.items()):
        reference_rows.append(
            {
                "comparison_group": phase,
                "instance": instance,
                "min_N_locked": float(ref["minima"][0]),
                "max_N_locked": float(ref["maxima"][0]),
                "min_negative_N_pf": float(ref["minima"][1]),
                "max_negative_N_pf": float(ref["maxima"][1]),
                "min_R_p": float(ref["minima"][2]),
                "max_R_p": float(ref["maxima"][2]),
                "hv_reference_point": "(1.1,1.1,1.1)",
                "normalization_scope": ref["normalization_scope"],
                "union_objectives_sha256": ref["union_objectives_sha256"],
                "reference_front_sha256": ref["reference_front_sha256"],
            }
        )
    write_csv(output_dir / "convergence_reference_metadata.csv", reference_rows)
    _write_plot_script_hint(output_dir)

    summary = {
        "post_processing_only": True,
        "results_root": str(results_root),
        "structural_dir": str(structural_dir),
        "output_dir": str(output_dir),
        "representative_instances": list(REPRESENTATIVE_INSTANCES),
        "seeds": f"{min(SEEDS)}-{max(SEEDS)}",
        "comparison_groups": {
            "phase11": [source.logical_label for source in PHASE_SOURCES["phase11"]],
            "phase12b": [source.logical_label for source in PHASE_SOURCES["phase12b"]],
            "phase12c": [source.logical_label for source in PHASE_SOURCES["phase12c"]],
        },
        "raw_trajectory_rows": len(raw_rows),
        "aligned_trajectory_rows": len(aligned_rows),
        "summary_rows": len(summary_rows),
        "table11_rows": len(table11_rows),
        "final_hv_validation_rows": len(validation_rows),
        "final_hv_validation_failures": len(failures),
        "validation_tolerance": VALIDATION_TOLERANCE,
        "figure7_band": "mean ± 1 SD across 30 seeds",
        "normalization_rule": (
            "Fixed per-instance, per-comparison-group bounds from "
            "structural/indicator_reference_metadata.csv; no cross-campaign HV comparison."
        ),
    }
    (output_dir / "convergence_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failures:
        preview = failures[:10]
        raise RuntimeError(
            "Convergence final-HV validation failed against structural seed-level evidence. "
            f"failures={len(failures)}; first={preview}. "
            "Do not use Figure 7/Table 11 until this mismatch is resolved."
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate final-clean Figure 7 / Table 11 convergence evidence "
            "from saved generation_objectives.csv files."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Final-clean structural campaign root, e.g. results/revision_final_30seed_nofg",
    )
    parser.add_argument(
        "--structural-dir",
        type=Path,
        required=True,
        help=(
            "Existing unified structural evidence directory containing "
            "indicator_reference_metadata.csv and phase*_seed_level.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for compact convergence reproducibility evidence.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = analyze(
        results_root=args.results_root,
        structural_dir=args.structural_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
