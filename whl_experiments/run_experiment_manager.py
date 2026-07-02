"""Config-driven experiment manager skeleton for proposed NSGA-II + Beam Search."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised implicitly when PyYAML is installed.
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from whl_core.blocks import detect_storage_blocks
from whl_core.layout_io import load_mask, mask_to_grid
from whl_core.paths import CONFIG_DIR, MASK_DIR, PROJECT_ROOT, RESULTS_DIR
from whl_core.scoring import compute_aisle_mask, compute_storage_mask, detect_pick_faces
from whl_algorithms.parameter_policy import auto_hyperparams
from whl_experiments import run_bs_only_direct as bs_only_direct
from whl_experiments import run_nsga2_bs as nsga2_bs
from whl_experiments import run_random_restart_bs as random_restart_bs

METHOD_NAME = "proposed_nsga2_bs"
BS_ONLY_DIRECT_METHOD_NAME = bs_only_direct.METHOD_NAME
RANDOM_RESTART_BS_METHOD_NAME = random_restart_bs.METHOD_NAME
SUPPORTED_METHODS = (METHOD_NAME, BS_ONLY_DIRECT_METHOD_NAME, RANDOM_RESTART_BS_METHOD_NAME)
ADAPTIVE_SPACING_MODE = "feasible_start_adaptive_spacing"
ADAPTIVE_SPACING_ALPHA = 0.5
ADAPTIVE_SPACING_BF = None
AblationVariant = str
DEFAULT_ABLATION_VARIANT = "none"
AUTO_PARAMS_USED = False
BUDGET_POLICIES = ("fixed", "auto_from_instance")
DEFAULT_BUDGET_POLICY = "auto_from_instance"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "experiment_plan.yaml"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "experiments"

DEFAULT_PARAMETERS = {
    "population_size": 6,
    "generations": 3,
    "beam_width": 3,
    "max_depth": 8,
}

DEFAULT_BS_ONLY_DIRECT_PARAMETERS = {
    "population_size": 0,
    "generations": 1,
    "beam_width": 3,
    "max_depth": 50,
}

DEFAULT_RANDOM_RESTART_BS_PARAMETERS = {
    "population_size": random_restart_bs.DEFAULT_POPULATION_SIZE,
    "generations": random_restart_bs.DEFAULT_GENERATIONS,
    "beam_width": 3,
    "max_depth": 10,
}

ARCHIVE_LAYOUT_MODES = (
    "none",
    "generation_elites",
    "final_ranked",
    "both",
    "all_debug",
    "all_candidates_debug",
)

CANDIDATE_COLUMNS = [
    "run_id",
    "method",
    "instance",
    "seed",
    "generation",
    "budget_policy",
    "auto_population_size",
    "auto_generations",
    "auto_beam_width",
    "auto_max_depth",
    "auto_decode_budget",
    "restart_index",
    "batch_index",
    "within_batch_index",
    "decode_budget",
    "candidate_id",
    "parent_chromosome_id",
    "depth",
    "trace",
    "status",
    "rank",
    "crowding_distance",
    "selected",
    "storage_total",
    "pick_faces",
    "interior_storage",
    "retrieval_penalty",
    "door_connectivity_index",
    "access_anchor_connectivity_index",
    "has_door_connected_aisle",
    "has_access_anchor_connected_aisle",
    "aisle_components",
    "anchor_connected_components",
    "unanchored_aisle_components",
    "single_aisle_component",
    "access_network_components",
    "aisle_access_network_components",
    "unreachable_aisle_components",
    "unreachable_aisle_cells",
    "has_access_anchor_reachable_aisle_network",
    "exact_width_ok",
    "exact_width_violation_count",
    "chromosome_h_active_count",
    "chromosome_v_active_count",
    "active_h_count",
    "active_v_count",
    "chromosome_index",
    "sorting_rule",
    "sorting_rule_mode",
    "bs_rule_policy",
    "bs_weight_policy",
    "uses_scalar_score",
    "rho",
    "beam_w1",
    "beam_w2",
    "beam_lambda",
    "adaptive_weight_mode",
    "mutation_operator",
    "initialization_mode",
    "initialization_spacing_mode",
    "adaptive_spacing_used",
    "feasible_h_start_count",
    "feasible_v_start_count",
    "adaptive_spacing_mode",
    "adaptive_spacing_alpha",
    "adaptive_spacing_bf",
    "auto_params_used",
    "parameter_source",
    "beta_h",
    "beta_v",
    "h_active_starts",
    "v_active_starts",
    "terminal",
    "safety_max_depth_reached",
    "feasible",
    "feasibility_reason",
    "storage_block_count",
    "chromosome_signature",
    "layout_signature",
]

GENERATION_SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "instance",
    "seed",
    "generation",
    "chromosome_count",
    "decoded_candidate_count",
    "feasible_candidate_count",
    "non_dominated_count",
    "selected_survivor_count",
    "best_pick_faces",
    "best_storage_total",
    "best_interior_storage",
    "best_retrieval_penalty",
    "mean_pick_faces",
    "mean_storage_total",
    "mean_interior_storage",
    "mean_retrieval_penalty",
    "runtime_seconds",
]

BS_DEPTH_SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "instance",
    "seed",
    "sorting_rule",
    "depth",
    "input_node_count",
    "generated_child_count",
    "unique_child_count",
    "retained_node_count",
    "terminal_node_count",
    "safety_max_depth_reached",
]

EXPERIMENT_SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "instance",
    "seed",
    "budget_policy",
    "auto_params_used",
    "auto_population_size",
    "auto_generations",
    "auto_beam_width",
    "auto_max_depth",
    "auto_decode_budget",
    "population_size",
    "generations",
    "beam_width",
    "max_depth",
    "adaptive_spacing_mode",
    "adaptive_spacing_alpha",
    "adaptive_spacing_bf",
    "parameter_source",
    "beta_h",
    "beta_v",
    "total_candidates",
    "final_generation_candidates",
    "final_generation_rank0_count",
    "final_generation_selected_count",
    "final_best_pick_faces",
    "final_best_storage_total",
    "final_best_interior_storage",
    "final_best_retrieval_penalty",
    "total_runtime_seconds",
    "status",
    "error_message",
]

PROFILE_TIME_KEYS = [
    "initialization_time_seconds",
    "nsga_selection_time_seconds",
    "nsga_crossover_time_seconds",
    "nsga_mutation_time_seconds",
    "nsga_operator_time_seconds",
    "nsga_survivor_selection_time_seconds",
    "beam_decode_time_seconds",
    "beam_expansion_time_seconds",
    "beam_child_generation_time_seconds",
    "feasibility_filter_time_seconds",
    "objective_evaluation_time_seconds",
    "nondominated_sorting_time_seconds",
    "crowding_distance_time_seconds",
    "archive_dedup_time_seconds",
    "archive_write_time_seconds",
    "io_write_time_seconds",
    "other_time_seconds",
]

RUNTIME_PROFILE_SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "method_or_variant",
    "instance",
    "seed",
    "total_runtime_seconds",
    *PROFILE_TIME_KEYS,
    "ranking_archive_time_seconds",
    "io_time_seconds",
    "profile_granularity",
    "profiling_note",
]

GENERATION_PROFILE_COLUMNS = [
    "run_id",
    "method",
    "method_or_variant",
    "instance",
    "seed",
    "generation",
    "generation_runtime_seconds",
    "population_size",
    "candidate_count",
    "decoded_count",
    "feasible_count",
    "infeasible_count",
    "selected_count",
    "rank0_count",
    "unique_layout_signature_count",
    "nsga_operator_time_seconds",
    "beam_decode_time_seconds",
    "beam_expansion_time_seconds",
    "feasibility_filter_time_seconds",
    "objective_evaluation_time_seconds",
    "ranking_archive_time_seconds",
    "io_time_seconds",
]

GENERATION_OBJECTIVE_COLUMNS = [
    "run_id",
    "method",
    "method_or_variant",
    "instance",
    "seed",
    "generation",
    "candidate_id",
    "rank",
    "layout_signature",
    "interior_storage",
    "retrieval_penalty",
    "pick_faces",
    "door_connectivity_index",
    "source",
]


@dataclass(frozen=True)
class ExperimentInstance:
    """Resolved instance reference for one planned run."""

    name: str
    mask_path: Path


@dataclass(frozen=True)
class PlannedRun:
    """One instance-seed-method combination."""

    run_id: str
    method: str
    instance: ExperimentInstance
    seed: int
    parameters: dict[str, int]
    parameter_source: str = "manager_defaults"
    budget_policy: str = DEFAULT_BUDGET_POLICY
    auto_parameters: dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentManagerResult:
    """Return value for manager runs and dry-runs."""

    experiment_id: str
    output_dir: Path
    planned_runs: list[PlannedRun]
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False


def utc_timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load experiment configuration.")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_experiment_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the main experiment config plus adjacent helper configs."""
    path = Path(config_path)
    config = _load_yaml_file(path)
    config_dir = path.parent
    for key, filename in (
        ("instances_config", "instances.yaml"),
        ("methods_config", "methods.yaml"),
        ("seeds_config", "seeds.yaml"),
        ("objective_sets_config", "objective_sets.yaml"),
    ):
        helper_path = config_dir / filename
        config[key] = _load_yaml_file(helper_path)
    return config


def _split_csv_values(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        values: list[str] = []
        for item in value:
            values.extend(_split_csv_values(str(item)))
        return values
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_instance_name(path_or_name: str) -> str:
    stem = Path(path_or_name).stem if path_or_name else "instance"
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return safe or "instance"


def _mask_path_from_spec(spec: str | Path) -> Path:
    path = Path(spec)
    if path.exists() or path.is_absolute() or path.parent != Path("."):
        return path
    filename = str(spec)
    if not filename.endswith(".npz"):
        filename = f"{filename}.npz"
    return MASK_DIR / filename


def _instances_from_config(config: dict[str, Any]) -> list[ExperimentInstance]:
    raw_instances = config.get("instances_config", {}).get("instances", [])
    instances: list[ExperimentInstance] = []
    if not isinstance(raw_instances, list):
        return instances

    for item in raw_instances:
        if isinstance(item, str):
            path = _mask_path_from_spec(item)
            instances.append(ExperimentInstance(name=_safe_instance_name(item), mask_path=path))
            continue
        if not isinstance(item, dict):
            continue
        raw_path = (
            item.get("mask_path")
            or item.get("path")
            or item.get("filename")
            or item.get("file")
        )
        if raw_path is None:
            continue
        name = str(item.get("name") or _safe_instance_name(str(raw_path)))
        instances.append(ExperimentInstance(name=name, mask_path=_mask_path_from_spec(raw_path)))
    return instances


def resolve_instances(
    config: dict[str, Any],
    instances_arg: str | list[str] | None = None,
) -> list[ExperimentInstance]:
    """Resolve CLI/config/local mask instances."""
    requested = _split_csv_values(instances_arg)
    if requested:
        return [
            ExperimentInstance(
                name=_safe_instance_name(item),
                mask_path=_mask_path_from_spec(item),
            )
            for item in requested
        ]

    configured = _instances_from_config(config)
    if configured:
        return configured

    masks = nsga2_bs.discover_instance_masks(limit=None)
    return [
        ExperimentInstance(name=path.stem, mask_path=path)
        for path in masks
    ]


def resolve_seeds(config: dict[str, Any], seeds_arg: str | None = None) -> list[int]:
    """Resolve CLI/config seed list with a small fallback."""
    requested = _split_csv_values(seeds_arg)
    if requested:
        return [int(seed) for seed in requested]

    raw_seeds = config.get("seeds_config", {}).get("seeds", [])
    if isinstance(raw_seeds, list) and raw_seeds:
        return [int(seed) for seed in raw_seeds]
    return [1]


def _method_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_methods = config.get("methods_config", {}).get("methods", [])
    return raw_methods if isinstance(raw_methods, list) else []


def _parameter_defaults_for_method(method: str) -> dict[str, int]:
    if method == BS_ONLY_DIRECT_METHOD_NAME:
        return DEFAULT_BS_ONLY_DIRECT_PARAMETERS
    if method == RANDOM_RESTART_BS_METHOD_NAME:
        return DEFAULT_RANDOM_RESTART_BS_PARAMETERS
    return DEFAULT_PARAMETERS


def resolve_parameter_source(
    config: dict[str, Any],
    method: str = METHOD_NAME,
    overrides: dict[str, int | None] | None = None,
    budget_policy: str = DEFAULT_BUDGET_POLICY,
) -> str:
    """Return whether core search parameters came from defaults, config, or CLI."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")
    if budget_policy == "auto_from_instance":
        return "auto_from_instance"
    if any(value is not None for value in (overrides or {}).values()):
        return "cli"

    defaults = _parameter_defaults_for_method(method)
    for entry in _method_entries(config):
        if not isinstance(entry, dict) or entry.get("name") != method:
            continue
        raw_params = entry.get("parameters", {})
        if isinstance(raw_params, dict) and any(key in raw_params for key in defaults):
            return "config"
    return "manager_defaults"


def manual_budget_overrides(overrides: dict[str, int | None] | None) -> dict[str, int]:
    """Return explicitly provided manual budget values."""
    return {
        key: int(value)
        for key, value in (overrides or {}).items()
        if value is not None
    }

def apply_parameter_overrides(
    parameters: dict[str, int],
    overrides: dict[str, int | None] | None,
) -> dict[str, int]:
    """Apply explicitly provided CLI budget overrides."""
    updated = dict(parameters)
    for key, value in (overrides or {}).items():
        if value is not None:
            updated[key] = int(value)
    return updated


def validate_parameters(parameters: dict[str, int], method: str) -> None:
    """Validate resolved search budget parameters."""
    for key, value in parameters.items():
        if method == BS_ONLY_DIRECT_METHOD_NAME and key == "population_size":
            if value < 0:
                raise ValueError(f"{key} must be non-negative.")
            continue
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

def _auto_parameters_for_instance(
    instance: ExperimentInstance,
    method: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Resolve auto budget values and map them to one method's parameters."""
    masks = load_mask(instance.mask_path)
    grid = mask_to_grid(masks)
    auto = auto_hyperparams(*grid.shape)
    auto_values = {
        "population_size": int(auto["population_size"]),
        "generations": int(auto["generations"]),
        "beam_width": int(auto["beam_width"]),
        "max_depth": int(auto["max_depth"]),
    }
    auto_values["decode_budget"] = (
        auto_values["population_size"] * auto_values["generations"]
    )

    if method == BS_ONLY_DIRECT_METHOD_NAME:
        parameters = {
            "population_size": 0,
            "generations": 1,
            "beam_width": auto_values["beam_width"],
            "max_depth": auto_values["max_depth"],
        }
    elif method == RANDOM_RESTART_BS_METHOD_NAME:
        parameters = {
            "population_size": auto_values["population_size"],
            "generations": auto_values["generations"],
            "beam_width": auto_values["beam_width"],
            "max_depth": auto_values["max_depth"],
            "decode_budget": auto_values["decode_budget"],
        }
    else:
        parameters = {
            "population_size": auto_values["population_size"],
            "generations": auto_values["generations"],
            "beam_width": auto_values["beam_width"],
            "max_depth": auto_values["max_depth"],
        }
    return parameters, auto_values


def resolve_parameters(
    config: dict[str, Any],
    method: str = METHOD_NAME,
    overrides: dict[str, int | None] | None = None,
) -> dict[str, int]:
    """Resolve method defaults and CLI overrides."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")

    defaults = _parameter_defaults_for_method(method)
    parameters: dict[str, int] = dict(defaults)
    for entry in _method_entries(config):
        if not isinstance(entry, dict) or entry.get("name") != method:
            continue
        raw_params = entry.get("parameters", {})
        if isinstance(raw_params, dict):
            for key in defaults:
                if key in raw_params:
                    parameters[key] = int(raw_params[key])

    parameters = apply_parameter_overrides(parameters, overrides)
    validate_parameters(parameters, method)
    return parameters


def make_run_id(method: str, instance_name: str, seed: int) -> str:
    return f"{method}_{instance_name}_seed_{int(seed)}"


def build_plan(
    config: dict[str, Any],
    instances_arg: str | list[str] | None = None,
    seeds_arg: str | None = None,
    method: str = METHOD_NAME,
    overrides: dict[str, int | None] | None = None,
    budget_policy: str = DEFAULT_BUDGET_POLICY,
    beam_width_delta: int = 0,
) -> list[PlannedRun]:
    """Create planned instance-seed runs."""
    if budget_policy not in BUDGET_POLICIES:
        raise ValueError(f"budget_policy must be one of {BUDGET_POLICIES}.")
    instances = resolve_instances(config, instances_arg)
    seeds = resolve_seeds(config, seeds_arg)
    if budget_policy == "auto_from_instance":
        if beam_width_delta < 0:
            raise ValueError("beam_width_delta must be non-negative.")
    else:
        if beam_width_delta:
            raise ValueError("beam_width_delta requires budget_policy=auto_from_instance.")
        parameters = resolve_parameters(config, method, overrides)
    parameter_source = resolve_parameter_source(
        config,
        method,
        overrides,
        budget_policy=budget_policy,
    )
    planned: list[PlannedRun] = []
    for instance in instances:
        auto_parameters: dict[str, int] = {}
        if budget_policy == "auto_from_instance":
            parameters, auto_parameters = _auto_parameters_for_instance(instance, method)
            parameters = apply_parameter_overrides(parameters, overrides)
            if beam_width_delta:
                parameters["beam_width"] = int(parameters["beam_width"]) + int(beam_width_delta)
            validate_parameters(parameters, method)
        for seed in seeds:
            planned.append(
                PlannedRun(
                    run_id=make_run_id(method, instance.name, int(seed)),
                    method=method,
                    instance=instance,
                    seed=int(seed),
                    parameters=dict(parameters),
                    parameter_source=parameter_source,
                    budget_policy=budget_policy,
                    auto_parameters=dict(auto_parameters),
                )
            )
    return planned


def ensure_experiment_dirs(experiment_dir: Path) -> Path:
    """Create the top-level experiment directory."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "runs").mkdir(parents=True, exist_ok=True)
    return experiment_dir


def run_output_dir(experiment_dir: Path, run: PlannedRun) -> Path:
    return experiment_dir / "runs" / run.instance.name / f"seed_{run.seed}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def _write_json_list(path: Path, data: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def _best_value(rows: list[dict[str, Any]], key: str, *, maximize: bool) -> Any:
    values = [row.get(key) for row in rows if row.get(key) != ""]
    if not values:
        return ""
    numeric = [float(value) for value in values]
    best = max(numeric) if maximize else min(numeric)
    return int(best) if float(best).is_integer() else best


def _mean_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in rows if row.get(key) != ""]
    if not values:
        return ""
    return float(np.mean([float(value) for value in values]))


def _first_row_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in {"", None}:
            return value
    return ""


def _observed_decode_metadata_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorting_rules = sorted(
        {
            str(row.get("sorting_rule"))
            for row in rows
            if row.get("sorting_rule") not in {"", None}
        }
    )
    weight_modes = sorted(
        {
            str(row.get("adaptive_weight_mode"))
            for row in rows
            if row.get("adaptive_weight_mode") not in {"", None}
        }
    )
    rows_with_weight_fields = sum(
        1
        for row in rows
        if all(row.get(key) not in {"", None} for key in ("beam_w1", "beam_w2", "beam_lambda"))
    )
    if not rows:
        weight_fields_available = ""
    elif rows_with_weight_fields == len(rows):
        weight_fields_available = "all_candidate_rows"
    elif rows_with_weight_fields:
        weight_fields_available = "some_candidate_rows"
    else:
        weight_fields_available = "none"
    return {
        "observed_sorting_rules": sorting_rules,
        "observed_weight_mode": weight_modes[0] if len(weight_modes) == 1 else weight_modes,
        "observed_weight_fields_available": weight_fields_available,
        "observed_scalar_score_candidate_count": sum(
            1 for row in rows if str(row.get("uses_scalar_score")).lower() == "true"
        ),
        "observed_scalar_score_usage": sorted(
            {
                str(row.get("uses_scalar_score")).lower()
                for row in rows
                if row.get("uses_scalar_score") not in {"", None}
            }
        ),
        "observed_mutation_operators": sorted(
            {
                str(row.get("mutation_operator"))
                for row in rows
                if row.get("mutation_operator") not in {"", None}
            }
        ),
        "observed_weight_samples_summary": _weight_samples_summary(rows),
    }


def _weight_samples_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = {"beam_w1": [], "beam_w2": [], "beam_lambda": []}
    for row in rows:
        for key in values_by_key:
            value = row.get(key)
            if value in {"", None}:
                continue
            try:
                values_by_key[key].append(float(value))
            except (TypeError, ValueError):
                continue
    summary: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if not values:
            summary[key] = {}
            continue
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "unique_count": len(set(values)),
        }
    return summary


def _bs_policy_metadata_fields(
    method: str,
    bs_rule_policy: str,
    bs_weight_policy: str,
) -> dict[str, Any]:
    if method == BS_ONLY_DIRECT_METHOD_NAME:
        return {
            "bs_policy_scope": "bs_only_direct",
            "bs_rule_policy": bs_rule_policy,
            "bs_weight_policy": bs_weight_policy,
            "bs_rule_policy_for_bs_only_direct": bs_rule_policy,
            "bs_weight_policy_for_bs_only_direct": bs_weight_policy,
        }
    return {
        "bs_policy_scope": "not_applicable_for_method",
        "bs_rule_policy": None,
        "bs_weight_policy": None,
        "bs_rule_policy_for_bs_only_direct": bs_rule_policy,
        "bs_weight_policy_for_bs_only_direct": bs_weight_policy,
    }


def _auto_field(run: PlannedRun, key: str) -> Any:
    return run.auto_parameters.get(key, "") if run.auto_parameters else ""


def budget_fields_for_run(run: PlannedRun) -> dict[str, Any]:
    """Return budget-policy metadata fields for CSV rows and JSON metadata."""
    auto_used = run.budget_policy == "auto_from_instance"
    return {
        "budget_policy": run.budget_policy,
        "auto_params_used": bool(auto_used),
        "auto_population_size": _auto_field(run, "population_size"),
        "auto_generations": _auto_field(run, "generations"),
        "auto_beam_width": _auto_field(run, "beam_width"),
        "auto_max_depth": _auto_field(run, "max_depth"),
        "auto_decode_budget": _auto_field(run, "decode_budget"),
    }


def generation_summary_row(
    run: PlannedRun,
    generation_summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one generation summary row from candidate-level rows."""
    return {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        "generation": generation_summary["generation"],
        "chromosome_count": generation_summary["chromosome_count"],
        "decoded_candidate_count": generation_summary["decoded_candidate_count"],
        "feasible_candidate_count": generation_summary["feasible_candidate_count"],
        "non_dominated_count": generation_summary["non_dominated_count"],
        "selected_survivor_count": generation_summary["selected_survivor_count"],
        "best_pick_faces": _best_value(candidate_rows, "pick_faces", maximize=True),
        "best_storage_total": _best_value(candidate_rows, "storage_total", maximize=True),
        "best_interior_storage": _best_value(candidate_rows, "interior_storage", maximize=False),
        "best_retrieval_penalty": _best_value(candidate_rows, "retrieval_penalty", maximize=False),
        "mean_pick_faces": _mean_value(candidate_rows, "pick_faces"),
        "mean_storage_total": _mean_value(candidate_rows, "storage_total"),
        "mean_interior_storage": _mean_value(candidate_rows, "interior_storage"),
        "mean_retrieval_penalty": _mean_value(candidate_rows, "retrieval_penalty"),
        "runtime_seconds": generation_summary["runtime_seconds"],
    }


def _empty_profile_times() -> dict[str, float]:
    return {key: 0.0 for key in PROFILE_TIME_KEYS}


def _add_profile_time(profile_times: dict[str, float], key: str, elapsed: float) -> None:
    profile_times[key] = float(profile_times.get(key, 0.0)) + max(0.0, float(elapsed))


def _effective_method_or_variant(run: PlannedRun, ablation_variant: str) -> str:
    if ablation_variant and ablation_variant != DEFAULT_ABLATION_VARIANT:
        return ablation_variant
    return run.method


def _is_selected_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _is_feasible_row(row: dict[str, Any]) -> bool:
    feasible_value = row.get("feasible")
    if isinstance(feasible_value, str) and feasible_value:
        return feasible_value.lower() == "true"
    if feasible_value is True:
        return True
    status = str(row.get("status") or "").lower()
    if status == "infeasible":
        return False
    return row.get("rank") not in {"", None}


def _profile_output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "runtime_profile_summary_csv": run_dir / "runtime_profile_summary.csv",
        "generation_profile_csv": run_dir / "generation_profile.csv",
        "generation_objectives_csv": run_dir / "generation_objectives.csv",
    }


def _generation_profile_row(
    *,
    run: PlannedRun,
    ablation_variant: str,
    generation: int,
    generation_runtime_seconds: float,
    population_size: int,
    decoded_count: int,
    candidate_rows: list[dict[str, Any]],
    profile_times: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile_times = profile_times or _empty_profile_times()
    feasible_count = sum(1 for row in candidate_rows if _is_feasible_row(row))
    selected_count = sum(1 for row in candidate_rows if _is_selected_value(row.get("selected")))
    rank0_count = sum(1 for row in candidate_rows if str(row.get("rank")) == "0")
    layout_signatures = {
        str(row.get("layout_signature"))
        for row in candidate_rows
        if row.get("layout_signature") not in {"", None}
    }
    ranking_archive_time = (
        float(profile_times.get("nsga_survivor_selection_time_seconds", 0.0))
        + float(profile_times.get("nondominated_sorting_time_seconds", 0.0))
        + float(profile_times.get("crowding_distance_time_seconds", 0.0))
        + float(profile_times.get("archive_dedup_time_seconds", 0.0))
        + float(profile_times.get("archive_write_time_seconds", 0.0))
    )
    return {
        "run_id": run.run_id,
        "method": run.method,
        "method_or_variant": _effective_method_or_variant(run, ablation_variant),
        "instance": run.instance.name,
        "seed": run.seed,
        "generation": int(generation),
        "generation_runtime_seconds": float(generation_runtime_seconds),
        "population_size": int(population_size),
        "candidate_count": len(candidate_rows),
        "decoded_count": int(decoded_count),
        "feasible_count": feasible_count,
        "infeasible_count": len(candidate_rows) - feasible_count,
        "selected_count": selected_count,
        "rank0_count": rank0_count,
        "unique_layout_signature_count": len(layout_signatures),
        "nsga_operator_time_seconds": float(profile_times.get("nsga_operator_time_seconds", 0.0)),
        "beam_decode_time_seconds": float(profile_times.get("beam_decode_time_seconds", 0.0)),
        "beam_expansion_time_seconds": float(
            profile_times.get("beam_expansion_time_seconds", 0.0)
        ),
        "feasibility_filter_time_seconds": float(
            profile_times.get("feasibility_filter_time_seconds", 0.0)
        ),
        "objective_evaluation_time_seconds": float(
            profile_times.get("objective_evaluation_time_seconds", 0.0)
        ),
        "ranking_archive_time_seconds": ranking_archive_time,
        "io_time_seconds": float(profile_times.get("io_write_time_seconds", 0.0)),
    }


def _generation_profile_rows_from_existing(
    run: PlannedRun,
    ablation_variant: str,
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_generation: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        if row.get("generation") in {"", None}:
            continue
        rows_by_generation.setdefault(int(row["generation"]), []).append(row)
    profile_rows: list[dict[str, Any]] = []
    for summary in generation_rows:
        generation = int(summary["generation"])
        profile_rows.append(
            _generation_profile_row(
                run=run,
                ablation_variant=ablation_variant,
                generation=generation,
                generation_runtime_seconds=float(summary.get("runtime_seconds") or 0.0),
                population_size=int(run.parameters.get("population_size", 0)),
                decoded_count=int(summary.get("decoded_candidate_count") or 0),
                candidate_rows=rows_by_generation.get(generation, []),
            )
        )
    return profile_rows


def _generation_objective_rows(
    run: PlannedRun,
    ablation_variant: str,
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rank0_rows = [row for row in candidate_rows if str(row.get("rank")) == "0"]
    selected_rows = [row for row in candidate_rows if _is_selected_value(row.get("selected"))]
    source_rows = rank0_rows or selected_rows
    source = "rank0" if rank0_rows else "selected"
    objective_rows: list[dict[str, Any]] = []
    for row in source_rows:
        objective_rows.append(
            {
                "run_id": run.run_id,
                "method": run.method,
                "method_or_variant": _effective_method_or_variant(run, ablation_variant),
                "instance": run.instance.name,
                "seed": run.seed,
                "generation": row.get("generation", ""),
                "candidate_id": row.get("candidate_id", ""),
                "rank": row.get("rank", ""),
                "layout_signature": row.get("layout_signature", ""),
                "interior_storage": row.get("interior_storage", ""),
                "retrieval_penalty": row.get("retrieval_penalty", ""),
                "pick_faces": row.get("pick_faces", ""),
                "door_connectivity_index": row.get("door_connectivity_index", ""),
                "source": source,
            }
        )
    return objective_rows


def _runtime_profile_summary_row(
    *,
    run: PlannedRun,
    ablation_variant: str,
    total_runtime_seconds: float,
    profile_times: dict[str, float],
) -> dict[str, Any]:
    row_times = _empty_profile_times()
    for key in PROFILE_TIME_KEYS:
        row_times[key] = float(profile_times.get(key, 0.0))
    timed_keys = [key for key in PROFILE_TIME_KEYS if key != "other_time_seconds"]
    row_times["other_time_seconds"] = max(
        0.0,
        float(total_runtime_seconds) - sum(row_times[key] for key in timed_keys),
    )
    ranking_archive_time = (
        row_times["nsga_survivor_selection_time_seconds"]
        + row_times["nondominated_sorting_time_seconds"]
        + row_times["crowding_distance_time_seconds"]
        + row_times["archive_dedup_time_seconds"]
        + row_times["archive_write_time_seconds"]
    )
    return {
        "run_id": run.run_id,
        "method": run.method,
        "method_or_variant": _effective_method_or_variant(run, ablation_variant),
        "instance": run.instance.name,
        "seed": run.seed,
        "total_runtime_seconds": float(total_runtime_seconds),
        **row_times,
        "ranking_archive_time_seconds": ranking_archive_time,
        "io_time_seconds": row_times["io_write_time_seconds"],
        "profile_granularity": "light",
        "profiling_note": (
            "Light timing only; proposed runs split Beam Search expansion, "
            "feasibility checks, and objective metric extraction when profile-light is enabled. "
            "Nondominated sorting and crowding remain grouped with survivor/archive timing."
        ),
    }


def _write_profile_outputs(
    *,
    run: PlannedRun,
    run_dir: Path,
    ablation_variant: str,
    total_runtime_seconds: float,
    profile_times: dict[str, float],
    generation_profile_rows: list[dict[str, Any]],
    generation_objective_rows: list[dict[str, Any]],
    profile_light: bool,
    save_generation_objectives: bool,
) -> None:
    paths = _profile_output_paths(run_dir)
    if profile_light:
        _write_csv(
            paths["runtime_profile_summary_csv"],
            [
                _runtime_profile_summary_row(
                    run=run,
                    ablation_variant=ablation_variant,
                    total_runtime_seconds=total_runtime_seconds,
                    profile_times=profile_times,
                )
            ],
            RUNTIME_PROFILE_SUMMARY_COLUMNS,
        )
        _write_csv(
            paths["generation_profile_csv"],
            generation_profile_rows,
            GENERATION_PROFILE_COLUMNS,
        )
    if save_generation_objectives:
        _write_csv(
            paths["generation_objectives_csv"],
            generation_objective_rows,
            GENERATION_OBJECTIVE_COLUMNS,
        )


def _candidate_row_with_method(row: dict[str, Any], method: str) -> dict[str, Any]:
    updated = dict(row)
    updated["method"] = method
    updated["selected"] = bool(row.get("status") == "selected")
    return updated


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _candidate_chromosome_counts(candidate: Any) -> tuple[Any, Any]:
    chromosome = getattr(candidate, "chromosome", None)
    if chromosome is None:
        return "", ""
    return chromosome.active_count()


def _candidate_chromosome_signature(candidate: Any) -> str:
    chromosome = getattr(candidate, "chromosome", None)
    if chromosome is None:
        return ""
    return nsga2_bs.chromosome_signature_text(chromosome)


def _candidate_archive_index_entry(
    *,
    archive_key: str,
    run: PlannedRun,
    candidate: nsga2_bs.LayoutCandidate,
    generation: int | None,
    rank: int | None = None,
    crowding_distance: float | None = None,
) -> dict[str, Any]:
    h_count, v_count = _candidate_chromosome_counts(candidate)
    layout = np.asarray(candidate.node.layout)
    blocks = detect_storage_blocks(layout)
    pick_faces = detect_pick_faces(layout, blocks)
    pick_face_mask = np.zeros(layout.shape, dtype=bool)
    for row, col in pick_faces:
        pick_face_mask[row, col] = True
    storage_mask = compute_storage_mask(layout)
    aisle_mask = compute_aisle_mask(layout)
    feature_keys = {
        "pick_face_mask": f"{archive_key}__pick_face_mask",
        "storage_mask": f"{archive_key}__storage_mask",
        "aisle_mask": f"{archive_key}__aisle_mask",
    }
    selected_rank = candidate.rank if rank is None else rank
    selected_crowding = (
        candidate.crowding_distance
        if crowding_distance is None
        else crowding_distance
    )
    metrics = {
        "storage_total": candidate.metrics.get("storage_total"),
        "pick_faces": candidate.metrics.get("pick_faces"),
        "interior_storage": candidate.metrics.get("interior_storage"),
        "retrieval_penalty": candidate.metrics.get("retrieval_penalty"),
        "door_connectivity_index": candidate.metrics.get("door_connectivity_index"),
        "access_anchor_connectivity_index": candidate.metrics.get(
            "access_anchor_connectivity_index"
        ),
        "has_door_connected_aisle": candidate.metrics.get("has_door_connected_aisle"),
        "has_access_anchor_connected_aisle": candidate.metrics.get(
            "has_access_anchor_connected_aisle"
        ),
        "aisle_components": candidate.metrics.get("aisle_components"),
        "anchor_connected_components": candidate.metrics.get(
            "anchor_connected_components"
        ),
        "unanchored_aisle_components": candidate.metrics.get(
            "unanchored_aisle_components"
        ),
        "single_aisle_component": candidate.metrics.get("single_aisle_component"),
        "access_network_components": candidate.metrics.get("access_network_components"),
        "aisle_access_network_components": candidate.metrics.get(
            "aisle_access_network_components"
        ),
        "unreachable_aisle_components": candidate.metrics.get(
            "unreachable_aisle_components"
        ),
        "unreachable_aisle_cells": candidate.metrics.get("unreachable_aisle_cells"),
        "has_access_anchor_reachable_aisle_network": candidate.metrics.get(
            "has_access_anchor_reachable_aisle_network"
        ),
        "exact_width_ok": not candidate.exact_width_violations,
        "exact_width_violation_count": len(candidate.exact_width_violations),
    }
    stored_pick_face_count = int(np.count_nonzero(pick_face_mask))
    metrics_pick_faces = metrics.get("pick_faces")
    if metrics_pick_faces is not None and stored_pick_face_count != int(metrics_pick_faces):
        print(
            "WARNING: archive pick-face mask count does not match candidate metrics: "
            f"archive_key={archive_key}, mask_count={stored_pick_face_count}, "
            f"metrics_pick_faces={metrics_pick_faces}"
        )
    return {
        "archive_key": archive_key,
        "layout_key": archive_key,
        "feature_keys": feature_keys,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        "generation": generation,
        "candidate_id": int(candidate.candidate_id),
        "rank": None if selected_rank is None else int(selected_rank),
        "crowding_distance": _json_safe_value(selected_crowding),
        "selected": bool(candidate.selected),
        "trace": " > ".join(str(item) for item in candidate.node.trace),
        "action": candidate.node.action,
        "depth": int(candidate.node.depth),
        "layout_signature": nsga2_bs._signature_digest(
            nsga2_bs.layout_signature(candidate.node.layout)
        ),
        "chromosome_signature": _candidate_chromosome_signature(candidate),
        "chromosome_h_active_count": _json_safe_value(h_count),
        "chromosome_v_active_count": _json_safe_value(v_count),
        "decode_metadata": {
            key: _json_safe_value(value)
            for key, value in candidate.decode_metadata.items()
        },
        "metrics": {key: _json_safe_value(value) for key, value in metrics.items()},
        "stored_pick_face_count": stored_pick_face_count,
        "storage_mask_count": int(np.count_nonzero(storage_mask)),
        "aisle_mask_count": int(np.count_nonzero(aisle_mask)),
        "block_count": int(len(blocks)),
        "blocks": [
            {
                "id": int(block.id),
                "bbox": [
                    int(block.rmin),
                    int(block.rmax),
                    int(block.cmin),
                    int(block.cmax),
                ],
                "height": int(block.height),
                "width": int(block.width),
                "orientation": block.orientation,
                "access_side_names": sorted(block.access_side_names),
                "pick_face_side_names": sorted(block.pick_face_side_names),
                "raw_adjacent_access_side_names": sorted(
                    block.raw_adjacent_access_side_names
                ),
                "pick_face_count": int(len(block.pick_faces)),
            }
            for block in blocks
        ],
    }


def _candidate_archive_eligible(candidate: nsga2_bs.LayoutCandidate) -> bool:
    """Return whether a candidate may enter saved layout archives."""
    if not candidate.is_feasible:
        return False
    metrics = candidate.metrics or {}
    network_reachable = metrics.get(
        "has_access_anchor_reachable_aisle_network",
        metrics.get("single_aisle_component"),
    )
    return bool(metrics.get("has_access_anchor_connected_aisle")) and bool(
        network_reachable
    )


def _save_layout_archive(
    archive_path: Path,
    index_path: Path,
    archive_items: list[tuple[str, nsga2_bs.LayoutCandidate, dict[str, Any]]],
) -> tuple[Path, Path]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for archive_key, candidate, metadata in archive_items:
        layout = np.asarray(candidate.node.layout)
        arrays[archive_key] = layout
        feature_keys = metadata.get("feature_keys", {})
        if not isinstance(feature_keys, dict):
            continue

        pick_face_key = str(feature_keys.get("pick_face_mask", "") or "")
        storage_key = str(feature_keys.get("storage_mask", "") or "")
        aisle_key = str(feature_keys.get("aisle_mask", "") or "")
        if pick_face_key:
            pick_face_mask = np.zeros(layout.shape, dtype=bool)
            for row, col in detect_pick_faces(layout):
                pick_face_mask[row, col] = True
            arrays[pick_face_key] = pick_face_mask
        if storage_key:
            arrays[storage_key] = compute_storage_mask(layout)
        if aisle_key:
            arrays[aisle_key] = compute_aisle_mask(layout)
    np.savez_compressed(archive_path, **arrays)
    _write_json_list(index_path, [metadata for _, _, metadata in archive_items])
    return archive_path, index_path


def write_generation_elites_archive(
    run: PlannedRun,
    run_dir: Path,
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
    archive_rank_max: int,
) -> tuple[Path, Path] | None:
    """Save rank <= K feasible candidates from each generation."""
    archive_items: list[tuple[str, nsga2_bs.LayoutCandidate, dict[str, Any]]] = []
    seen: set[str] = set()
    for generation, candidates in generation_candidates:
        for candidate in candidates:
            if not _candidate_archive_eligible(candidate) or candidate.rank is None:
                continue
            if int(candidate.rank) > int(archive_rank_max):
                continue
            signature = nsga2_bs._signature_digest(
                nsga2_bs.layout_signature(candidate.node.layout)
            )
            if signature in seen:
                continue
            seen.add(signature)
            archive_key = (
                f"gen_{generation:03d}_rank_{int(candidate.rank):02d}_"
                f"candidate_{int(candidate.candidate_id):03d}"
            )
            metadata = _candidate_archive_index_entry(
                archive_key=archive_key,
                run=run,
                candidate=candidate,
                generation=generation,
            )
            archive_items.append((archive_key, candidate, metadata))

    if not archive_items:
        return None
    return _save_layout_archive(
        run_dir / "generation_elites.npz",
        run_dir / "generation_elites_index.json",
        archive_items,
    )


def _unique_feasible_candidates(
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
) -> list[tuple[int, nsga2_bs.LayoutCandidate]]:
    unique: list[tuple[int, nsga2_bs.LayoutCandidate]] = []
    seen: set[str] = set()
    for generation, candidates in generation_candidates:
        for candidate in candidates:
            if not _candidate_archive_eligible(candidate):
                continue
            signature = nsga2_bs._signature_digest(
                nsga2_bs.layout_signature(candidate.node.layout)
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((generation, candidate))
    return unique


def _unique_evaluated_candidates(
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
) -> list[tuple[int, nsga2_bs.LayoutCandidate]]:
    unique: list[tuple[int, nsga2_bs.LayoutCandidate]] = []
    seen: set[str] = set()
    for generation, candidates in generation_candidates:
        for candidate in candidates:
            signature = nsga2_bs._signature_digest(
                nsga2_bs.layout_signature(candidate.node.layout)
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((generation, candidate))
    return unique


def _generation_elites_archive_candidate_count(
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
    archive_rank_max: int,
) -> int:
    seen: set[str] = set()
    count = 0
    for _, candidates in generation_candidates:
        for candidate in candidates:
            if not _candidate_archive_eligible(candidate) or candidate.rank is None:
                continue
            if int(candidate.rank) > int(archive_rank_max):
                continue
            signature = nsga2_bs._signature_digest(
                nsga2_bs.layout_signature(candidate.node.layout)
            )
            if signature in seen:
                continue
            seen.add(signature)
            count += 1
    return count


def _rank_unique_candidates(
    unique_candidates: list[tuple[int, nsga2_bs.LayoutCandidate]],
) -> tuple[list[int], list[float]]:
    if not unique_candidates:
        return [], []
    candidates = [candidate for _, candidate in unique_candidates]
    objectives = nsga2_bs.objective_array(candidates)
    fronts = nsga2_bs.non_dominated_sort(
        objectives,
        nsga2_bs.OBJECTIVE_DIRECTIONS,
    )
    ranks = nsga2_bs.assign_ranks(fronts, len(candidates))
    crowding = nsga2_bs.crowding_distances_for_fronts(
        objectives,
        fronts,
        nsga2_bs.OBJECTIVE_DIRECTIONS,
    )
    return ranks, crowding


def write_final_ranked_archive(
    run: PlannedRun,
    run_dir: Path,
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
    archive_rank_max: int,
) -> tuple[Path, Path] | None:
    """Re-rank and save unique feasible layouts with final rank <= archive_rank_max."""
    unique_candidates = _unique_feasible_candidates(generation_candidates)
    ranks, crowding = _rank_unique_candidates(unique_candidates)
    archive_items: list[tuple[str, nsga2_bs.LayoutCandidate, dict[str, Any]]] = []

    rank_layout_counts: dict[int, int] = {}
    for index, (generation, candidate) in enumerate(unique_candidates):
        rank = int(ranks[index])
        if rank > int(archive_rank_max):
            continue
        rank_layout_counts[rank] = rank_layout_counts.get(rank, 0) + 1
        archive_key = (
            f"final_rank_{rank:02d}_layout_{rank_layout_counts[rank]:03d}"
        )
        metadata = _candidate_archive_index_entry(
            archive_key=archive_key,
            run=run,
            candidate=candidate,
            generation=generation,
            rank=rank,
            crowding_distance=float(crowding[index]),
        )
        archive_items.append((archive_key, candidate, metadata))

    if not archive_items:
        return None
    return _save_layout_archive(
        run_dir / "final_ranked_layouts.npz",
        run_dir / "final_ranked_layouts_index.json",
        archive_items,
    )


def write_all_debug_archive(
    run: PlannedRun,
    run_dir: Path,
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
) -> tuple[Path, Path] | None:
    """Save every unique feasible evaluated layout for debugging."""
    archive_items: list[tuple[str, nsga2_bs.LayoutCandidate, dict[str, Any]]] = []
    for generation, candidate in _unique_feasible_candidates(generation_candidates):
        archive_key = (
            f"debug_gen_{generation:03d}_candidate_{int(candidate.candidate_id):03d}"
        )
        metadata = _candidate_archive_index_entry(
            archive_key=archive_key,
            run=run,
            candidate=candidate,
            generation=generation,
        )
        archive_items.append((archive_key, candidate, metadata))

    if not archive_items:
        return None
    return _save_layout_archive(
        run_dir / "all_debug_layouts.npz",
        run_dir / "all_debug_layouts_index.json",
        archive_items,
    )


def write_all_candidates_debug_archive(
    run: PlannedRun,
    run_dir: Path,
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]],
) -> tuple[Path, Path] | None:
    """Save every unique evaluated layout, including infeasible candidates."""
    archive_items: list[tuple[str, nsga2_bs.LayoutCandidate, dict[str, Any]]] = []
    for generation, candidate in _unique_evaluated_candidates(generation_candidates):
        status = "feasible" if candidate.is_feasible else "infeasible"
        archive_key = (
            f"{status}_gen_{generation:03d}_candidate_{int(candidate.candidate_id):03d}"
        )
        metadata = _candidate_archive_index_entry(
            archive_key=archive_key,
            run=run,
            candidate=candidate,
            generation=generation,
        )
        metadata["is_feasible"] = bool(candidate.is_feasible)
        metadata["feasibility_violations"] = [
            _json_safe_value(value)
            for value in candidate.feasibility.get("violations", [])
        ]
        archive_items.append((archive_key, candidate, metadata))

    if not archive_items:
        return None
    return _save_layout_archive(
        run_dir / "all_candidates_debug_layouts.npz",
        run_dir / "all_candidates_debug_layouts_index.json",
        archive_items,
    )


def _final_generation_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generations = [
        int(row["generation"])
        for row in candidate_rows
        if row.get("generation") not in {"", None}
    ]
    if not generations:
        return []
    final_generation = max(generations)
    return [row for row in candidate_rows if int(row["generation"]) == final_generation]


def experiment_summary_row(
    run: PlannedRun,
    candidate_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    runtime_seconds: float,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    """Build one global experiment summary row."""
    final_rows = _final_generation_rows(candidate_rows)
    selected_final = [row for row in final_rows if row.get("selected") is True]
    rank0_final = [row for row in final_rows if str(row.get("rank")) == "0"]
    return {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        **budget_fields_for_run(run),
        "population_size": run.parameters["population_size"],
        "generations": run.parameters["generations"],
        "beam_width": run.parameters["beam_width"],
        "max_depth": run.parameters["max_depth"],
        "adaptive_spacing_mode": _first_row_value(candidate_rows, "adaptive_spacing_mode"),
        "adaptive_spacing_alpha": _first_row_value(candidate_rows, "adaptive_spacing_alpha"),
        "adaptive_spacing_bf": _first_row_value(candidate_rows, "adaptive_spacing_bf"),
        "parameter_source": run.parameter_source,
        "beta_h": _first_row_value(candidate_rows, "beta_h"),
        "beta_v": _first_row_value(candidate_rows, "beta_v"),
        "total_candidates": len(candidate_rows),
        "final_generation_candidates": len(final_rows),
        "final_generation_rank0_count": len(rank0_final),
        "final_generation_selected_count": len(selected_final),
        "final_best_pick_faces": _best_value(final_rows, "pick_faces", maximize=True),
        "final_best_storage_total": _best_value(final_rows, "storage_total", maximize=True),
        "final_best_interior_storage": _best_value(final_rows, "interior_storage", maximize=False),
        "final_best_retrieval_penalty": _best_value(final_rows, "retrieval_penalty", maximize=False),
        "total_runtime_seconds": runtime_seconds,
        "status": status,
        "error_message": error_message,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def experiment_metadata(
    experiment_id: str,
    method: str,
    planned_runs: list[PlannedRun],
    parameters: dict[str, int],
    overrides: dict[str, int | None],
    archive_layouts: str = "none",
    archive_rank_max: int = 3,
    sorting_rule_mode: str = nsga2_bs.DEFAULT_SORTING_RULE_MODE,
    sorting_rule: str = nsga2_bs.DEFAULT_SORTING_RULE,
    adaptive_weight_mode: str = nsga2_bs.DEFAULT_ADAPTIVE_WEIGHT_MODE,
    fixed_beam_w1: float = bs_only_direct.FIXED_BS_WEIGHTS["w1"],
    fixed_beam_w2: float = bs_only_direct.FIXED_BS_WEIGHTS["w2"],
    fixed_beam_lambda: float = bs_only_direct.FIXED_BS_WEIGHTS["lambda"],
    mutation_mode: str = nsga2_bs.DEFAULT_MUTATION_MODE,
    initialization_spacing_mode: str = nsga2_bs.DEFAULT_INITIALIZATION_MODE,
    ablation_variant: str = DEFAULT_ABLATION_VARIANT,
    bs_rule_policy: str = bs_only_direct.DEFAULT_BS_RULE_POLICY,
    bs_weight_policy: str = bs_only_direct.DEFAULT_BS_WEIGHT_POLICY,
    decode_budget: int | None = None,
    profile_light: bool = False,
    save_generation_objectives: bool = False,
    command_line: list[str] | None = None,
) -> dict[str, Any]:
    sorting_rules = nsga2_bs.load_sorting_rules()
    parameter_sources = sorted({run.parameter_source for run in planned_runs})
    budget_policies = sorted({run.budget_policy for run in planned_runs})
    auto_used = any(run.budget_policy == "auto_from_instance" for run in planned_runs)
    auto_by_instance = {
        run.instance.name: dict(run.auto_parameters)
        for run in planned_runs
        if run.auto_parameters
    }
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "instances": [run.instance.name for run in planned_runs],
        "seeds": sorted({run.seed for run in planned_runs}),
        "default_parameters": parameters,
        "command_line_overrides": overrides,
        "archive_layouts": archive_layouts,
        "archive_rank_max": int(archive_rank_max),
        "objective_keys": list(nsga2_bs.OBJECTIVE_KEYS),
        "objective_directions": list(nsga2_bs.OBJECTIVE_DIRECTIONS),
        "sorting_rule_mode": sorting_rule_mode,
        "sorting_rule": sorting_rule,
        "fixed_sorting_rule": sorting_rule,
        "sorting_rule_pool": sorted(sorting_rules),
        "sorting_rule_pool_path": str(CONFIG_DIR / "sorting_rules.yaml"),
        "adaptive_weight_mode": adaptive_weight_mode,
        "fixed_w1": float(fixed_beam_w1),
        "fixed_w2": float(fixed_beam_w2),
        "lambda": float(fixed_beam_lambda),
        "mutation_mode": mutation_mode,
        "mutation_operator_probabilities": nsga2_bs.mutation_probabilities_for_mode(mutation_mode)
        if method == METHOD_NAME
        else {},
        "symmetry_breaking_enabled": nsga2_bs.symmetry_breaking_enabled_for_mode(mutation_mode)
        if method == METHOD_NAME
        else False,
        "initialization_spacing_mode": initialization_spacing_mode,
        "ablation_variant": ablation_variant,
        "decode_budget": decode_budget,
        "profile_light": bool(profile_light),
        "save_generation_objectives": bool(save_generation_objectives),
        "budget_policy": budget_policies[0] if len(budget_policies) == 1 else budget_policies,
        "adaptive_spacing_mode": ADAPTIVE_SPACING_MODE if method == METHOD_NAME else None,
        "adaptive_spacing_alpha": ADAPTIVE_SPACING_ALPHA if method == METHOD_NAME else None,
        "adaptive_spacing_bf": ADAPTIVE_SPACING_BF,
        "auto_params_used": bool(auto_used),
        "auto_parameters_by_instance": auto_by_instance,
        "parameter_source": parameter_sources[0] if len(parameter_sources) == 1 else parameter_sources,
        **_bs_policy_metadata_fields(method, bs_rule_policy, bs_weight_policy),
        "bs_fixed_weights": dict(bs_only_direct.FIXED_BS_WEIGHTS),
        "initialization_mode": initialization_spacing_mode,
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_line": command_line or sys.argv,
    }


def _bs_depth_summary_row(
    run: PlannedRun,
    depth_row: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
    }
    row.update(depth_row)
    return row


def _run_one_bs_only_direct_run(
    run: PlannedRun,
    output_dir: Path,
    archive_layouts: str,
    archive_rank_max: int,
    sorting_rule: str,
    bs_rule_policy: str,
    bs_weight_policy: str,
    profile_light: bool = False,
    save_generation_objectives: bool = False,
) -> dict[str, Any]:
    run_dir = run_output_dir(output_dir, run)
    figures_dir = run_dir / "figures"
    candidates_csv = run_dir / "candidates.csv"
    generation_csv = run_dir / "generation_summary.csv"
    bs_depth_csv = run_dir / "bs_depth_summary.csv"
    run_metadata_path = run_dir / "run_metadata.json"
    generation_elites_path = run_dir / "generation_elites.npz"
    generation_elites_index_path = run_dir / "generation_elites_index.json"
    final_ranked_layouts_path = run_dir / "final_ranked_layouts.npz"
    final_ranked_layouts_index_path = run_dir / "final_ranked_layouts_index.json"
    all_debug_layouts_path = run_dir / "all_debug_layouts.npz"
    all_debug_layouts_index_path = run_dir / "all_debug_layouts_index.json"
    all_candidates_debug_layouts_path = run_dir / "all_candidates_debug_layouts.npz"
    all_candidates_debug_layouts_index_path = run_dir / "all_candidates_debug_layouts_index.json"
    profile_paths = _profile_output_paths(run_dir)
    start_time = datetime.now(timezone.utc)
    started = time.perf_counter()
    status = "completed"
    error_message = ""
    candidate_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    generation_candidates: list[tuple[int, list[Any]]] = []
    profile_times = _empty_profile_times()
    sorting_rules = nsga2_bs.load_sorting_rules()
    sorting_rule_pool = sorted(sorting_rules)

    metadata = {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        "parameters": run.parameters,
        **budget_fields_for_run(run),
        "archive_layouts": archive_layouts,
        "archive_rank_max": int(archive_rank_max),
        "objective_keys": list(nsga2_bs.OBJECTIVE_KEYS),
        "objective_directions": list(nsga2_bs.OBJECTIVE_DIRECTIONS),
        "sorting_rule": sorting_rule,
        "sorting_rule_pool": sorting_rule_pool,
        "sorting_rule_pool_path": str(CONFIG_DIR / "sorting_rules.yaml"),
        "bs_rule_policy": bs_rule_policy,
        "bs_weight_policy": bs_weight_policy,
        "beam_w1": bs_only_direct.FIXED_BS_WEIGHTS["w1"],
        "beam_w2": bs_only_direct.FIXED_BS_WEIGHTS["w2"],
        "beam_lambda": bs_only_direct.FIXED_BS_WEIGHTS["lambda"],
        "initialization_mode": "direct_root",
        "aisle_width": None,
        "min_fragment_size": None,
        "population_size": 0,
        "uses_chromosomes": False,
        "mutation_probabilities": "not_applicable",
        "crossover_probability": 0.0,
        "beam_width": run.parameters["beam_width"],
        "safety_max_depth": run.parameters["max_depth"],
        "max_depth": run.parameters["max_depth"],
        "safety_max_depth_reached": False,
        "max_depth_reached": 0,
        "archive_generation_elites_requested": archive_layouts in {"generation_elites", "both"},
        "archive_final_ranked_requested": archive_layouts in {"final_ranked", "both"},
        "archive_generation_elites_written": False,
        "archive_final_ranked_written": False,
        "archive_generation_elites_candidate_count": 0,
        "archive_final_ranked_candidate_count": 0,
        "archive_skip_reason": "",
        "archive_error_message": "",
        "profile_light": bool(profile_light),
        "save_generation_objectives": bool(save_generation_objectives),
        "input_mask_path": str(run.instance.mask_path),
        "output_paths": {
            "candidates_csv": str(candidates_csv),
            "generation_summary_csv": str(generation_csv),
            "bs_depth_summary_csv": str(bs_depth_csv),
            "runtime_profile_summary_csv": str(profile_paths["runtime_profile_summary_csv"]),
            "generation_profile_csv": str(profile_paths["generation_profile_csv"]),
            "generation_objectives_csv": str(profile_paths["generation_objectives_csv"]),
            "run_metadata_json": str(run_metadata_path),
            "figures_dir": str(figures_dir),
            "generation_elites_path": str(generation_elites_path),
            "generation_elites_index_path": str(generation_elites_index_path),
            "final_ranked_layouts_path": str(final_ranked_layouts_path),
            "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
            "all_debug_layouts_path": str(all_debug_layouts_path),
            "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
            "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
            "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        },
        "generation_elites_path": str(generation_elites_path),
        "generation_elites_index_path": str(generation_elites_index_path),
        "final_ranked_layouts_path": str(final_ranked_layouts_path),
        "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
        "all_debug_layouts_path": str(all_debug_layouts_path),
        "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
        "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
        "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        "start_time": start_time.isoformat(),
        "end_time": None,
        "runtime_seconds": None,
        "status": status,
        "error_message": error_message,
    }

    try:
        initialization_started = time.perf_counter()
        masks = load_mask(run.instance.mask_path)
        grid = mask_to_grid(masks)
        aisle_width = int(masks["aisle_width"])
        metadata["aisle_width"] = aisle_width
        metadata["min_fragment_size"] = nsga2_bs.min_fragment_size(aisle_width)
        started_bs = time.perf_counter()
        candidates, depth_rows, bs_metadata = bs_only_direct.build_direct_bs_candidates(
            grid,
            masks,
            aisle_width=aisle_width,
            beam_width=run.parameters["beam_width"],
            safety_max_depth=run.parameters["max_depth"],
            sorting_rule=sorting_rule,
            bs_rule_policy=bs_rule_policy,
            bs_weight_policy=bs_weight_policy,
            sorting_rules=sorting_rules,
        )
        generation_elapsed = time.perf_counter() - started_bs
        _add_profile_time(profile_times, "beam_decode_time_seconds", generation_elapsed)
        metadata.update(bs_metadata)
        generation_candidates.append((0, candidates))

        for candidate in candidates:
            row = bs_only_direct.candidate_to_csv_row(
                candidate,
                run_id=run.run_id,
                method=run.method,
                instance_name=run.instance.name,
                seed=run.seed,
                generation=0,
            )
            row.update(budget_fields_for_run(run))
            candidate_rows.append(row)

        feasible_count = sum(1 for candidate in candidates if candidate.is_feasible)
        non_dominated_count = sum(
            1 for candidate in candidates if candidate.is_feasible and candidate.rank == 0
        )
        summary = {
            "generation": 0,
            "chromosome_count": 0,
            "decoded_candidate_count": int(bs_metadata["decoded_count"]),
            "feasible_candidate_count": feasible_count,
            "non_dominated_count": non_dominated_count,
            "selected_survivor_count": non_dominated_count,
            "runtime_seconds": generation_elapsed,
        }
        generation_rows.append(generation_summary_row(run, summary, candidate_rows))
        bs_depth_rows = [_bs_depth_summary_row(run, row) for row in depth_rows]

        io_started = time.perf_counter()
        _write_csv(candidates_csv, candidate_rows, CANDIDATE_COLUMNS)
        _write_csv(generation_csv, generation_rows, GENERATION_SUMMARY_COLUMNS)
        _write_csv(bs_depth_csv, bs_depth_rows, BS_DEPTH_SUMMARY_COLUMNS)
        _add_profile_time(profile_times, "io_write_time_seconds", time.perf_counter() - io_started)

        metadata["sorting_rule_counts"] = {
            rule: sum(1 for row in candidate_rows if row.get("sorting_rule") == rule)
            for rule in sorted({str(row.get("sorting_rule")) for row in candidate_rows if row.get("sorting_rule")})
        }
        metadata["scalar_score_candidate_count"] = sum(
            1 for row in candidate_rows if str(row.get("uses_scalar_score")) == "True"
        )
        metadata.update(_observed_decode_metadata_from_rows(candidate_rows))
        metadata["initialization_mode_counts"] = {
            "direct_root": len(candidate_rows),
        }

        archive_skip_reasons: list[str] = []
        try:
            archive_dedup_started = time.perf_counter()
            generation_elites_candidate_count = _generation_elites_archive_candidate_count(
                generation_candidates,
                archive_rank_max,
            )
            final_ranked_candidate_count = len(_unique_feasible_candidates(generation_candidates))
            _add_profile_time(
                profile_times,
                "archive_dedup_time_seconds",
                time.perf_counter() - archive_dedup_started,
            )
            metadata["archive_generation_elites_candidate_count"] = (
                generation_elites_candidate_count
            )
            metadata["archive_final_ranked_candidate_count"] = final_ranked_candidate_count

            if archive_layouts in {"generation_elites", "both"}:
                archive_write_started = time.perf_counter()
                generation_archive = write_generation_elites_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_generation_elites_written"] = generation_archive is not None
                if generation_archive is None:
                    archive_skip_reasons.append("generation_elites: no rank <= archive_rank_max feasible layouts")
            elif archive_layouts == "all_debug":
                archive_write_started = time.perf_counter()
                write_all_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
            elif archive_layouts == "all_candidates_debug":
                archive_write_started = time.perf_counter()
                write_all_candidates_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )

            if archive_layouts in {"final_ranked", "both"}:
                archive_write_started = time.perf_counter()
                final_archive = write_final_ranked_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_final_ranked_written"] = final_archive is not None
                if final_archive is None:
                    archive_skip_reasons.append("final_ranked: no eligible feasible layouts")

            if (
                archive_layouts in {"generation_elites", "final_ranked", "both"}
                and generation_elites_candidate_count == 0
                and final_ranked_candidate_count == 0
            ):
                metadata["archive_skip_reason"] = "no eligible feasible layouts"
            else:
                metadata["archive_skip_reason"] = "; ".join(archive_skip_reasons)
        except Exception as archive_exc:
            metadata["archive_error_message"] = f"{type(archive_exc).__name__}: {archive_exc}"
            raise

        print(
            "run_id={run_id} instance={instance} seed={seed} method=bs_only_direct "
            "decoded={decoded} terminal={terminal} feasible={feasible} "
            "rank0={rank0} safety_max_depth_reached={safety} runtime_seconds={runtime:.3f}".format(
                run_id=run.run_id,
                instance=run.instance.name,
                seed=run.seed,
                decoded=bs_metadata["decoded_count"],
                terminal=len(candidates),
                feasible=feasible_count,
                rank0=non_dominated_count,
                safety=bs_metadata["safety_max_depth_reached"],
                runtime=generation_elapsed,
            )
        )
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        if not metadata.get("archive_error_message") and "archive" in str(exc).lower():
            metadata["archive_error_message"] = error_message
        run_dir.mkdir(parents=True, exist_ok=True)

    runtime_seconds = time.perf_counter() - started
    end_time = datetime.now(timezone.utc)
    metadata.update(
        {
            "end_time": end_time.isoformat(),
            "runtime_seconds": runtime_seconds,
            "status": status,
            "error_message": error_message,
        }
    )
    if profile_light or save_generation_objectives:
        _write_profile_outputs(
            run=run,
            run_dir=run_dir,
            ablation_variant=DEFAULT_ABLATION_VARIANT,
            total_runtime_seconds=runtime_seconds,
            profile_times=profile_times,
            generation_profile_rows=_generation_profile_rows_from_existing(
                run,
                DEFAULT_ABLATION_VARIANT,
                generation_rows,
                candidate_rows,
            ),
            generation_objective_rows=_generation_objective_rows(
                run,
                DEFAULT_ABLATION_VARIANT,
                candidate_rows,
            ),
            profile_light=profile_light,
            save_generation_objectives=save_generation_objectives,
        )
    _write_json(run_metadata_path, metadata)

    return experiment_summary_row(
        run,
        candidate_rows,
        generation_rows,
        runtime_seconds,
        status,
        error_message=error_message,
    )


def _run_one_random_restart_bs_run(
    run: PlannedRun,
    output_dir: Path,
    save_figures: bool,
    archive_layouts: str,
    archive_rank_max: int,
    sorting_rule_mode: str,
    sorting_rule: str,
    adaptive_weight_mode: str,
    decode_budget: int | None,
    profile_light: bool = False,
    save_generation_objectives: bool = False,
) -> dict[str, Any]:
    run_dir = run_output_dir(output_dir, run)
    figures_dir = run_dir / "figures"
    candidates_csv = run_dir / "candidates.csv"
    generation_csv = run_dir / "generation_summary.csv"
    run_metadata_path = run_dir / "run_metadata.json"
    generation_elites_path = run_dir / "generation_elites.npz"
    generation_elites_index_path = run_dir / "generation_elites_index.json"
    final_ranked_layouts_path = run_dir / "final_ranked_layouts.npz"
    final_ranked_layouts_index_path = run_dir / "final_ranked_layouts_index.json"
    all_debug_layouts_path = run_dir / "all_debug_layouts.npz"
    all_debug_layouts_index_path = run_dir / "all_debug_layouts_index.json"
    all_candidates_debug_layouts_path = run_dir / "all_candidates_debug_layouts.npz"
    all_candidates_debug_layouts_index_path = run_dir / "all_candidates_debug_layouts_index.json"
    profile_paths = _profile_output_paths(run_dir)
    start_time = datetime.now(timezone.utc)
    started = time.perf_counter()
    status = "completed"
    error_message = ""
    candidate_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]] = []
    profile_times = _empty_profile_times()
    sorting_rules = nsga2_bs.load_sorting_rules()
    sorting_rule_pool = sorted(sorting_rules)
    effective_decode_budget = run.parameters.get("decode_budget", decode_budget)
    resolved_decode_budget = random_restart_bs.resolve_decode_budget(
        run.parameters["population_size"],
        run.parameters["generations"],
        effective_decode_budget,
    )

    metadata = {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        "parameters": run.parameters,
        **budget_fields_for_run(run),
        "archive_layouts": archive_layouts,
        "archive_rank_max": int(archive_rank_max),
        "objective_keys": list(nsga2_bs.OBJECTIVE_KEYS),
        "objective_directions": list(nsga2_bs.OBJECTIVE_DIRECTIONS),
        "sorting_rule_mode": sorting_rule_mode,
        "sorting_rule": sorting_rule,
        "sorting_rule_pool": sorting_rule_pool,
        "sorting_rule_pool_path": str(CONFIG_DIR / "sorting_rules.yaml"),
        "adaptive_weight_mode": adaptive_weight_mode,
        "initialization_mode": nsga2_bs.DEFAULT_INITIALIZATION_MODE,
        "initialization_spacing_mode": nsga2_bs.DEFAULT_INITIALIZATION_MODE,
        "aisle_width": None,
        "min_fragment_size": None,
        "population_size_equivalent": run.parameters["population_size"],
        "generation_count_equivalent": run.parameters["generations"],
        "decode_budget": resolved_decode_budget,
        "uses_chromosomes": True,
        "uses_nsga2": False,
        "uses_parent_selection": False,
        "uses_survivor_selection": False,
        "uses_crossover": False,
        "uses_mutation": False,
        "mutation_probabilities": "not_applicable",
        "crossover_probability": 0.0,
        "beam_width": run.parameters["beam_width"],
        "max_depth": run.parameters["max_depth"],
        "archive_generation_elites_requested": archive_layouts in {"generation_elites", "both"},
        "archive_final_ranked_requested": archive_layouts in {"final_ranked", "both"},
        "archive_generation_elites_written": False,
        "archive_final_ranked_written": False,
        "archive_generation_elites_candidate_count": 0,
        "archive_final_ranked_candidate_count": 0,
        "archive_skip_reason": "",
        "archive_error_message": "",
        "profile_light": bool(profile_light),
        "save_generation_objectives": bool(save_generation_objectives),
        "input_mask_path": str(run.instance.mask_path),
        "output_paths": {
            "candidates_csv": str(candidates_csv),
            "generation_summary_csv": str(generation_csv),
            "runtime_profile_summary_csv": str(profile_paths["runtime_profile_summary_csv"]),
            "generation_profile_csv": str(profile_paths["generation_profile_csv"]),
            "generation_objectives_csv": str(profile_paths["generation_objectives_csv"]),
            "run_metadata_json": str(run_metadata_path),
            "figures_dir": str(figures_dir),
            "generation_elites_path": str(generation_elites_path),
            "generation_elites_index_path": str(generation_elites_index_path),
            "final_ranked_layouts_path": str(final_ranked_layouts_path),
            "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
            "all_debug_layouts_path": str(all_debug_layouts_path),
            "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
            "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
            "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        },
        "generation_elites_path": str(generation_elites_path),
        "generation_elites_index_path": str(generation_elites_index_path),
        "final_ranked_layouts_path": str(final_ranked_layouts_path),
        "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
        "all_debug_layouts_path": str(all_debug_layouts_path),
        "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
        "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
        "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        "start_time": start_time.isoformat(),
        "end_time": None,
        "runtime_seconds": None,
        "status": status,
        "error_message": error_message,
    }

    try:
        initialization_started = time.perf_counter()
        masks = load_mask(run.instance.mask_path)
        grid = mask_to_grid(masks)
        aisle_width = int(masks["aisle_width"])
        metadata["aisle_width"] = aisle_width
        metadata["min_fragment_size"] = nsga2_bs.min_fragment_size(aisle_width)

        build_started = time.perf_counter()
        candidates, batch_summaries, restart_metadata = (
            random_restart_bs.build_random_restart_candidates(
                grid,
                masks,
                aisle_width=aisle_width,
                population_size=run.parameters["population_size"],
                generations=run.parameters["generations"],
                decode_budget=resolved_decode_budget,
                beam_width=run.parameters["beam_width"],
                max_depth=run.parameters["max_depth"],
                seed=run.seed,
                sorting_rule_mode=sorting_rule_mode,
                sorting_rule=sorting_rule,
                adaptive_weight_mode=adaptive_weight_mode,
                sorting_rules=sorting_rules,
            )
        )
        build_elapsed = time.perf_counter() - build_started
        _add_profile_time(profile_times, "beam_decode_time_seconds", build_elapsed)
        metadata.update(restart_metadata)

        grouped_candidates: dict[int, list[nsga2_bs.LayoutCandidate]] = {}
        for candidate in candidates:
            batch = int(candidate.decode_metadata.get("batch_index", 0))
            grouped_candidates.setdefault(batch, []).append(candidate)
            row = random_restart_bs.candidate_to_csv_row(
                candidate,
                run_id=run.run_id,
                method=run.method,
                instance_name=run.instance.name,
                seed=run.seed,
                generation=batch,
            )
            row.update(budget_fields_for_run(run))
            candidate_rows.append(row)
        generation_candidates = [
            (batch, grouped_candidates.get(batch, []))
            for batch in range(run.parameters["generations"])
        ]

        rows_by_batch: dict[int, list[dict[str, Any]]] = {}
        for row in candidate_rows:
            rows_by_batch.setdefault(int(row.get("generation", 0)), []).append(row)
        for summary in batch_summaries:
            batch = int(summary["generation"])
            generation_rows.append(
                generation_summary_row(
                    run,
                    summary,
                    rows_by_batch.get(batch, []),
                )
            )
        if not generation_rows and not candidate_rows:
            generation_rows.append(
                generation_summary_row(
                    run,
                    {
                        "generation": 0,
                        "chromosome_count": resolved_decode_budget,
                        "decoded_candidate_count": 0,
                        "feasible_candidate_count": 0,
                        "non_dominated_count": 0,
                        "selected_survivor_count": 0,
                        "runtime_seconds": build_elapsed,
                    },
                    [],
                )
            )

        io_started = time.perf_counter()
        _write_csv(candidates_csv, candidate_rows, CANDIDATE_COLUMNS)
        _write_csv(generation_csv, generation_rows, GENERATION_SUMMARY_COLUMNS)
        _add_profile_time(profile_times, "io_write_time_seconds", time.perf_counter() - io_started)
        metadata.update(_observed_decode_metadata_from_rows(candidate_rows))

        if save_figures:
            selected = [candidate for candidate in candidates if candidate.selected]
            nsga2_bs._save_selected_figures(
                selected,
                figures_dir,
                run.instance.name,
                run.seed,
                run.parameters["generations"] - 1,
            )

        archive_skip_reasons: list[str] = []
        try:
            archive_dedup_started = time.perf_counter()
            generation_elites_candidate_count = _generation_elites_archive_candidate_count(
                generation_candidates,
                archive_rank_max,
            )
            final_ranked_candidate_count = len(_unique_feasible_candidates(generation_candidates))
            _add_profile_time(
                profile_times,
                "archive_dedup_time_seconds",
                time.perf_counter() - archive_dedup_started,
            )
            metadata["archive_generation_elites_candidate_count"] = (
                generation_elites_candidate_count
            )
            metadata["archive_final_ranked_candidate_count"] = final_ranked_candidate_count

            if archive_layouts in {"generation_elites", "both"}:
                archive_write_started = time.perf_counter()
                generation_archive = write_generation_elites_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_generation_elites_written"] = generation_archive is not None
                if generation_archive is None:
                    archive_skip_reasons.append("generation_elites: no rank <= archive_rank_max feasible layouts")
            elif archive_layouts == "all_debug":
                archive_write_started = time.perf_counter()
                write_all_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
            elif archive_layouts == "all_candidates_debug":
                archive_write_started = time.perf_counter()
                write_all_candidates_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )

            if archive_layouts in {"final_ranked", "both"}:
                archive_write_started = time.perf_counter()
                final_archive = write_final_ranked_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_final_ranked_written"] = final_archive is not None
                if final_archive is None:
                    archive_skip_reasons.append("final_ranked: no eligible feasible layouts")

            if (
                archive_layouts in {"generation_elites", "final_ranked", "both"}
                and generation_elites_candidate_count == 0
                and final_ranked_candidate_count == 0
            ):
                metadata["archive_skip_reason"] = "no eligible feasible layouts"
            else:
                metadata["archive_skip_reason"] = "; ".join(archive_skip_reasons)
        except Exception as archive_exc:
            metadata["archive_error_message"] = f"{type(archive_exc).__name__}: {archive_exc}"
            raise

        print(
            "run_id={run_id} instance={instance} seed={seed} method=random_restart_bs "
            "restarts={restarts} decoded={decoded} unique={unique} duplicates={dupes} "
            "feasible={feasible} rank0={rank0} runtime_seconds={runtime:.3f}".format(
                run_id=run.run_id,
                instance=run.instance.name,
                seed=run.seed,
                restarts=resolved_decode_budget,
                decoded=restart_metadata["decoded_node_count"],
                unique=restart_metadata["unique_candidate_count"],
                dupes=restart_metadata["duplicate_layout_count"],
                feasible=restart_metadata["feasible_candidate_count"],
                rank0=restart_metadata["rank0_candidate_count"],
                runtime=build_elapsed,
            )
        )
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        if not metadata.get("archive_error_message") and "archive" in str(exc).lower():
            metadata["archive_error_message"] = error_message
        run_dir.mkdir(parents=True, exist_ok=True)

    runtime_seconds = time.perf_counter() - started
    end_time = datetime.now(timezone.utc)
    metadata.update(
        {
            "end_time": end_time.isoformat(),
            "runtime_seconds": runtime_seconds,
            "status": status,
            "error_message": error_message,
        }
    )
    if profile_light or save_generation_objectives:
        _write_profile_outputs(
            run=run,
            run_dir=run_dir,
            ablation_variant=DEFAULT_ABLATION_VARIANT,
            total_runtime_seconds=runtime_seconds,
            profile_times=profile_times,
            generation_profile_rows=_generation_profile_rows_from_existing(
                run,
                DEFAULT_ABLATION_VARIANT,
                generation_rows,
                candidate_rows,
            ),
            generation_objective_rows=_generation_objective_rows(
                run,
                DEFAULT_ABLATION_VARIANT,
                candidate_rows,
            ),
            profile_light=profile_light,
            save_generation_objectives=save_generation_objectives,
        )
    _write_json(run_metadata_path, metadata)

    return experiment_summary_row(
        run,
        candidate_rows,
        generation_rows,
        runtime_seconds,
        status,
        error_message=error_message,
    )


def _run_one_planned_run(
    run: PlannedRun,
    output_dir: Path,
    save_figures: bool,
    archive_layouts: str,
    archive_rank_max: int,
    sorting_rule_mode: str,
    sorting_rule: str,
    adaptive_weight_mode: str,
    fixed_beam_w1: float,
    fixed_beam_w2: float,
    fixed_beam_lambda: float,
    mutation_mode: str,
    initialization_spacing_mode: str,
    ablation_variant: str,
    bs_rule_policy: str,
    bs_weight_policy: str,
    decode_budget: int | None,
    profile_light: bool = False,
    save_generation_objectives: bool = False,
) -> dict[str, Any]:
    if run.method == BS_ONLY_DIRECT_METHOD_NAME:
        return _run_one_bs_only_direct_run(
            run,
            output_dir,
            archive_layouts,
            archive_rank_max,
            sorting_rule,
            bs_rule_policy,
            bs_weight_policy,
            profile_light,
            save_generation_objectives,
        )
    if run.method == RANDOM_RESTART_BS_METHOD_NAME:
        return _run_one_random_restart_bs_run(
            run,
            output_dir,
            save_figures,
            archive_layouts,
            archive_rank_max,
            sorting_rule_mode,
            sorting_rule,
            adaptive_weight_mode,
            decode_budget,
            profile_light,
            save_generation_objectives,
        )

    run_dir = run_output_dir(output_dir, run)
    figures_dir = run_dir / "figures"
    candidates_csv = run_dir / "candidates.csv"
    generation_csv = run_dir / "generation_summary.csv"
    run_metadata_path = run_dir / "run_metadata.json"
    generation_elites_path = run_dir / "generation_elites.npz"
    generation_elites_index_path = run_dir / "generation_elites_index.json"
    final_ranked_layouts_path = run_dir / "final_ranked_layouts.npz"
    final_ranked_layouts_index_path = run_dir / "final_ranked_layouts_index.json"
    all_debug_layouts_path = run_dir / "all_debug_layouts.npz"
    all_debug_layouts_index_path = run_dir / "all_debug_layouts_index.json"
    all_candidates_debug_layouts_path = run_dir / "all_candidates_debug_layouts.npz"
    all_candidates_debug_layouts_index_path = run_dir / "all_candidates_debug_layouts_index.json"
    profile_paths = _profile_output_paths(run_dir)
    start_time = datetime.now(timezone.utc)
    started = time.perf_counter()
    status = "completed"
    error_message = ""
    candidate_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    generation_profile_rows: list[dict[str, Any]] = []
    generation_objective_rows: list[dict[str, Any]] = []
    generation_candidates: list[tuple[int, list[nsga2_bs.LayoutCandidate]]] = []
    profile_times = _empty_profile_times()
    sorting_rules = nsga2_bs.load_sorting_rules()
    sorting_rule_pool = sorted(sorting_rules)

    metadata = {
        "run_id": run.run_id,
        "method": run.method,
        "instance": run.instance.name,
        "seed": run.seed,
        "parameters": run.parameters,
        **budget_fields_for_run(run),
        "archive_layouts": archive_layouts,
        "archive_rank_max": int(archive_rank_max),
        "objective_keys": list(nsga2_bs.OBJECTIVE_KEYS),
        "objective_directions": list(nsga2_bs.OBJECTIVE_DIRECTIONS),
        "sorting_rule_mode": sorting_rule_mode,
        "sorting_rule": sorting_rule,
        "fixed_sorting_rule": sorting_rule,
        "sorting_rule_pool": sorting_rule_pool,
        "sorting_rule_pool_path": str(CONFIG_DIR / "sorting_rules.yaml"),
        "adaptive_weight_mode": adaptive_weight_mode,
        "fixed_w1": float(fixed_beam_w1),
        "fixed_w2": float(fixed_beam_w2),
        "lambda": float(fixed_beam_lambda),
        "mutation_mode": mutation_mode,
        "mutation_operator_probabilities": nsga2_bs.mutation_probabilities_for_mode(mutation_mode),
        "symmetry_breaking_enabled": nsga2_bs.symmetry_breaking_enabled_for_mode(mutation_mode),
        "initialization_spacing_mode": initialization_spacing_mode,
        "ablation_variant": ablation_variant,
        "initialization_mode": initialization_spacing_mode,
        "adaptive_spacing_mode": ADAPTIVE_SPACING_MODE,
        "adaptive_spacing_alpha": ADAPTIVE_SPACING_ALPHA,
        "adaptive_spacing_bf": ADAPTIVE_SPACING_BF,
        "auto_params_used": run.budget_policy == "auto_from_instance",
        "parameter_source": run.parameter_source,
        "beta_h": None,
        "beta_v": None,
        "aisle_width": None,
        "min_fragment_size": None,
        "mutation_probabilities": mutation_mode,
        "crossover_probability": 1.0,
        "beam_width": run.parameters["beam_width"],
        "max_depth": run.parameters["max_depth"],
        "actual_population_size": run.parameters["population_size"],
        "actual_generations": run.parameters["generations"],
        "actual_beam_width": run.parameters["beam_width"],
        "actual_max_depth": run.parameters["max_depth"],
        "archive_generation_elites_requested": archive_layouts in {"generation_elites", "both"},
        "archive_final_ranked_requested": archive_layouts in {"final_ranked", "both"},
        "archive_generation_elites_written": False,
        "archive_final_ranked_written": False,
        "archive_generation_elites_candidate_count": 0,
        "archive_final_ranked_candidate_count": 0,
        "archive_skip_reason": "",
        "archive_error_message": "",
        "profile_light": bool(profile_light),
        "save_generation_objectives": bool(save_generation_objectives),
        "input_mask_path": str(run.instance.mask_path),
        "output_paths": {
            "candidates_csv": str(candidates_csv),
            "generation_summary_csv": str(generation_csv),
            "runtime_profile_summary_csv": str(profile_paths["runtime_profile_summary_csv"]),
            "generation_profile_csv": str(profile_paths["generation_profile_csv"]),
            "generation_objectives_csv": str(profile_paths["generation_objectives_csv"]),
            "run_metadata_json": str(run_metadata_path),
            "figures_dir": str(figures_dir),
            "generation_elites_path": str(generation_elites_path),
            "generation_elites_index_path": str(generation_elites_index_path),
            "final_ranked_layouts_path": str(final_ranked_layouts_path),
            "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
            "all_debug_layouts_path": str(all_debug_layouts_path),
            "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
            "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
            "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        },
        "generation_elites_path": str(generation_elites_path),
        "generation_elites_index_path": str(generation_elites_index_path),
        "final_ranked_layouts_path": str(final_ranked_layouts_path),
        "final_ranked_layouts_index_path": str(final_ranked_layouts_index_path),
        "all_debug_layouts_path": str(all_debug_layouts_path),
        "all_debug_layouts_index_path": str(all_debug_layouts_index_path),
        "all_candidates_debug_layouts_path": str(all_candidates_debug_layouts_path),
        "all_candidates_debug_layouts_index_path": str(all_candidates_debug_layouts_index_path),
        "start_time": start_time.isoformat(),
        "end_time": None,
        "runtime_seconds": None,
        "status": status,
        "error_message": error_message,
    }

    try:
        initialization_started = time.perf_counter()
        masks = load_mask(run.instance.mask_path)
        grid = mask_to_grid(masks)
        aisle_width = int(masks["aisle_width"])
        beta_h, beta_v = nsga2_bs.compute_aspect_ratio_betas(*grid.shape)
        metadata["aisle_width"] = aisle_width
        metadata["min_fragment_size"] = nsga2_bs.min_fragment_size(aisle_width)
        metadata["beta_h"] = beta_h
        metadata["beta_v"] = beta_v
        rng = np.random.default_rng(run.seed)
        population = nsga2_bs.create_initial_population_for_grid(
            grid,
            aisle_width,
            run.parameters["population_size"],
            seed=run.seed,
            initialization_spacing_mode=initialization_spacing_mode,
        )
        fixed_weights = {
            "w1": float(fixed_beam_w1),
            "w2": float(fixed_beam_w2),
            "lambda": float(fixed_beam_lambda),
        }
        _add_profile_time(
            profile_times,
            "initialization_time_seconds",
            time.perf_counter() - initialization_started,
        )

        for generation in range(run.parameters["generations"]):
            generation_started = time.perf_counter()
            generation_profile_times = _empty_profile_times()
            build_started = time.perf_counter()
            candidates, decoded_count = nsga2_bs.build_layout_candidates(
                population,
                grid,
                masks,
                aisle_width,
                run.parameters["beam_width"],
                run.parameters["max_depth"],
                seed=run.seed + generation * 1000,
                generation=generation,
                total_generations=run.parameters["generations"],
                sorting_rule_mode=sorting_rule_mode,
                sorting_rule=sorting_rule,
                adaptive_weight_mode=adaptive_weight_mode,
                fixed_weights=fixed_weights,
                sorting_rules=sorting_rules,
                profile_times=generation_profile_times,
            )
            build_elapsed = time.perf_counter() - build_started
            split_build_elapsed = sum(
                float(generation_profile_times.get(key, 0.0))
                for key in (
                    "beam_expansion_time_seconds",
                    "feasibility_filter_time_seconds",
                    "objective_evaluation_time_seconds",
                )
            )
            build_overhead_elapsed = max(0.0, build_elapsed - split_build_elapsed)
            _add_profile_time(
                generation_profile_times,
                "beam_decode_time_seconds",
                build_overhead_elapsed,
            )
            for key in (
                "beam_decode_time_seconds",
                "beam_expansion_time_seconds",
                "feasibility_filter_time_seconds",
                "objective_evaluation_time_seconds",
            ):
                _add_profile_time(profile_times, key, generation_profile_times.get(key, 0.0))
            survivor_started = time.perf_counter()
            selected = nsga2_bs.select_nsga2_survivors(
                candidates,
                run.parameters["population_size"],
            )
            survivor_elapsed = time.perf_counter() - survivor_started
            _add_profile_time(
                profile_times,
                "nsga_survivor_selection_time_seconds",
                survivor_elapsed,
            )
            _add_profile_time(
                generation_profile_times,
                "nsga_survivor_selection_time_seconds",
                survivor_elapsed,
            )
            generation_candidates.append((generation, candidates))
            feasible_count = sum(1 for candidate in candidates if candidate.is_feasible)
            non_dominated_count = sum(
                1
                for candidate in candidates
                if candidate.is_feasible and candidate.rank == 0
            )

            generation_candidate_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                row = nsga2_bs.candidate_to_csv_row(
                    candidate,
                    run.run_id,
                    run.instance.name,
                    run.seed,
                    generation,
                )
                row = _candidate_row_with_method(row, run.method)
                row.update(
                    {
                        **budget_fields_for_run(run),
                        "adaptive_spacing_mode": ADAPTIVE_SPACING_MODE,
                        "adaptive_spacing_alpha": ADAPTIVE_SPACING_ALPHA,
                        "adaptive_spacing_bf": ADAPTIVE_SPACING_BF,
                        "mutation_mode": mutation_mode,
                        "initialization_spacing_mode": initialization_spacing_mode,
                        "ablation_variant": ablation_variant,
                        "parameter_source": run.parameter_source,
                        "beta_h": beta_h,
                        "beta_v": beta_v,
                    }
                )
                generation_candidate_rows.append(row)
                candidate_rows.append(row)

            generation_elapsed = time.perf_counter() - generation_started
            summary = {
                "generation": generation,
                "chromosome_count": len(population),
                "decoded_candidate_count": decoded_count,
                "feasible_candidate_count": feasible_count,
                "non_dominated_count": non_dominated_count,
                "selected_survivor_count": len(selected),
                "runtime_seconds": generation_elapsed,
            }
            generation_rows.append(
                generation_summary_row(run, summary, generation_candidate_rows)
            )
            print(
                "run_id={run_id} instance={instance} seed={seed} generation={generation} "
                "decoded={decoded} feasible={feasible} selected={selected} "
                "runtime_seconds={runtime:.3f}".format(
                    run_id=run.run_id,
                    instance=run.instance.name,
                    seed=run.seed,
                    generation=generation,
                    decoded=decoded_count,
                    feasible=feasible_count,
                    selected=len(selected),
                    runtime=generation_elapsed,
                )
            )

            if save_figures:
                nsga2_bs._save_selected_figures(
                    selected,
                    figures_dir,
                    run.instance.name,
                    run.seed,
                    generation,
                )

            next_generation_started = time.perf_counter()
            population = nsga2_bs.make_next_generation(
                selected,
                population,
                run.parameters["population_size"],
                rng,
                aisle_width,
                mutation_mode=mutation_mode,
            )
            nsga_operator_elapsed = time.perf_counter() - next_generation_started
            _add_profile_time(profile_times, "nsga_operator_time_seconds", nsga_operator_elapsed)
            _add_profile_time(
                generation_profile_times,
                "nsga_operator_time_seconds",
                nsga_operator_elapsed,
            )
            generation_profile_rows.append(
                _generation_profile_row(
                    run=run,
                    ablation_variant=ablation_variant,
                    generation=generation,
                    generation_runtime_seconds=time.perf_counter() - generation_started,
                    population_size=run.parameters["population_size"],
                    decoded_count=decoded_count,
                    candidate_rows=generation_candidate_rows,
                    profile_times=generation_profile_times,
                )
            )
            if save_generation_objectives:
                generation_objective_rows.extend(
                    _generation_objective_rows(
                        run,
                        ablation_variant,
                        generation_candidate_rows,
                    )
                )

        io_started = time.perf_counter()
        _write_csv(candidates_csv, candidate_rows, CANDIDATE_COLUMNS)
        _write_csv(generation_csv, generation_rows, GENERATION_SUMMARY_COLUMNS)
        _add_profile_time(profile_times, "io_write_time_seconds", time.perf_counter() - io_started)
        metadata["sorting_rule_counts"] = {
            rule: sum(1 for row in candidate_rows if row.get("sorting_rule") == rule)
            for rule in sorted({str(row.get("sorting_rule")) for row in candidate_rows if row.get("sorting_rule")})
        }
        metadata["scalar_score_candidate_count"] = sum(
            1 for row in candidate_rows if str(row.get("uses_scalar_score")) == "True"
        )
        metadata.update(_observed_decode_metadata_from_rows(candidate_rows))
        metadata["initialization_mode_counts"] = {
            mode: sum(1 for row in candidate_rows if row.get("initialization_mode") == mode)
            for mode in sorted(
                {
                    str(row.get("initialization_mode"))
                    for row in candidate_rows
                    if row.get("initialization_mode")
                }
            )
        }

        archive_skip_reasons: list[str] = []
        try:
            archive_dedup_started = time.perf_counter()
            generation_elites_candidate_count = _generation_elites_archive_candidate_count(
                generation_candidates,
                archive_rank_max,
            )
            final_ranked_candidate_count = len(_unique_feasible_candidates(generation_candidates))
            _add_profile_time(
                profile_times,
                "archive_dedup_time_seconds",
                time.perf_counter() - archive_dedup_started,
            )
            metadata["archive_generation_elites_candidate_count"] = (
                generation_elites_candidate_count
            )
            metadata["archive_final_ranked_candidate_count"] = final_ranked_candidate_count

            if archive_layouts in {"generation_elites", "both"}:
                archive_write_started = time.perf_counter()
                generation_archive = write_generation_elites_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_generation_elites_written"] = generation_archive is not None
                if generation_archive is None:
                    archive_skip_reasons.append("generation_elites: no rank <= archive_rank_max feasible layouts")
            elif archive_layouts == "all_debug":
                archive_write_started = time.perf_counter()
                write_all_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
            elif archive_layouts == "all_candidates_debug":
                archive_write_started = time.perf_counter()
                write_all_candidates_debug_archive(run, run_dir, generation_candidates)
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )

            if archive_layouts in {"final_ranked", "both"}:
                archive_write_started = time.perf_counter()
                final_archive = write_final_ranked_archive(
                    run,
                    run_dir,
                    generation_candidates,
                    archive_rank_max,
                )
                _add_profile_time(
                    profile_times,
                    "archive_write_time_seconds",
                    time.perf_counter() - archive_write_started,
                )
                metadata["archive_final_ranked_written"] = final_archive is not None
                if final_archive is None:
                    archive_skip_reasons.append("final_ranked: no eligible feasible layouts")

            if (
                archive_layouts in {"generation_elites", "final_ranked", "both"}
                and generation_elites_candidate_count == 0
                and final_ranked_candidate_count == 0
            ):
                metadata["archive_skip_reason"] = "no eligible feasible layouts"
            else:
                metadata["archive_skip_reason"] = "; ".join(archive_skip_reasons)
        except Exception as archive_exc:
            metadata["archive_error_message"] = f"{type(archive_exc).__name__}: {archive_exc}"
            raise
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        if not metadata.get("archive_error_message") and "archive" in str(exc).lower():
            metadata["archive_error_message"] = error_message
        run_dir.mkdir(parents=True, exist_ok=True)

    runtime_seconds = time.perf_counter() - started
    end_time = datetime.now(timezone.utc)
    metadata.update(
        {
            "end_time": end_time.isoformat(),
            "runtime_seconds": runtime_seconds,
            "status": status,
            "error_message": error_message,
        }
    )
    if profile_light or save_generation_objectives:
        _write_profile_outputs(
            run=run,
            run_dir=run_dir,
            ablation_variant=ablation_variant,
            total_runtime_seconds=runtime_seconds,
            profile_times=profile_times,
            generation_profile_rows=(
                generation_profile_rows
                if generation_profile_rows
                else _generation_profile_rows_from_existing(
                    run,
                    ablation_variant,
                    generation_rows,
                    candidate_rows,
                )
            ),
            generation_objective_rows=(
                generation_objective_rows
                if generation_objective_rows
                else _generation_objective_rows(
                    run,
                    ablation_variant,
                    candidate_rows,
                )
            ),
            profile_light=profile_light,
            save_generation_objectives=save_generation_objectives,
        )
    _write_json(run_metadata_path, metadata)

    return experiment_summary_row(
        run,
        candidate_rows,
        generation_rows,
        runtime_seconds,
        status,
        error_message=error_message,
    )


def run_experiment_manager(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    instances: str | list[str] | None = None,
    seeds: str | None = None,
    method: str = METHOD_NAME,
    population_size: int | None = None,
    generations: int | None = None,
    beam_width: int | None = None,
    max_depth: int | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    save_figures: bool = True,
    dry_run: bool = False,
    archive_layouts: str = "none",
    archive_rank_max: int = 3,
    budget_policy: str = DEFAULT_BUDGET_POLICY,
    sorting_rule_mode: str = nsga2_bs.DEFAULT_SORTING_RULE_MODE,
    sorting_rule: str = nsga2_bs.DEFAULT_SORTING_RULE,
    adaptive_weight_mode: str = nsga2_bs.DEFAULT_ADAPTIVE_WEIGHT_MODE,
    fixed_beam_w1: float = bs_only_direct.FIXED_BS_WEIGHTS["w1"],
    fixed_beam_w2: float = bs_only_direct.FIXED_BS_WEIGHTS["w2"],
    fixed_beam_lambda: float = bs_only_direct.FIXED_BS_WEIGHTS["lambda"],
    mutation_mode: str = nsga2_bs.DEFAULT_MUTATION_MODE,
    initialization_spacing_mode: str = nsga2_bs.DEFAULT_INITIALIZATION_MODE,
    ablation_variant: str = DEFAULT_ABLATION_VARIANT,
    bs_rule_policy: str = bs_only_direct.DEFAULT_BS_RULE_POLICY,
    bs_weight_policy: str = bs_only_direct.DEFAULT_BS_WEIGHT_POLICY,
    decode_budget: int | None = None,
    experiment_id: str | None = None,
    command_line: list[str] | None = None,
    beam_width_delta: int = 0,
    profile_light: bool = False,
    save_generation_objectives: bool = False,
) -> ExperimentManagerResult:
    """Run or dry-run the experiment manager skeleton."""
    if budget_policy not in BUDGET_POLICIES:
        raise ValueError(f"budget_policy must be one of {BUDGET_POLICIES}.")
    if archive_layouts not in ARCHIVE_LAYOUT_MODES:
        raise ValueError(f"archive_layouts must be one of {ARCHIVE_LAYOUT_MODES}.")
    if archive_rank_max < 0:
        raise ValueError("archive_rank_max must be non-negative.")
    if sorting_rule_mode not in nsga2_bs.SORTING_RULE_MODES:
        raise ValueError(f"sorting_rule_mode must be one of {nsga2_bs.SORTING_RULE_MODES}.")
    if adaptive_weight_mode not in nsga2_bs.ADAPTIVE_WEIGHT_MODES:
        raise ValueError(
            f"adaptive_weight_mode must be one of {nsga2_bs.ADAPTIVE_WEIGHT_MODES}."
        )
    if mutation_mode not in nsga2_bs.MUTATION_MODES:
        raise ValueError(f"mutation_mode must be one of {nsga2_bs.MUTATION_MODES}.")
    if initialization_spacing_mode not in nsga2_bs.INITIALIZATION_SPACING_MODES:
        raise ValueError(
            "initialization_spacing_mode must be one of "
            f"{nsga2_bs.INITIALIZATION_SPACING_MODES}."
        )
    if bs_rule_policy not in bs_only_direct.BS_RULE_POLICIES:
        raise ValueError(f"bs_rule_policy must be one of {bs_only_direct.BS_RULE_POLICIES}.")
    if bs_weight_policy not in bs_only_direct.BS_WEIGHT_POLICIES:
        raise ValueError(f"bs_weight_policy must be one of {bs_only_direct.BS_WEIGHT_POLICIES}.")

    config = load_experiment_config(config_path)
    overrides = {
        "population_size": population_size,
        "generations": generations,
        "beam_width": beam_width,
        "max_depth": max_depth,
    }
    if decode_budget is not None and int(decode_budget) <= 0:
        raise ValueError("decode_budget must be positive.")
    if method == RANDOM_RESTART_BS_METHOD_NAME and decode_budget is not None:
        overrides["decode_budget"] = int(decode_budget)
    if budget_policy == "auto_from_instance":
        if beam_width_delta < 0:
            raise ValueError("beam_width_delta must be non-negative.")
    elif beam_width_delta:
        raise ValueError("beam_width_delta requires budget_policy=auto_from_instance.")
    planned = build_plan(
        config,
        instances_arg=instances,
        seeds_arg=seeds,
        method=method,
        overrides=overrides,
        budget_policy=budget_policy,
        beam_width_delta=beam_width_delta,
    )
    resolved_parameters = (
        dict(planned[0].parameters)
        if budget_policy == "auto_from_instance" and planned
        else resolve_parameters(config, method, overrides)
    )
    experiment_id = experiment_id or f"{method}_{utc_timestamp()}"
    experiment_dir = Path(output_dir) / experiment_id

    print(f"experiment_id={experiment_id}")
    print(f"method={method}")
    print(f"planned_runs={len(planned)}")
    for run in planned:
        print(
            f"planned run_id={run.run_id} instance={run.instance.name} "
            f"seed={run.seed} mask={run.instance.mask_path} "
            f"budget_policy={run.budget_policy} "
            f"parameter_source={run.parameter_source} "
            f"parameters={run.parameters}"
        )

    if dry_run:
        return ExperimentManagerResult(
            experiment_id=experiment_id,
            output_dir=experiment_dir,
            planned_runs=planned,
            dry_run=True,
        )

    ensure_experiment_dirs(experiment_dir)
    _write_json(
        experiment_dir / "experiment_metadata.json",
        experiment_metadata(
            experiment_id,
            method,
            planned,
            resolved_parameters,
            overrides,
            archive_layouts=archive_layouts,
            archive_rank_max=archive_rank_max,
            sorting_rule_mode=sorting_rule_mode,
            sorting_rule=sorting_rule,
            adaptive_weight_mode=adaptive_weight_mode,
            fixed_beam_w1=fixed_beam_w1,
            fixed_beam_w2=fixed_beam_w2,
            fixed_beam_lambda=fixed_beam_lambda,
            mutation_mode=mutation_mode,
            initialization_spacing_mode=initialization_spacing_mode,
            ablation_variant=ablation_variant,
            bs_rule_policy=bs_rule_policy,
            bs_weight_policy=bs_weight_policy,
            decode_budget=decode_budget,
            profile_light=profile_light,
            save_generation_objectives=save_generation_objectives,
            command_line=command_line,
        ),
    )

    summary_rows: list[dict[str, Any]] = []
    for run in planned:
        summary_rows.append(
            _run_one_planned_run(
                run,
                experiment_dir,
                save_figures,
                archive_layouts,
                archive_rank_max,
                sorting_rule_mode,
                sorting_rule,
                adaptive_weight_mode,
                fixed_beam_w1,
                fixed_beam_w2,
                fixed_beam_lambda,
                mutation_mode,
                initialization_spacing_mode,
                ablation_variant,
                bs_rule_policy,
                bs_weight_policy,
                decode_budget,
                profile_light,
                save_generation_objectives,
            )
        )

    _write_csv(
        experiment_dir / "experiment_summary.csv",
        summary_rows,
        EXPERIMENT_SUMMARY_COLUMNS,
    )
    return ExperimentManagerResult(
        experiment_id=experiment_id,
        output_dir=experiment_dir,
        planned_runs=planned,
        summary_rows=summary_rows,
        dry_run=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run config-driven warehouse layout experiments.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Instance names or paths. Accepts space-separated values or comma-separated lists.",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--method", default=METHOD_NAME)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument(
        "--decode-budget",
        type=int,
        default=None,
        help="Total random restarts for random_restart_bs. Defaults to population-size x generations.",
    )
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--beam-width-delta",
        type=int,
        default=0,
        help="Increase auto beam width by this amount when budget-policy=auto_from_instance.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--budget-policy",
        choices=BUDGET_POLICIES,
        default=DEFAULT_BUDGET_POLICY,
    )
    parser.add_argument(
        "--sorting-rule-mode",
        choices=nsga2_bs.SORTING_RULE_MODES,
        default=nsga2_bs.DEFAULT_SORTING_RULE_MODE,
    )
    parser.add_argument("--sorting-rule", default=nsga2_bs.DEFAULT_SORTING_RULE)
    parser.add_argument(
        "--fixed-sorting-rule",
        dest="sorting_rule",
        help="Alias for --sorting-rule when sorting-rule-mode=fixed.",
    )
    parser.add_argument(
        "--adaptive-weight-mode",
        choices=nsga2_bs.ADAPTIVE_WEIGHT_MODES,
        default=nsga2_bs.DEFAULT_ADAPTIVE_WEIGHT_MODE,
    )
    parser.add_argument(
        "--fixed-beam-w1",
        type=float,
        default=bs_only_direct.FIXED_BS_WEIGHTS["w1"],
    )
    parser.add_argument(
        "--fixed-beam-w2",
        type=float,
        default=bs_only_direct.FIXED_BS_WEIGHTS["w2"],
    )
    parser.add_argument(
        "--fixed-beam-lambda",
        type=float,
        default=bs_only_direct.FIXED_BS_WEIGHTS["lambda"],
    )
    parser.add_argument(
        "--mutation-mode",
        choices=nsga2_bs.MUTATION_MODES,
        default=nsga2_bs.DEFAULT_MUTATION_MODE,
    )
    parser.add_argument(
        "--initialization-spacing-mode",
        choices=nsga2_bs.INITIALIZATION_SPACING_MODES,
        default=nsga2_bs.DEFAULT_INITIALIZATION_MODE,
    )
    parser.add_argument("--ablation-variant", default=DEFAULT_ABLATION_VARIANT)
    parser.add_argument(
        "--bs-rule-policy",
        choices=bs_only_direct.BS_RULE_POLICIES,
        default=bs_only_direct.DEFAULT_BS_RULE_POLICY,
    )
    parser.add_argument(
        "--bs-weight-policy",
        choices=bs_only_direct.BS_WEIGHT_POLICIES,
        default=bs_only_direct.DEFAULT_BS_WEIGHT_POLICY,
    )
    parser.add_argument(
        "--archive-layouts",
        choices=ARCHIVE_LAYOUT_MODES,
        default="none",
    )
    parser.add_argument("--archive-rank-max", type=int, default=3)
    parser.add_argument(
        "--profile-light",
        action="store_true",
        help="Write lightweight runtime_profile_summary.csv and generation_profile.csv files.",
    )
    parser.add_argument(
        "--save-generation-objectives",
        action="store_true",
        help="Write generation_objectives.csv with rank-0 objective rows per generation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_experiment_manager(
        config_path=args.config,
        instances=args.instances,
        seeds=args.seeds,
        method=args.method,
        population_size=args.population_size,
        generations=args.generations,
        decode_budget=args.decode_budget,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
        dry_run=args.dry_run,
        archive_layouts=args.archive_layouts,
        archive_rank_max=args.archive_rank_max,
        budget_policy=args.budget_policy,
        experiment_id=args.experiment_id,
        sorting_rule_mode=args.sorting_rule_mode,
        sorting_rule=args.sorting_rule,
        adaptive_weight_mode=args.adaptive_weight_mode,
        fixed_beam_w1=args.fixed_beam_w1,
        fixed_beam_w2=args.fixed_beam_w2,
        fixed_beam_lambda=args.fixed_beam_lambda,
        mutation_mode=args.mutation_mode,
        initialization_spacing_mode=args.initialization_spacing_mode,
        ablation_variant=args.ablation_variant,
        bs_rule_policy=args.bs_rule_policy,
        bs_weight_policy=args.bs_weight_policy,
        beam_width_delta=args.beam_width_delta,
        profile_light=args.profile_light,
        save_generation_objectives=args.save_generation_objectives,
        command_line=sys.argv,
    )
    if args.dry_run:
        print("Dry run complete. No optimization executed.")
    else:
        print(f"Experiment output: {result.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_COLUMNS",
    "ARCHIVE_LAYOUT_MODES",
    "BS_DEPTH_SUMMARY_COLUMNS",
    "BS_ONLY_DIRECT_METHOD_NAME",
    "BUDGET_POLICIES",
    "DEFAULT_BUDGET_POLICY",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_BS_ONLY_DIRECT_PARAMETERS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PARAMETERS",
    "DEFAULT_RANDOM_RESTART_BS_PARAMETERS",
    "EXPERIMENT_SUMMARY_COLUMNS",
    "ExperimentInstance",
    "ExperimentManagerResult",
    "GENERATION_OBJECTIVE_COLUMNS",
    "GENERATION_PROFILE_COLUMNS",
    "GENERATION_SUMMARY_COLUMNS",
    "METHOD_NAME",
    "PlannedRun",
    "PROFILE_TIME_KEYS",
    "RANDOM_RESTART_BS_METHOD_NAME",
    "RUNTIME_PROFILE_SUMMARY_COLUMNS",
    "SUPPORTED_METHODS",
    "ADAPTIVE_SPACING_MODE",
    "ADAPTIVE_SPACING_ALPHA",
    "ADAPTIVE_SPACING_BF",
    "DEFAULT_ABLATION_VARIANT",
    "AUTO_PARAMS_USED",
    "build_parser",
    "build_plan",
    "budget_fields_for_run",
    "ensure_experiment_dirs",
    "experiment_metadata",
    "experiment_summary_row",
    "generation_summary_row",
    "load_experiment_config",
    "resolve_instances",
    "resolve_parameter_source",
    "resolve_parameters",
    "resolve_seeds",
    "run_experiment_manager",
    "write_all_debug_archive",
    "write_all_candidates_debug_archive",
    "write_final_ranked_archive",
    "write_generation_elites_archive",
]
