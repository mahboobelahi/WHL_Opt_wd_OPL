"""Reviewer-friendly orchestration for the paper's structural experiment campaign.

This module contains no optimization logic. Each task delegates to
``whl_experiments.run_experiment_manager`` with the documented public options.
Figures are disabled for campaign runs to reduce unnecessary I/O.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whl_experiments import run_experiment_manager as experiment_manager


DEFAULT_CAMPAIGN_ROOT = Path("results/revision_30seed_campaign")

CORE_INSTANCES = (
    "AT_S_comercial_layout_AW_3",
    "demo_layout_door_left_AW_2",
    "Gyorgy-KOVACS_WH_Narrow_AW_4",
    "Gyorgy-KOVACS_WH_Wide_AW_5",
)
STRESS_INSTANCES = (
    "Gyorgy-KOVACS_MWH_Narrow_AW_4",
    "Answer_Set_layout_AW_2",
)

PHASE11_METHODS = (
    "proposed_nsga2_bs",
    "random_restart_bs",
    "bs_only",
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
    "V6_depth15_beam_default",
    "V7_beam_plus1_depth_default",
)
V6B_VARIANT = "V6b_binding_depth10"

PHASE_SLUGS = {
    "phase11": "p11",
    "phase12b": "p12b",
    "phase12c": "p12c",
}
METHOD_OR_VARIANT_SLUGS = {
    "proposed_nsga2_bs": "nsga2",
    "random_restart_bs": "rrbs",
    "bs_only": "bsonly",
    "V0_full_proposed": "V0",
    "V1_fixed_sorting": "V1_fixsort",
    "V2_fixed_weights": "V2_fixw",
    "V3_uniform_mutation": "V3_um",
    "V4_no_symmetry_breaking": "V4_nsb",
    "V5_random_feasible_start_spacing": "V5_rfs",
    "V6_depth15_beam_default": "V6_d15",
    "V6b_binding_depth10": "V6b_d10",
    "V7_beam_plus1_depth_default": "V7_bw1",
}
INSTANCE_SLUGS = {
    "AT_S_comercial_layout_AW_3": "AT_S_AW3",
    "Answer_Set_layout_AW_1": "ANS_AW1",
    "Answer_Set_layout_AW_2": "ANS_AW2",
    "Answer_Set_layout_AW_3": "ANS_AW3",
    "demo_layout_door_bottom_AW_2": "DEMO_B_AW2",
    "demo_layout_door_bottom_AW_3": "DEMO_B_AW3",
    "demo_layout_door_left_AW_2": "DEMO_L_AW2",
    "demo_layout_door_left_AW_3": "DEMO_L_AW3",
    "demo_layout_door_UB_AW_2": "DEMO_UB_AW2",
    "demo_layout_door_UB_AW_3": "DEMO_UB_AW3",
    "Gyorgy-KOVACS_WH_Narrow_AW_4": "KOV_WH_N_AW4",
    "Gyorgy-KOVACS_WH_Wide_AW_5": "KOV_WH_W_AW5",
    "Gyorgy-KOVACS_MWH_Narrow_AW_4": "KOV_MWH_N_AW4",
    "Gyorgy-KOVACS_MWH_Wide_AW_5": "KOV_MWH_W_AW5",
}

MANIFEST_COLUMNS = [
    "phase",
    "method_or_variant",
    "instance",
    "seed",
    "command",
    "status",
    "started_at",
    "finished_at",
    "runtime_seconds",
    "return_code",
    "output_dir",
    "log_path",
    "error_message",
]


@dataclass(frozen=True)
class CampaignTask:
    phase: str
    method_or_variant: str
    instance: str
    seed: int
    command: tuple[str, ...]
    output_dir: Path
    log_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def selected_instances(scope: str) -> tuple[str, ...]:
    if scope == "core":
        return CORE_INSTANCES
    if scope == "stress":
        return STRESS_INSTANCES
    if scope == "all":
        return CORE_INSTANCES + STRESS_INSTANCES
    raise ValueError(f"Unsupported instance scope: {scope}")


def parse_instance_list(value: str | None) -> tuple[str, ...]:
    """Return unique comma-separated repository instance names in input order."""
    if not value:
        return ()
    instances: list[str] = []
    seen: set[str] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        # The manager resolves repository masks by stem; accepting an optional
        # .npz suffix here keeps the campaign CLI convenient without changing
        # the underlying experiment-manager contract.
        instance = Path(item).stem
        if instance not in seen:
            seen.add(instance)
            instances.append(instance)
    return tuple(instances)


def phase_members(phase: str) -> tuple[str, ...]:
    if phase == "phase11":
        return PHASE11_METHODS
    if phase == "phase12b":
        return PHASE12B_VARIANTS
    if phase == "phase12c":
        return PHASE12C_VARIANTS
    raise ValueError(f"Unsupported phase: {phase}")


def method_for_task(phase: str, method_or_variant: str) -> str:
    if phase == "phase11":
        return method_or_variant
    return "proposed_nsga2_bs"


def phase_specific_flags(phase: str, method_or_variant: str) -> list[str]:
    if phase == "phase11":
        return []
    if phase == "phase12b":
        return phase12b_flags(method_or_variant)
    if phase == "phase12c":
        return phase12c_flags(method_or_variant)
    raise ValueError(f"Unsupported phase: {phase}")


def phase12b_flags(variant: str) -> list[str]:
    label = variant if variant != "V1_fixed_sorting" else "V1_fixed_sorting_PF_LS_RP"
    base = ["--ablation-variant", label]
    if variant == "V0_full_proposed":
        return base + [
            "--sorting-rule-mode", "sampled_pool",
            "--adaptive-weight-mode", "adaptive",
            "--mutation-mode", "weighted",
            "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        ]
    if variant == "V1_fixed_sorting":
        return base + [
            "--sorting-rule-mode", "fixed",
            "--fixed-sorting-rule", "PF_LS_RP",
            "--adaptive-weight-mode", "adaptive",
            "--mutation-mode", "weighted",
            "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        ]
    if variant == "V2_fixed_weights":
        return base + [
            "--sorting-rule-mode", "sampled_pool",
            "--adaptive-weight-mode", "fixed",
            "--fixed-beam-w1", "0.5",
            "--fixed-beam-w2", "0.5",
            "--fixed-beam-lambda", "0.1",
            "--mutation-mode", "weighted",
            "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        ]
    if variant == "V3_uniform_mutation":
        return base + [
            "--sorting-rule-mode", "sampled_pool",
            "--adaptive-weight-mode", "adaptive",
            "--mutation-mode", "uniform",
            "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        ]
    if variant == "V4_no_symmetry_breaking":
        return base + [
            "--sorting-rule-mode", "sampled_pool",
            "--adaptive-weight-mode", "adaptive",
            "--mutation-mode", "weighted_no_symmetry_breaking",
            "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        ]
    if variant == "V5_random_feasible_start_spacing":
        return base + [
            "--sorting-rule-mode", "sampled_pool",
            "--adaptive-weight-mode", "adaptive",
            "--mutation-mode", "weighted",
            "--initialization-spacing-mode", "random_feasible_start_no_adaptive_spacing",
        ]
    raise ValueError(f"Unsupported Phase 12B variant: {variant}")


def phase12c_flags(variant: str) -> list[str]:
    base = [
        "--sorting-rule-mode", "sampled_pool",
        "--adaptive-weight-mode", "adaptive",
        "--mutation-mode", "weighted",
        "--initialization-spacing-mode", "feasible_start_adaptive_spacing",
        "--ablation-variant", variant,
    ]
    if variant == "V6_depth15_beam_default":
        return base + ["--max-depth", "15"]
    if variant == V6B_VARIANT:
        return base + ["--max-depth", "10"]
    if variant == "V7_beam_plus1_depth_default":
        return base + ["--beam-width-delta", "1"]
    raise ValueError(f"Unsupported Phase 12C variant: {variant}")


def compact_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")[:24]


def task_identifier(phase: str, method_or_variant: str, instance: str, seed: int) -> str:
    return (
        f"{PHASE_SLUGS[phase]}__"
        f"{METHOD_OR_VARIANT_SLUGS.get(method_or_variant, compact_slug(method_or_variant))}__"
        f"{INSTANCE_SLUGS.get(instance, compact_slug(instance))}__s{int(seed)}"
    )


def validate_no_figures_cli_contract() -> None:
    """Fail fast if the manager's public ``--no-figures`` semantics regress."""
    parser = experiment_manager.build_parser()
    default_args = parser.parse_args([])
    disabled_args = parser.parse_args(["--no-figures"])
    if bool(default_args.no_figures):
        raise RuntimeError(
            "run_experiment_manager CLI contract invalid: default no_figures must be False."
        )
    if not bool(disabled_args.no_figures):
        raise RuntimeError(
            "run_experiment_manager CLI contract invalid: --no-figures must set no_figures=True."
        )


def build_command(
    *,
    phase: str,
    method_or_variant: str,
    instance: str,
    seed: int,
    output_base: Path,
    experiment_id: str,
    profile_light: bool,
    save_generation_objectives: bool,
    archive_rank_max: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "whl_experiments.run_experiment_manager",
        "--instances", instance,
        "--seeds", str(seed),
        "--method", method_for_task(phase, method_or_variant),
        "--experiment-id", experiment_id,
        "--output-dir", str(output_base),
        "--budget-policy", "auto_from_instance",
        "--archive-layouts", "both",
        "--archive-rank-max", str(int(archive_rank_max)),
        "--no-figures",
    ]
    if profile_light:
        command.append("--profile-light")
    if save_generation_objectives:
        command.append("--save-generation-objectives")
    command.extend(phase_specific_flags(phase, method_or_variant))
    return command


def build_tasks(args: argparse.Namespace) -> list[CampaignTask]:
    campaign_root = Path(args.campaign_root)
    members = phase_members(args.phase)
    if args.only_variant:
        if args.only_variant == V6B_VARIANT:
            members = (V6B_VARIANT,)
        else:
            members = tuple(item for item in members if item == args.only_variant)

    explicit_instances = parse_instance_list(args.instance_list)
    instances = explicit_instances or selected_instances(args.instances)
    if args.only_instance:
        instances = (args.only_instance,)

    tasks: list[CampaignTask] = []
    phase_dir = campaign_root / PHASE_SLUGS[args.phase]
    log_dir = campaign_root / "logs"
    for member in members:
        member_dir = phase_dir / METHOD_OR_VARIANT_SLUGS[member]
        for instance in instances:
            for seed in range(int(args.seed_start), int(args.seed_end) + 1):
                task_id = task_identifier(args.phase, member, instance, seed)
                output_dir = member_dir / task_id
                tasks.append(
                    CampaignTask(
                        phase=args.phase,
                        method_or_variant=member,
                        instance=instance,
                        seed=seed,
                        command=tuple(
                            build_command(
                                phase=args.phase,
                                method_or_variant=member,
                                instance=instance,
                                seed=seed,
                                output_base=member_dir,
                                experiment_id=task_id,
                                profile_light=args.profile_light,
                                save_generation_objectives=args.save_generation_objectives,
                                archive_rank_max=args.archive_rank_max,
                            )
                        ),
                        output_dir=output_dir,
                        log_path=log_dir / f"{task_id}.log",
                    )
                )
    return tasks


def completed_successfully(task: CampaignTask) -> bool:
    summary_path = task.output_dir / "experiment_summary.csv"
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return False
    expected_method = method_for_task(task.phase, task.method_or_variant)
    return any(
        row.get("method") == expected_method
        and row.get("instance") == task.instance
        and str(row.get("seed")) == str(task.seed)
        and row.get("status") == "completed"
        for row in rows
    )


def manifest_row(
    task: CampaignTask,
    *,
    status: str,
    started_at: str = "",
    finished_at: str = "",
    runtime_seconds: float | str = "",
    return_code: int | str = "",
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "phase": task.phase,
        "method_or_variant": task.method_or_variant,
        "instance": task.instance,
        "seed": task.seed,
        "command": subprocess.list2cmdline(task.command),
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        "output_dir": str(task.output_dir),
        "log_path": str(task.log_path),
        "error_message": error_message,
    }


def run_task(task: CampaignTask, resume: bool) -> dict[str, Any]:
    started_at = utc_now()
    if resume and completed_successfully(task):
        return manifest_row(
            task,
            status="skipped_resume",
            started_at=started_at,
            finished_at=utc_now(),
            runtime_seconds=0.0,
            return_code=0,
            error_message="existing completed experiment_summary.csv",
        )

    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with task.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"command: {subprocess.list2cmdline(task.command)}\n\n")
        log_file.flush()
        result = subprocess.run(
            list(task.command),
            cwd=Path.cwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    return manifest_row(
        task,
        status="completed" if result.returncode == 0 else "failed",
        started_at=started_at,
        finished_at=utc_now(),
        runtime_seconds=elapsed,
        return_code=result.returncode,
        error_message="" if result.returncode == 0 else "see log",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def execute(
    tasks: list[CampaignTask],
    *,
    max_workers: int,
    resume: bool,
    manifest_path: Path,
) -> None:
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    failed = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, task, resume) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            append_row(manifest_path, row)
            print(
                f"{row['phase']} {row['method_or_variant']} {row['instance']} "
                f"seed={row['seed']} status={row['status']}"
            )
            failed = failed or row["status"] == "failed"
    if failed:
        raise SystemExit(1)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start.")
    if args.archive_rank_max < 0:
        parser.error("--archive-rank-max must be non-negative.")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive.")
    if args.instance_list and not parse_instance_list(args.instance_list):
        parser.error("--instance-list must contain at least one instance name.")
    if args.instance_list and args.only_instance:
        parser.error("Use either --instance-list or --only-instance, not both.")
    if args.only_variant == V6B_VARIANT:
        if args.phase != "phase12c":
            parser.error(f"{V6B_VARIANT} requires --phase phase12c.")
        if args.instance_list:
            parser.error(f"{V6B_VARIANT} requires --only-instance, not --instance-list.")
        if args.only_instance != "demo_layout_door_left_AW_2":
            parser.error(
                f"{V6B_VARIANT} requires --only-instance demo_layout_door_left_AW_2."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("phase11", "phase12b", "phase12c"),
        required=True,
    )
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-end", type=int, default=130)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--instances",
        choices=("core", "stress", "all"),
        default="core",
        help="Use a predefined instance group unless --instance-list is supplied.",
    )
    parser.add_argument(
        "--instance-list",
        type=str,
        help=(
            "Comma-separated repository mask names for an explicit campaign subset. "
            "Optional .npz suffixes are accepted. This overrides --instances."
        ),
    )
    parser.add_argument(
        "--only-variant",
        choices=PHASE11_METHODS + PHASE12B_VARIANTS + PHASE12C_VARIANTS + (V6B_VARIANT,),
        help="Restrict execution to one Phase 11 method or Phase 12 variant.",
    )
    parser.add_argument(
        "--only-instance",
        choices=CORE_INSTANCES + STRESS_INSTANCES,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive-rank-max", type=int, default=3)
    parser.add_argument("--profile-light", action="store_true")
    parser.add_argument("--save-generation-objectives", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    validate_no_figures_cli_contract()
    tasks = build_tasks(args)
    if not tasks:
        parser.error("No tasks selected. Check --phase, --only-variant, and instance options.")

    campaign_root = Path(args.campaign_root)
    manifest_dir = campaign_root / "manifests"
    if args.dry_run:
        path = manifest_dir / f"dry_run_{args.phase}.csv"
        rows = [manifest_row(task, status="dry_run") for task in tasks]
        write_rows(path, rows)
        print(f"dry_run_tasks={len(rows)}")
        print(f"dry_run_task_list={path}")
        return

    manifest_path = manifest_dir / "batch_manifest.csv"
    execute(
        tasks,
        max_workers=int(args.max_workers),
        resume=bool(args.resume),
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
