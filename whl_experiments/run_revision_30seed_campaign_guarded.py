"""Guarded orchestration for the paper's structural revision campaigns.

This entry point reuses the canonical task construction and campaign semantics
from :mod:`whl_experiments.run_revision_30seed_campaign`, while adding
orchestration safeguards for long replicated campaigns:

* strict validation of critical argv token boundaries before launch;
* unambiguous JSON argv logging alongside the shell-form command;
* a conservative per-task timeout (default: 3 hours);
* explicit ``timed_out``/``failed`` manifest rows rather than indefinite wait;
* exception containment at the worker/future boundary.

No optimization semantics, seeds, budgets, objectives, or archive settings are
changed. Timed-out tasks are NOT retried automatically; reruns remain explicit
and can use the existing ``--resume`` contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from whl_experiments import run_revision_30seed_campaign as base

DEFAULT_TASK_TIMEOUT_SECONDS = 10_800  # 3 h; well above observed completed runs.
TIMEOUT_RETURN_CODE = 124


def _arg_value(command: tuple[str, ...], flag: str) -> str:
    """Return the single value following *flag*, rejecting malformed argv."""
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one {flag!r} token; found {len(positions)}.")
    index = positions[0]
    if index + 1 >= len(command):
        raise ValueError(f"Missing value after {flag!r}.")
    value = command[index + 1]
    if value.startswith("--"):
        raise ValueError(f"Missing value after {flag!r}; next token is {value!r}.")
    return value


def validate_task_command(task: base.CampaignTask) -> None:
    """Fail before launch if critical task identity tokens are malformed."""
    command = task.command

    seed_value = _arg_value(command, "--seeds")
    if seed_value != str(task.seed):
        raise ValueError(
            f"Seed argv mismatch: task seed={task.seed}, --seeds value={seed_value!r}."
        )

    expected_method = base.method_for_task(task.phase, task.method_or_variant)
    method_value = _arg_value(command, "--method")
    if method_value != expected_method:
        raise ValueError(
            f"Method argv mismatch: expected {expected_method!r}, got {method_value!r}."
        )

    expected_experiment_id = base.task_identifier(
        task.phase, task.method_or_variant, task.instance, task.seed
    )
    experiment_id = _arg_value(command, "--experiment-id")
    if experiment_id != expected_experiment_id:
        raise ValueError(
            "Experiment-id argv mismatch: "
            f"expected {expected_experiment_id!r}, got {experiment_id!r}."
        )

    # Detect the exact class of anomaly observed in the incident log, e.g.
    # ``113--method``. Flags must occupy their own argv tokens.
    embedded_flags = [
        token for token in command
        if token != "--method" and "--method" in token
    ]
    if embedded_flags:
        raise ValueError(
            "Malformed argv: '--method' is embedded in another token: "
            + ", ".join(repr(token) for token in embedded_flags)
        )


def _failure_row(
    task: base.CampaignTask,
    *,
    status: str,
    started_at: str,
    start_perf: float,
    return_code: int,
    error_message: str,
) -> dict[str, Any]:
    return base.manifest_row(
        task,
        status=status,
        started_at=started_at,
        finished_at=base.utc_now(),
        runtime_seconds=time.perf_counter() - start_perf,
        return_code=return_code,
        error_message=error_message,
    )


def run_task(
    task: base.CampaignTask,
    resume: bool,
    task_timeout_seconds: int,
) -> dict[str, Any]:
    """Run one campaign task with validation and a bounded wait."""
    started_at = base.utc_now()
    start_perf = time.perf_counter()

    if resume and base.completed_successfully(task):
        return base.manifest_row(
            task,
            status="skipped_resume",
            started_at=started_at,
            finished_at=base.utc_now(),
            runtime_seconds=0.0,
            return_code=0,
            error_message="existing completed experiment_summary.csv",
        )

    try:
        validate_task_command(task)
    except (TypeError, ValueError) as exc:
        return _failure_row(
            task,
            status="failed",
            started_at=started_at,
            start_perf=start_perf,
            return_code=2,
            error_message=f"command validation failed: {exc}",
        )

    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with task.log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"command: {subprocess.list2cmdline(task.command)}\n")
            log_file.write(
                "argv_json: "
                + json.dumps(list(task.command), ensure_ascii=False)
                + "\n"
            )
            log_file.write(f"task_timeout_seconds: {task_timeout_seconds}\n\n")
            log_file.flush()

            try:
                result = subprocess.run(
                    list(task.command),
                    cwd=Path.cwd(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=task_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                log_file.write(
                    "\nGUARD: task exceeded timeout; direct subprocess was terminated.\n"
                )
                log_file.flush()
                return _failure_row(
                    task,
                    status="timed_out",
                    started_at=started_at,
                    start_perf=start_perf,
                    return_code=TIMEOUT_RETURN_CODE,
                    error_message=(
                        f"task exceeded {task_timeout_seconds} seconds; "
                        "no automatic retry performed"
                    ),
                )
    except OSError as exc:
        return _failure_row(
            task,
            status="failed",
            started_at=started_at,
            start_perf=start_perf,
            return_code=1,
            error_message=f"orchestration I/O error: {exc}",
        )

    return base.manifest_row(
        task,
        status="completed" if result.returncode == 0 else "failed",
        started_at=started_at,
        finished_at=base.utc_now(),
        runtime_seconds=time.perf_counter() - start_perf,
        return_code=result.returncode,
        error_message="" if result.returncode == 0 else "see log",
    )


def execute(
    tasks: list[base.CampaignTask],
    *,
    max_workers: int,
    resume: bool,
    manifest_path: Path,
    task_timeout_seconds: int,
) -> None:
    """Execute tasks while ensuring every finished future yields a manifest row."""
    failed = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_task, task, resume, task_timeout_seconds): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:  # defensive orchestration boundary
                row = base.manifest_row(
                    task,
                    status="failed",
                    started_at="",
                    finished_at=base.utc_now(),
                    runtime_seconds="",
                    return_code=1,
                    error_message=f"unexpected worker exception: {type(exc).__name__}: {exc}",
                )

            base.append_row(manifest_path, row)
            print(
                f"{row['phase']} {row['method_or_variant']} {row['instance']} "
                f"seed={row['seed']} status={row['status']}"
            )
            failed = failed or row["status"] not in {"completed", "skipped_resume"}

    if failed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help=(
            "Maximum wall-clock seconds for one subprocess before it is recorded "
            "as timed_out. Default: 10800 (3 hours)."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base.validate_args(args, parser)
    if args.task_timeout_seconds <= 0:
        parser.error("--task-timeout-seconds must be positive.")

    base.validate_no_figures_cli_contract()
    tasks = base.build_tasks(args)
    if not tasks:
        parser.error("No tasks selected. Check --phase, --only-variant, and instance options.")

    # Validate argv even for dry runs. This makes command-boundary regressions fail
    # immediately without launching any optimization task.
    for task in tasks:
        try:
            validate_task_command(task)
        except (TypeError, ValueError) as exc:
            parser.error(f"Invalid generated command for seed {task.seed}: {exc}")

    manifest_dir = Path(args.campaign_root) / "manifests"
    if args.dry_run:
        path = manifest_dir / f"dry_run_{args.phase}.csv"
        rows = [base.manifest_row(task, status="dry_run") for task in tasks]
        base.write_rows(path, rows)
        print(f"dry_run_tasks={len(rows)}")
        print(f"dry_run_task_list={path}")
        return

    execute(
        tasks,
        max_workers=args.max_workers,
        resume=args.resume,
        manifest_path=manifest_dir / "batch_manifest.csv",
        task_timeout_seconds=args.task_timeout_seconds,
    )


if __name__ == "__main__":
    main()
