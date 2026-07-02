"""Batch renderer for archived layouts from an experiment folder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from whl_experiments.render_saved_layouts import (
    ARCHIVE_FILTERS,
    TITLE_FORMATS,
    render_saved_layouts,
)

ARCHIVE_TYPE_FILES = {
    "final_ranked": (
        "final_ranked_layouts.npz",
        "final_ranked_layouts_index.json",
    ),
    "generation_elites": (
        "generation_elites.npz",
        "generation_elites_index.json",
    ),
    "all_debug": (
        "all_debug_layouts.npz",
        "all_debug_layouts_index.json",
    ),
    "all_candidates_debug": (
        "all_candidates_debug_layouts.npz",
        "all_candidates_debug_layouts_index.json",
    ),
}


@dataclass(frozen=True)
class ArchiveRenderJob:
    """One archive/index pair discovered under an experiment run folder."""

    archive_path: str
    index_path: str
    output_dir: str
    instance: str
    seed_folder: str
    skip_reason: str | None = None


def _parse_title_fields(value: str | None) -> list[str] | None:
    if not value:
        return None
    fields = [item.strip() for item in value.split(",") if item.strip()]
    return fields or None


def archive_filenames(archive_type: str) -> tuple[str, str]:
    """Return archive and index filenames for a supported archive type."""
    if archive_type not in ARCHIVE_TYPE_FILES:
        raise ValueError(f"archive_type must be one of {tuple(ARCHIVE_TYPE_FILES)}.")
    return ARCHIVE_TYPE_FILES[archive_type]


def discover_archive_jobs(
    experiment_dir: Path | str,
    archive_type: str = "final_ranked",
    output_dir: Path | str | None = None,
    filter_name: str = "rank0_to_rank3",
) -> list[ArchiveRenderJob]:
    """Discover archive render jobs under ``experiment_dir/runs``."""
    archive_name, index_name = archive_filenames(archive_type)
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        raise FileNotFoundError(f"experiment-dir does not exist: {experiment_path}")
    if not experiment_path.is_dir():
        raise NotADirectoryError(f"experiment-dir is not a directory: {experiment_path}")

    runs_dir = experiment_path / "runs"
    if not runs_dir.exists():
        return []

    jobs: list[ArchiveRenderJob] = []
    for archive_path in sorted(runs_dir.rglob(archive_name)):
        run_dir = archive_path.parent
        index_path = run_dir / index_name
        seed_folder = run_dir.name
        instance = run_dir.parent.name
        if output_dir is None:
            job_output_dir = run_dir / "figures" / archive_type / filter_name
        else:
            job_output_dir = Path(output_dir) / archive_type / instance / seed_folder
        skip_reason = None if index_path.exists() else "missing_index_json"
        jobs.append(
            ArchiveRenderJob(
                archive_path=str(archive_path),
                index_path=str(index_path),
                output_dir=str(job_output_dir),
                instance=instance,
                seed_folder=seed_folder,
                skip_reason=skip_reason,
            )
        )
    return jobs


def render_experiment_archives(
    experiment_dir: Path | str,
    archive_type: str = "final_ranked",
    filter_name: str = "rank0_to_rank3",
    output_dir: Path | str | None = None,
    max_layouts: int | None = None,
    dpi: int = 150,
    title_fields: list[str] | None = None,
    title_format: str = "fields",
    show_legend: bool = True,
    show_coords: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    write_summary: bool = True,
) -> dict[str, Any]:
    """Render all matching archives under one experiment directory."""
    if filter_name not in ARCHIVE_FILTERS:
        raise ValueError(f"filter_name must be one of {ARCHIVE_FILTERS}.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    jobs = discover_archive_jobs(
        experiment_dir=experiment_dir,
        archive_type=archive_type,
        output_dir=output_dir,
        filter_name=filter_name,
    )
    rendered_jobs: list[dict[str, Any]] = []
    skipped_jobs: list[dict[str, Any]] = []

    for job in jobs:
        if job.skip_reason:
            print(
                "Skipping archive with missing index: "
                f"archive={job.archive_path} index={job.index_path}"
            )
            skipped_jobs.append(asdict(job))
            continue

        print(f"archive={job.archive_path}")
        print(f"index={job.index_path}")
        print(f"output_dir={job.output_dir}")
        if dry_run:
            continue

        rendered = render_saved_layouts(
            archive=job.archive_path,
            index=job.index_path,
            output_dir=job.output_dir,
            filter_name=filter_name,
            max_layouts=max_layouts,
            dpi=dpi,
            title_fields=title_fields,
            title_format=title_format,
            show_legend=show_legend,
            show_coords=show_coords,
        )
        job_record = asdict(job)
        job_record["rendered_layout_count"] = len(rendered)
        job_record["rendered_paths"] = [str(path) for path in rendered]
        job_record["overwrite"] = bool(overwrite)
        rendered_jobs.append(job_record)
        print(f"rendered_layouts={len(rendered)}")

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(Path(experiment_dir)),
        "archive_type": archive_type,
        "filter": filter_name,
        "dry_run": bool(dry_run),
        "max_layouts": max_layouts,
        "dpi": dpi,
        "title_format": title_format,
        "show_legend": bool(show_legend),
        "show_coords": bool(show_coords),
        "jobs_discovered": len(jobs),
        "rendered_jobs": len(rendered_jobs),
        "skipped_jobs": len(skipped_jobs),
        "output_dir": str(Path(output_dir)) if output_dir is not None else None,
        "generated_output_dirs": sorted({job["output_dir"] for job in rendered_jobs}),
        "jobs": [asdict(job) for job in jobs],
        "rendered": rendered_jobs,
        "skipped": skipped_jobs,
    }

    if write_summary:
        summary_dir = Path(output_dir) if output_dir is not None else Path(experiment_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{archive_type}_render_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)

    print(f"experiment_dir={summary['experiment_dir']}")
    print(f"archive_type={archive_type}")
    print(f"archives_discovered={len(jobs)}")
    print(f"rendered_successfully={len(rendered_jobs)}")
    print(f"skipped={len(skipped_jobs)}")
    print(f"output_dir={summary['output_dir']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-render archived layouts from an experiment folder.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to a completed experiment folder.",
    )
    parser.add_argument(
        "--archive-type",
        choices=tuple(ARCHIVE_TYPE_FILES),
        default="final_ranked",
    )
    parser.add_argument("--filter", choices=ARCHIVE_FILTERS, default="rank0_to_rank3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional root output directory. If omitted, figures are saved inside each run folder.",
    )
    parser.add_argument("--max-layouts", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--title-fields",
        default=None,
        help="Comma-separated metadata/metric fields to include in titles.",
    )
    parser.add_argument(
        "--title-format",
        choices=TITLE_FORMATS,
        default="fields",
        help="Title format. Use metrics_trace for Step 10B archive figures.",
    )
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--show-coords", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print planned render jobs without writing PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for explicit intent; the current single-archive renderer overwrites existing PNGs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        render_experiment_archives(
            experiment_dir=args.experiment_dir,
            archive_type=args.archive_type,
            filter_name=args.filter,
            output_dir=args.output_dir,
            max_layouts=args.max_layouts,
            dpi=args.dpi,
            title_fields=_parse_title_fields(args.title_fields),
            title_format=args.title_format,
            show_legend=not args.no_legend,
            show_coords=args.show_coords,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_TYPE_FILES",
    "ArchiveRenderJob",
    "archive_filenames",
    "build_parser",
    "discover_archive_jobs",
    "render_experiment_archives",
]
