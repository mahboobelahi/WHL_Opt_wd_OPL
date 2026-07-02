"""Generate manuscript Pareto/objective-space plots for the WHL repository.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
PLOT_INPUT_DIR = WORKSPACE_ROOT / "data" / "plot_inputs" / "paper"
OUTPUT_DIR = SCRIPT_DIR

# Input CSVs used by the manuscript Pareto plots. These are the 121-series
# rank-0-to-3 point files, not the unique-layout manifests.
ATEFEH_POINTS_CSV = PLOT_INPUT_DIR / "121_atefeh_rank03_points.csv"
KOV_POINTS_CSV = PLOT_INPUT_DIR / "121_kov1ow4_rank03_points.csv"
ATEFEH_PUBLISHED_CSV = PLOT_INPUT_DIR / "121_atefeh_published_reference_metrics.csv"


# -----------------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------------
RANK_COLORS = {
    0: "red",
    1: "#32a852",
    2: "#4663c2",
    3: "#d179e0",
}
RANKS = [0, 1, 2, 3]
RANK_DRAW_ORDER = [3, 2, 1, 0]

GENERATED_STORAGE_COLOR = "#4f86c6"
PARETO_LINE_COLOR = "#1b4f72"
PUBLISHED_COLOR = "black"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


# -----------------------------------------------------------------------------
# CSV and column handling
# -----------------------------------------------------------------------------
def resolve_input_path(preferred: Path) -> Path:
    """Resolve a 121-series CSV from the repository plotting-input folder."""
    candidates = [
        preferred,
        PLOT_INPUT_DIR / preferred.name,
        SCRIPT_DIR / preferred.name,
        WORKSPACE_ROOT / preferred.name,
        WORKSPACE_ROOT / "docs" / preferred.name,
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find input CSV {preferred.name}. Checked:\n{checked}")


def first_existing(columns: list[str], candidates: list[str], label: str) -> str:
    """Return the first available column name from a list of allowed aliases."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Missing required {label} column. Tried: {candidates}")


def load_rank_points(path: Path) -> pd.DataFrame:
    """Load rank 0-3 generated-layout points and normalize plotting columns."""
    raw = pd.read_csv(path)
    columns = list(raw.columns)

    rank_col = first_existing(columns, ["rank", "pareto_rank"], "rank")
    generation_col = first_existing(columns, ["generation", "logged_generation_index"], "generation")
    seed_col = first_existing(columns, ["seed", "random_seed"], "seed")
    pick_col = first_existing(columns, ["pick_faces", "n_pick_faces", "Npf"], "pick faces")
    ids_col = first_existing(
        columns,
        ["interior_storage", "interior_deep_storage", "deep_storage", "IDS"],
        "interior/deep storage",
    )
    retrieval_col = first_existing(
        columns,
        ["retrieval_penalty", "retrieval_cost", "RP"],
        "retrieval penalty",
    )
    capacity_col = first_existing(
        columns,
        ["storage_capacity", "storage_total", "capacity", "storage_cells"],
        "storage capacity",
    )

    df = pd.DataFrame(
        {
            "rank": raw[rank_col],
            "seed": raw[seed_col],
            "generation": raw[generation_col],
            "pick_faces": raw[pick_col],
            "interior_storage": raw[ids_col],
            "retrieval_penalty": raw[retrieval_col],
            "storage_capacity": raw[capacity_col],
        }
    )

    if "runtime_seconds" in raw.columns:
        df["runtime_seconds"] = raw["runtime_seconds"]
    if "source_candidates_csv" in raw.columns:
        df["source_candidates_csv"] = raw["source_candidates_csv"]

    df = df[df["rank"].isin(RANKS)].copy()
    df["rank"] = df["rank"].astype(int)
    return df


def load_published_atefeh(path: Path) -> pd.DataFrame:
    """Load published Atefeh reference-layout metrics and normalize columns."""
    raw = pd.read_csv(path)
    columns = list(raw.columns)

    pick_col = first_existing(columns, ["pick_faces", "n_pick_faces", "Npf"], "published pick faces")
    ids_col = first_existing(
        columns,
        ["interior_storage", "interior_deep_storage", "deep_storage", "IDS"],
        "published interior/deep storage",
    )
    retrieval_col = first_existing(
        columns,
        ["retrieval_penalty", "retrieval_cost", "RP"],
        "published retrieval penalty",
    )
    capacity_col = first_existing(
        columns,
        ["storage_capacity", "storage_total", "capacity", "storage_cells"],
        "published storage capacity",
    )

    return pd.DataFrame(
        {
            "pick_faces": raw[pick_col],
            "interior_storage": raw[ids_col],
            "retrieval_penalty": raw[retrieval_col],
            "storage_capacity": raw[capacity_col],
        }
    )


# -----------------------------------------------------------------------------
# Pareto-front computation
# -----------------------------------------------------------------------------
def compute_storage_pick_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the frontier maximizing storage capacity and pick faces.

    Duplicate pick-face values are collapsed by retaining the maximum storage
    capacity at that pick-face value. The non-dominated frontier is then found by
    scanning from high to low pick faces and keeping points whose storage
    capacity exceeds all previously scanned points. Returned points are sorted by
    increasing pick faces for plotting the line.
    """
    points = (
        df[["pick_faces", "storage_capacity"]]
        .dropna()
        .groupby("pick_faces", as_index=False)["storage_capacity"]
        .max()
        .sort_values("pick_faces", ascending=False)
    )

    frontier_rows = []
    best_capacity = -math.inf
    for _, row in points.iterrows():
        capacity = float(row["storage_capacity"])
        if capacity > best_capacity:
            frontier_rows.append(row)
            best_capacity = capacity

    if not frontier_rows:
        return pd.DataFrame(columns=["pick_faces", "storage_capacity"])

    return pd.DataFrame(frontier_rows).sort_values("pick_faces", ascending=True).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def padded_limits(*series: pd.Series, pad: float = 0.05, include_zero: bool = False) -> tuple[float, float] | None:
    values = []
    for item in series:
        if item is None:
            continue
        arr = np.asarray(item, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return None

    merged = np.concatenate(values)
    low = float(np.min(merged))
    high = float(np.max(merged))
    if math.isclose(low, high):
        delta = max(1.0, abs(high) * pad)
        low -= delta
        high += delta
    else:
        delta = (high - low) * pad
        low -= delta
        high += delta
    if include_zero:
        low = min(0.0, low)
    return low, high


def scatter_ranks_2d(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for rank in RANK_DRAW_ORDER:
        part = df[df["rank"] == rank]
        if part.empty:
            continue
        ax.scatter(
            part[x_col],
            part[y_col],
            marker="o",
            s=52,
            alpha=0.80,
            c=RANK_COLORS[rank],
            edgecolors="none",
            linewidths=0,
            label=f"Rank {rank}",
            zorder=10 - rank,
        )


def scatter_published_2d(ax: plt.Axes, published: pd.DataFrame, x_col: str, y_col: str) -> None:
    ax.scatter(
        published[x_col],
        published[y_col],
        marker="x",
        s=45,
        c=PUBLISHED_COLOR,
        alpha=0.9,
        linewidths=0.8,
        label="Published Atefeh layouts",
    )


def save_3d_plot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    published: pd.DataFrame | None = None,
) -> Path:
    fig = plt.figure(figsize=(12.5, 9.2))
    ax = fig.add_subplot(111, projection="3d")

    for rank in RANK_DRAW_ORDER:
        part = df[df["rank"] == rank]
        if part.empty:
            continue
        ax.scatter(
            part["pick_faces"],
            part["interior_storage"],
            part["retrieval_penalty"],
            marker="o",
            s=52,
            alpha=0.80,
            c=RANK_COLORS[rank],
            edgecolors="none",
            linewidths=0,
            depthshade=False,
            label=f"Rank {rank}",
            zorder=10 - rank,
        )

    if published is not None:
        ax.scatter(
            published["pick_faces"],
            published["interior_storage"],
            published["retrieval_penalty"],
            marker="x",
            s=45,
            c=PUBLISHED_COLOR,
            alpha=0.9,
            linewidths=0.8,
            depthshade=False,
            label="Published Atefeh layouts",
        )

    # ax.set_title(title, pad=18)
    ax.set_xlabel(r"Pick faces = $N_{pf}$", labelpad=12)
    ax.set_ylabel(r"Interior/deep storage = $N_{\mathrm{locked}}$", labelpad=12)
    # Matplotlib 3D z-axis labels are often clipped by tight_layout/bbox_inches.
    # Reserve a right margin and draw a figure-level z label instead.
    ax.set_zlabel("")
    ax.view_init(elev=22, azim=-52)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92)
    fig.subplots_adjust(left=0.02, right=0.78, top=0.92, bottom=0.08)
    fig.text(0.93, 0.50, r"Retrieval penalty = $R_p$", rotation=90, va="center", ha="center", fontsize=15)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return output_path


def save_rank_scatter_plot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    published: pd.DataFrame | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    scatter_ranks_2d(ax, df, x_col, y_col)

    if published is not None:
        scatter_published_2d(ax, published, x_col, y_col)

    # ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#cfcfcf", linewidth=0.75, alpha=0.65)

    x_published = published[x_col] if published is not None else None
    y_published = published[y_col] if published is not None else None
    xlim = padded_limits(df[x_col], x_published)
    ylim = padded_limits(df[y_col], y_published)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_storage_capacity_pareto_plot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    published: pd.DataFrame | None = None,
) -> tuple[Path, int]:
    """Save storage capacity vs pick faces with one generated style and frontier line."""
    frontier = compute_storage_pick_pareto_front(df)

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    ax.scatter(
        df["pick_faces"],
        df["storage_capacity"],
        marker="o",
        s=46,
        alpha=0.46,
        c=GENERATED_STORAGE_COLOR,
        edgecolors="none",
        linewidths=0,
        label="Generated layouts",
    )

    if not frontier.empty:
        ax.plot(
            frontier["pick_faces"],
            frontier["storage_capacity"],
            color=PARETO_LINE_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="Pareto front",
        )

    if published is not None:
        scatter_published_2d(ax, published, "pick_faces", "storage_capacity")

    # ax.set_title(title, pad=12)
    ax.set_xlabel(r"Pick faces = $N_{pf}$")
    ax.set_ylabel(r"Storage capacity = $SC(L)$")
    ax.grid(True, color="#cfcfcf", linewidth=0.75, alpha=0.65)

    x_published = published["pick_faces"] if published is not None else None
    y_published = published["storage_capacity"] if published is not None else None
    xlim = padded_limits(df["pick_faces"], x_published)
    ylim = padded_limits(df["storage_capacity"], y_published)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, len(frontier)


# -----------------------------------------------------------------------------
# Timing chart helpers
# -----------------------------------------------------------------------------
def timing_summary_from_rank_points(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return mean runtime by generation.

    If runtime_seconds is present in the rank CSV, it is used directly. The
    current 121 rank CSVs instead contain source_candidates_csv, so this function
    follows those paths to each sibling generation_summary.csv and uses its
    runtime_seconds column.
    """
    if "runtime_seconds" in df.columns:
        timing = df[["seed", "generation", "runtime_seconds"]].drop_duplicates()
        summary = timing.groupby("generation", as_index=False).agg(
            mean_runtime_seconds=("runtime_seconds", "mean")
        )
        return summary, "runtime_seconds column in rank CSV"

    if "source_candidates_csv" not in df.columns:
        return pd.DataFrame(columns=["generation", "mean_runtime_seconds"]), "no timing source available"

    frames = []
    for source in sorted(df["source_candidates_csv"].dropna().unique()):
        summary_path = Path(source).parent / "generation_summary.csv"
        if not summary_path.exists():
            continue
        generation_summary = pd.read_csv(summary_path)
        if {"generation", "runtime_seconds"}.issubset(generation_summary.columns):
            cols = ["generation", "runtime_seconds"]
            if "seed" in generation_summary.columns:
                cols.insert(0, "seed")
            frames.append(generation_summary[cols].copy())

    if not frames:
        return pd.DataFrame(columns=["generation", "mean_runtime_seconds"]), "no generation_summary.csv timing found"

    timing = pd.concat(frames, ignore_index=True).drop_duplicates()
    summary = timing.groupby("generation", as_index=False).agg(
        mean_runtime_seconds=("runtime_seconds", "mean")
    )
    return summary, "sibling generation_summary.csv files"


def save_timing_plot(df: pd.DataFrame, title: str, output_path: Path) -> tuple[Path, str]:
    summary, source = timing_summary_from_rank_points(df)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    if summary.empty:
        ax.text(0.5, 0.5, "Timing data unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(
            summary["generation"],
            summary["mean_runtime_seconds"],
            color="#4c78a8",
            edgecolor="#2f4f73",
            linewidth=0.6,
        )
        ax.grid(True, axis="y", color="#cfcfcf", linewidth=0.75, alpha=0.65)

    # ax.set_title(title, pad=12)
    ax.set_xlabel("Generation index")
    ax.set_ylabel("Mean computation time (s)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, source


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------
def main() -> None:
    atefeh_csv = resolve_input_path(ATEFEH_POINTS_CSV)
    kov_csv = resolve_input_path(KOV_POINTS_CSV)
    published_csv = resolve_input_path(ATEFEH_PUBLISHED_CSV)

    atefeh = load_rank_points(atefeh_csv)
    kov = load_rank_points(kov_csv)
    published = load_published_atefeh(published_csv)

    generated_files: list[Path] = []
    pareto_counts: dict[str, int] = {}
    timing_sources: dict[str, str] = {}

    generated_files.append(
        save_3d_plot(
            atefeh,
            "Atefeh: 3D Pareto objective space",
            OUTPUT_DIR / "Atefeh_3D_pareto_objective_space.png",
            published=published,
        )
    )
    generated_files.append(
        save_rank_scatter_plot(
            atefeh,
            "Atefeh: Retrieval penalty vs interior/deep storage",
            OUTPUT_DIR / "Atefeh_retrieval_penalty_vs_interior_deep_storage.png",
            "interior_storage",
            "retrieval_penalty",
            r"Interior/deep storage = $N_{\mathrm{locked}}$",
            r"Retrieval penalty = $R_p$",
            published=published,
        )
    )
    generated_files.append(
        save_rank_scatter_plot(
            atefeh,
            "Atefeh: Interior/deep storage vs pick faces",
            OUTPUT_DIR / "Atefeh_interior_deep_storage_vs_pick_faces.png",
            "pick_faces",
            "interior_storage",
            r"Pick faces = $N_{pf}$",
            r"Interior/deep storage = $N_{\mathrm{locked}}$",
            published=published,
        )
    )
    path, count = save_storage_capacity_pareto_plot(
        atefeh,
        "Atefeh: Storage capacity vs pick faces",
        OUTPUT_DIR / "20Figure_atefeh_storage_capacity_vs_pick_faces.png",
        published=published,
    )
    generated_files.append(path)
    pareto_counts["Atefeh"] = count
    path, source = save_timing_plot(
        atefeh,
        "Atefeh: Mean computation time by generation",
        OUTPUT_DIR / "Atefeh_mean_computation_time_by_generation.png",
    )
    generated_files.append(path)
    timing_sources["Atefeh"] = source

    generated_files.append(
        save_3d_plot(
            kov,
            "Kov-1-O-w4: 3D Pareto objective space",
            OUTPUT_DIR / "Kov1Ow4_3D_pareto_objective_space.png",
        )
    )
    generated_files.append(
        save_rank_scatter_plot(
            kov,
            "Kov-1-O-w4: Retrieval penalty vs interior/deep storage",
            OUTPUT_DIR / "Kov1Ow4_retrieval_penalty_vs_interior_deep_storage.png",
            "interior_storage",
            "retrieval_penalty",
            r"Interior/deep storage = $N_{\mathrm{locked}}$",
            r"Retrieval penalty = $R_p$",
        )
    )
    generated_files.append(
        save_rank_scatter_plot(
            kov,
            "Kov-1-O-w4: Interior/deep storage vs pick faces",
            OUTPUT_DIR / "Kov1Ow4_interior_deep_storage_vs_pick_faces.png",
            "pick_faces",
            "interior_storage",
            r"Pick faces = $N_{pf}$",
            r"Interior/deep storage = $N_{\mathrm{locked}}$",
        )
    )
    path, count = save_storage_capacity_pareto_plot(
        kov,
        "Kov-1-O-w4: Storage capacity vs pick faces",
        OUTPUT_DIR / "20Figure_kov1ow4_storage_capacity_vs_pick_faces.png",
    )
    generated_files.append(path)
    pareto_counts["Kov-1-O-w4"] = count
    path, source = save_timing_plot(
        kov,
        "Kov-1-O-w4: Mean computation time by generation",
        OUTPUT_DIR / "Kov1Ow4_mean_computation_time_by_generation.png",
    )
    generated_files.append(path)
    timing_sources["Kov-1-O-w4"] = source

    print("Pareto plot generation summary")
    print(f"Atefeh CSV: {atefeh_csv}")
    print(f"  rows read after rank 0-3 filter: {len(atefeh)}")
    print(f"Kov-1-O-w4 CSV: {kov_csv}")
    print(f"  rows read after rank 0-3 filter: {len(kov)}")
    print(f"Published Atefeh CSV: {published_csv}")
    print(f"  rows read: {len(published)}")
    print("Pareto-front point counts for storage-capacity plots:")
    for label, count in pareto_counts.items():
        print(f"  {label}: {count}")
    print("Timing sources:")
    for label, source in timing_sources.items():
        print(f"  {label}: {source}")
    print("Output files generated:")
    for output in generated_files:
        print(f"  {output}")


if __name__ == "__main__":
    main()


