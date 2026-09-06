"""Regenerate manuscript Figures 8 and 9 from the final-clean Phase-11 Proposed archive.

POST-PROCESSING ONLY. This script never runs an optimizer.

Fresh generated-layout source
-----------------------------
results/revision_final_30seed_nofg/p11/nsga2

For the two main-text instances:
- AT_S_comercial_layout_AW_3
- Gyorgy-KOVACS_WH_Narrow_AW_4

The script:
1. discovers seeds 101--130 and their final_ranked_layouts_index.json files;
2. pools final feasible Pareto-rank 0--3 archive entries;
3. deduplicates by exact-grid layout_signature;
4. assigns each signature its best (lowest) observed final within-seed Pareto rank;
5. validates the fresh unique-signature counts (Atefeh=62, Kov-1-O-w4=39);
6. writes fresh manifest CSVs;
7. re-scores the published Atefeh AT_1--AT_13 .npz masks with the repository's
   current `load_mask -> mask_to_grid -> score_layout` implementation;
8. writes a derived `atefeh_published_reference_metrics.csv` for traceability;
9. recomputes the published-reference dominance audit;
10. regenerates the Figure 8/9 objective-space panels and runtime panels;
11. writes a machine-readable JSON audit and a concise Markdown audit.

Important provenance rules
--------------------------
- The old logged rank-point CSVs are not inputs.
- The old pre-scored Atefeh reference CSV is not an input.
- Published Atefeh AT_1--AT_13 masks are re-scored from `data/instances/masks`.
- Published Atefeh layouts are not included in optimization indicators/statistics.
- Pareto dominance is evaluated on the three structural objectives:
      minimize Nlocked, maximize Npf, minimize Rp.
- Storage capacity SC is a derived descriptor only. Its SC-vs-Npf curve is labelled
  "Non-dominated envelope", not "Pareto front".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whl_core.layout_io import load_mask, mask_to_grid
from whl_core.scoring import score_layout


SEEDS = tuple(range(101, 131))
RANKS = (0, 1, 2, 3)
DRAW_RANKS = (3, 2, 1, 0)

ATEFEH_INSTANCE = "AT_S_comercial_layout_AW_3"
KOV_INSTANCE = "Gyorgy-KOVACS_WH_Narrow_AW_4"

INSTANCE_SPECS = {
    ATEFEH_INSTANCE: {
        "short": "Atefeh",
        "expected_unique": 62,
        "manifest_name": "atefeh_unique_rank03_manifest.csv",
    },
    KOV_INSTANCE: {
        "short": "Kov-1-O-w4",
        "expected_unique": 39,
        "manifest_name": "kov1ow4_unique_rank03_manifest.csv",
    },
}

RANK_COLORS = {
    0: "red",
    1: "#32a852",
    2: "#4663c2",
    3: "#d179e0",
}
RANK_ZORDERS = {3: 2, 2: 3, 1: 4, 0: 5}

PICK_LABEL = r"Pick faces $N_{pf}$"
RP_LABEL = r"Retrieval penalty $R_p$"
DEEP_LABEL = r"Interior/deep storage $N_{\mathrm{locked}}$"
CAPACITY_LABEL = r"Storage capacity $SC$"

PUBLISHED_LABEL = "Digitized published layouts"
GENERATED_STORAGE_COLOR = "#4f86c6"
ENVELOPE_LINE_COLOR = "#c55dd8"
PUBLISHED_COLOR = "black"

PLOT_FILES = {
    ATEFEH_INSTANCE: {
        "3d": "Atefeh_3D_pareto_objective_space.png",
        "rp_locked": "Atefeh_retrieval_penalty_vs_interior_deep_storage.png",
        "pf_locked": "Atefeh_interior_deep_storage_vs_pick_faces.png",
        "capacity": "20Figure_atefeh_storage_capacity_vs_pick_faces.png",
        "runtime": "Atefeh_mean_computation_time_by_generation.png",
    },
    KOV_INSTANCE: {
        "3d": "Kov1Ow4_3D_pareto_objective_space.png",
        "rp_locked": "Kov1Ow4_retrieval_penalty_vs_interior_deep_storage.png",
        "pf_locked": "Kov1Ow4_interior_deep_storage_vs_pick_faces.png",
        "capacity": "20Figure_kov1ow4_storage_capacity_vs_pick_faces.png",
        "runtime": "Kov1Ow4_mean_computation_time_by_generation.png",
    },
}

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


def _as_int(value: Any, label: str) -> int:
    if value in (None, ""):
        raise ValueError(f"missing integer value for {label}")
    return int(float(value))


def _as_float(value: Any, label: str) -> float:
    if value in (None, ""):
        raise ValueError(f"missing float value for {label}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return x


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _discover_final_indexes(
    p11_nsga2_root: Path,
    instance: str,
    seeds: Sequence[int] = SEEDS,
) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(p11_nsga2_root.rglob("final_ranked_layouts_index.json")):
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
                f"duplicate final-ranked index for {instance}, seed {seed}: "
                f"{mapping[seed]} and {path}"
            )
        mapping[seed] = path

    expected = set(int(seed) for seed in seeds)
    found = set(mapping)
    if found != expected:
        raise ValueError(
            f"final-index coverage mismatch for {instance}; "
            f"missing={sorted(expected-found)}, extra={sorted(found-expected)}"
        )
    return mapping


def _entry_to_row(
    entry: dict[str, Any],
    *,
    instance: str,
    seed: int,
    index_path: Path,
) -> dict[str, Any]:
    rank = _as_int(entry.get("rank"), "rank")
    if rank not in RANKS:
        raise ValueError(f"unexpected rank in final rank0-3 index: {rank}")

    signature = str(entry.get("layout_signature", "")).strip()
    if not signature:
        raise ValueError(f"missing layout_signature in {index_path}")

    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"missing metrics object in {index_path}")

    nlocked = _as_int(metrics.get("interior_storage"), "metrics.interior_storage")
    npf = _as_int(metrics.get("pick_faces"), "metrics.pick_faces")
    rp = _as_float(metrics.get("retrieval_penalty"), "metrics.retrieval_penalty")
    sc = _as_int(metrics.get("storage_total"), "metrics.storage_total")

    if sc != nlocked + npf:
        raise ValueError(
            f"capacity identity failed for {instance}/seed_{seed}/{signature}: "
            f"SC={sc}, Nlocked={nlocked}, Npf={npf}"
        )

    npz_file = str(entry.get("npz_file", ""))
    npz_path = index_path.parent / npz_file if npz_file else None

    return {
        "instance": instance,
        "seed": seed,
        "rank": rank,
        "layout_signature": signature,
        "storage_capacity": sc,
        "pick_faces": npf,
        "interior_storage": nlocked,
        "retrieval_penalty": rp,
        "door_connectivity_index": metrics.get("door_connectivity_index", ""),
        "candidate_id": entry.get("candidate_id", ""),
        "archive_key": entry.get("archive_key", entry.get("layout_key", "")),
        "source_generation": entry.get("source_generation", entry.get("generation", "")),
        "source": entry.get("source", ""),
        "archive_index_path": str(index_path),
        "archive_npz_path": str(npz_path) if npz_path is not None else "",
    }


def _pool_unique_signatures(
    p11_nsga2_root: Path,
    instance: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexes = _discover_final_indexes(p11_nsga2_root, instance)
    archive_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        index_path = indexes[seed]
        payload = json.loads(index_path.read_text(encoding="utf-8"))

        # Current experiment archives store final_ranked_layouts_index.json as a
        # top-level JSON list of entry dictionaries. Keep compatibility with an
        # older wrapped {"seed": ..., "entries": [...]} shape as well.
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            if int(payload.get("seed", seed)) != seed:
                raise ValueError(f"seed metadata mismatch in {index_path}")
            entries = payload.get("entries")
        else:
            raise ValueError(
                f"unsupported final-ranked index JSON type in {index_path}: "
                f"{type(payload).__name__}"
            )

        if not isinstance(entries, list):
            raise ValueError(f"missing entries list in {index_path}")

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"non-object final-ranked entry in {index_path}: "
                    f"{type(entry).__name__}"
                )

            entry_seed = entry.get("seed")
            if entry_seed not in (None, "") and int(entry_seed) != seed:
                raise ValueError(
                    f"entry seed mismatch in {index_path}: "
                    f"expected {seed}, found {entry_seed}"
                )

            entry_instance = str(entry.get("instance", instance))
            if entry_instance != instance:
                raise ValueError(
                    f"entry instance mismatch in {index_path}: "
                    f"expected {instance}, found {entry_instance}"
                )

            archive_rows.append(
                _entry_to_row(
                    entry,
                    instance=instance,
                    seed=seed,
                    index_path=index_path,
                )
            )

    if not archive_rows:
        raise ValueError(f"no final archive rows found for {instance}")

    raw = pd.DataFrame(archive_rows)
    expected_entries = {
        ATEFEH_INSTANCE: 709,
        KOV_INSTANCE: 655,
    }.get(instance)
    if expected_entries is not None and len(raw) != expected_entries:
        raise ValueError(
            f"fresh archive-entry count mismatch for {instance}: "
            f"expected {expected_entries}, got {len(raw)}"
        )

    consistency_cols = [
        "storage_capacity",
        "pick_faces",
        "interior_storage",
        "retrieval_penalty",
    ]
    inconsistent: list[str] = []
    for signature, group in raw.groupby("layout_signature", sort=False):
        tuples = set(
            tuple(row)
            for row in group[consistency_cols].itertuples(index=False, name=None)
        )
        if len(tuples) != 1:
            inconsistent.append(signature)
    if inconsistent:
        raise ValueError(
            "same layout_signature has inconsistent structural metrics; "
            f"first signatures={inconsistent[:10]}"
        )

    # One representative per exact-grid signature. Lowest final within-seed rank wins.
    # Remaining tie-breakers are deterministic and do not change its objective tuple.
    ordered = raw.sort_values(
        [
            "rank",
            "retrieval_penalty",
            "pick_faces",
            "storage_capacity",
            "seed",
            "candidate_id",
        ],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    )
    unique = ordered.drop_duplicates("layout_signature", keep="first").copy()

    occurrence = raw.groupby("layout_signature").agg(
        occurrence_count=("layout_signature", "size"),
        seed_count=("seed", "nunique"),
        first_seed=("seed", "min"),
        last_seed=("seed", "max"),
        best_observed_rank=("rank", "min"),
    )
    unique = unique.merge(
        occurrence,
        left_on="layout_signature",
        right_index=True,
        how="left",
    )
    unique["rank"] = unique["best_observed_rank"].astype(int)
    unique = unique.sort_values(
        ["rank", "retrieval_penalty", "pick_faces", "storage_capacity", "layout_signature"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    expected_unique = int(INSTANCE_SPECS[instance]["expected_unique"])
    if len(unique) != expected_unique:
        raise ValueError(
            f"fresh unique-signature count mismatch for {instance}: "
            f"expected {expected_unique}, got {len(unique)}"
        )

    rank_counts = {
        str(int(k)): int(v)
        for k, v in unique["rank"].value_counts().sort_index().items()
    }
    info = {
        "instance": instance,
        "seed_coverage": [min(SEEDS), max(SEEDS)],
        "seed_count": len(SEEDS),
        "archive_entry_count": int(len(raw)),
        "unique_signature_count": int(len(unique)),
        "rank_counts": rank_counts,
        "source_index_count": len(indexes),
    }

    keep_cols = [
        "instance",
        "rank",
        "layout_signature",
        "storage_capacity",
        "pick_faces",
        "interior_storage",
        "retrieval_penalty",
        "door_connectivity_index",
        "occurrence_count",
        "seed_count",
        "first_seed",
        "last_seed",
        "candidate_id",
        "archive_key",
        "source_generation",
        "source",
        "archive_index_path",
        "archive_npz_path",
    ]
    return unique[keep_cols].copy(), info


def _score_published_reference_masks(
    mask_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score the published Atefeh AT_1--AT_13 masks with current repository metrics."""
    if not mask_dir.exists():
        raise FileNotFoundError(f"published Atefeh mask directory not found: {mask_dir}")

    rows: list[dict[str, Any]] = []
    mask_hashes: dict[str, str] = {}

    for index in range(1, 14):
        layout_id = f"AT_{index}"
        mask_path = mask_dir / f"{layout_id}.npz"
        if not mask_path.exists():
            raise FileNotFoundError(f"published Atefeh mask not found: {mask_path}")

        masks = load_mask(mask_path)
        grid = mask_to_grid(masks)
        metrics = score_layout(grid)

        storage_capacity = _as_int(metrics.get("storage_total"), "storage_total")
        pick_faces = _as_int(metrics.get("pick_faces"), "pick_faces")
        interior_storage = _as_int(metrics.get("interior_storage"), "interior_storage")
        retrieval_penalty = _as_float(
            metrics.get("retrieval_penalty"),
            "retrieval_penalty",
        )

        if storage_capacity != pick_faces + interior_storage:
            raise ValueError(
                f"published-reference capacity identity failed for {layout_id}: "
                f"SC={storage_capacity}, Npf={pick_faces}, Nlocked={interior_storage}"
            )

        digest = _sha256(mask_path)
        mask_hashes[layout_id] = digest
        rows.append(
            {
                "layout_id": layout_id,
                "storage_capacity": storage_capacity,
                "pick_faces": pick_faces,
                "interior_storage": interior_storage,
                "retrieval_penalty": retrieval_penalty,
                "mask_path": str(mask_path),
                "mask_sha256": digest,
            }
        )

    df = pd.DataFrame(rows)
    info = {
        "source": "AT_1--AT_13 .npz masks re-scored with current repository code",
        "mask_dir": str(mask_dir),
        "row_count": int(len(df)),
        "mask_sha256": mask_hashes,
        "scoring_status": (
            "recomputed from published-reference masks via "
            "load_mask -> mask_to_grid -> score_layout"
        ),
    }
    return df, info


def _objective_vector(row: pd.Series | dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["interior_storage"]),
            -float(row["pick_faces"]),
            float(row["retrieval_penalty"]),
        ],
        dtype=float,
    )


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _published_dominance_audit(
    generated: pd.DataFrame,
    published: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rank0 = generated[generated["rank"] == 0].copy()
    g_vecs = np.vstack([_objective_vector(row) for _, row in rank0.iterrows()])
    p_vecs = np.vstack([_objective_vector(row) for _, row in published.iterrows()])

    published_dominated = []
    for p_idx, p_vec in enumerate(p_vecs):
        dominators = [
            rank0.iloc[g_idx]["layout_signature"]
            for g_idx, g_vec in enumerate(g_vecs)
            if _dominates(g_vec, p_vec)
        ]
        published_dominated.append((p_idx, dominators))

    generated_dominated = []
    for g_idx, g_vec in enumerate(g_vecs):
        dominators = [
            published.iloc[p_idx]["layout_id"]
            for p_idx, p_vec in enumerate(p_vecs)
            if _dominates(p_vec, g_vec)
        ]
        generated_dominated.append((g_idx, dominators))

    union_records: list[dict[str, Any]] = []
    for _, row in rank0.iterrows():
        union_records.append(
            {
                "source": "generated_rank0",
                "design_id": row["layout_signature"],
                "interior_storage": float(row["interior_storage"]),
                "pick_faces": float(row["pick_faces"]),
                "retrieval_penalty": float(row["retrieval_penalty"]),
            }
        )
    for _, row in published.iterrows():
        union_records.append(
            {
                "source": "published",
                "design_id": row["layout_id"],
                "interior_storage": float(row["interior_storage"]),
                "pick_faces": float(row["pick_faces"]),
                "retrieval_penalty": float(row["retrieval_penalty"]),
            }
        )

    union_df = pd.DataFrame(union_records)
    union_vecs = np.vstack([_objective_vector(row) for _, row in union_df.iterrows()])
    nondominated_mask = []
    for i, vec in enumerate(union_vecs):
        dominated = any(
            j != i and _dominates(other, vec)
            for j, other in enumerate(union_vecs)
        )
        nondominated_mask.append(not dominated)
    union_df["nondominated_in_pooled_union"] = nondominated_mask
    nd = union_df[union_df["nondominated_in_pooled_union"]].copy()
    tuple_count = (
        nd[["interior_storage", "pick_faces", "retrieval_penalty"]]
        .drop_duplicates()
        .shape[0]
    )

    published_dom_count = sum(bool(doms) for _, doms in published_dominated)
    generated_dom_count = sum(bool(doms) for _, doms in generated_dominated)

    audit = {
        "generated_rank0_signature_count": int(len(rank0)),
        "published_layout_count": int(len(published)),
        "published_dominated_by_generated_rank0_count": int(published_dom_count),
        "published_dominated_by_generated_rank0_fraction": (
            float(published_dom_count / len(published)) if len(published) else None
        ),
        "generated_rank0_dominated_by_published_count": int(generated_dom_count),
        "generated_rank0_dominated_by_published_fraction": (
            float(generated_dom_count / len(rank0)) if len(rank0) else None
        ),
        "pooled_union_design_count": int(len(union_df)),
        "pooled_nondominated_design_count": int(len(nd)),
        "pooled_nondominated_unique_objective_tuple_count": int(tuple_count),
        "pooled_nondominated_generated_count": int((nd["source"] == "generated_rank0").sum()),
        "pooled_nondominated_published_count": int((nd["source"] == "published").sum()),
    }

    # Per-design dominance details make reviewer-response numbers auditable.
    detail_rows: list[dict[str, Any]] = []
    for p_idx, dominators in published_dominated:
        row = published.iloc[p_idx]
        detail_rows.append(
            {
                "source": "published",
                "design_id": row["layout_id"],
                "dominated_by_other_source": bool(dominators),
                "dominator_count": len(dominators),
                "dominators": ";".join(map(str, dominators)),
            }
        )
    for g_idx, dominators in generated_dominated:
        row = rank0.iloc[g_idx]
        detail_rows.append(
            {
                "source": "generated_rank0",
                "design_id": row["layout_signature"],
                "dominated_by_other_source": bool(dominators),
                "dominator_count": len(dominators),
                "dominators": ";".join(map(str, dominators)),
            }
        )
    return audit, pd.DataFrame(detail_rows), union_df


def _ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for col, name in [
        ("storage_capacity", "SC"),
        ("pick_faces", "Npf"),
        ("interior_storage", "Nlocked"),
        ("retrieval_penalty", "Rp"),
    ]:
        series = pd.to_numeric(df[col], errors="raise").astype(float)
        result[name] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return result


def _compute_storage_pick_envelope(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated envelope for the derived SC-vs-Npf diagnostic (maximize both)."""
    points = (
        df[["pick_faces", "storage_capacity"]]
        .dropna()
        .drop_duplicates()
        .groupby("pick_faces", as_index=False)["storage_capacity"]
        .max()
        .sort_values("pick_faces", ascending=False)
    )
    keep: list[pd.Series] = []
    best_capacity = -math.inf
    for _, row in points.iterrows():
        capacity = float(row["storage_capacity"])
        if capacity > best_capacity:
            keep.append(row)
            best_capacity = capacity
    if not keep:
        return pd.DataFrame(columns=["pick_faces", "storage_capacity"])
    return (
        pd.DataFrame(keep)
        .sort_values("pick_faces")
        .reset_index(drop=True)
    )


def _padded_limits(*series: pd.Series, pad: float = 0.05) -> tuple[float, float] | None:
    chunks: list[np.ndarray] = []
    for item in series:
        if item is None:
            continue
        arr = np.asarray(item, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return None
    merged = np.concatenate(chunks)
    low = float(np.min(merged))
    high = float(np.max(merged))
    if math.isclose(low, high):
        delta = max(1.0, abs(high) * pad)
    else:
        delta = (high - low) * pad
    return low - delta, high + delta


def _ordered_legend(ax: plt.Axes, *, include_generated: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered: list[str] = []
    for rank in RANKS:
        label = f"Pareto rank {rank}"
        if label in by_label:
            ordered.append(label)
    if include_generated and "Generated layouts" in by_label:
        ordered.append("Generated layouts")
    if "Non-dominated envelope" in by_label:
        ordered.append("Non-dominated envelope")
    if PUBLISHED_LABEL in by_label:
        ordered.append(PUBLISHED_LABEL)
    ordered.extend(label for label in labels if label not in ordered)
    ax.legend(
        [by_label[label] for label in ordered],
        ordered,
        loc="best",
        frameon=True,
        framealpha=0.92,
    )


def _scatter_ranks_2d(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    for rank in DRAW_RANKS:
        part = df[df["rank"] == rank]
        if part.empty:
            continue
        ax.scatter(
            part[x_col],
            part[y_col],
            marker="o",
            s=50,
            alpha=0.78,
            c=RANK_COLORS[rank],
            edgecolors="none",
            linewidths=0,
            label=f"Pareto rank {rank}",
            zorder=RANK_ZORDERS[rank],
        )


def _scatter_published_2d(
    ax: plt.Axes,
    published: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    ax.scatter(
        published[x_col],
        published[y_col],
        marker="x",
        s=45,
        c=PUBLISHED_COLOR,
        alpha=0.9,
        linewidths=1.1,
        label=PUBLISHED_LABEL,
        zorder=8,
    )


def _save_3d(
    df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    published: pd.DataFrame | None,
) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for rank in DRAW_RANKS:
        part = df[df["rank"] == rank]
        if part.empty:
            continue
        ax.scatter(
            part["pick_faces"],
            part["interior_storage"],
            part["retrieval_penalty"],
            marker="o",
            s=50,
            alpha=0.78,
            c=RANK_COLORS[rank],
            edgecolors="none",
            linewidths=0,
            depthshade=False,
            label=f"Pareto rank {rank}",
            zorder=RANK_ZORDERS[rank],
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
            linewidths=1.1,
            depthshade=False,
            label=PUBLISHED_LABEL,
            zorder=8,
        )
    ax.set_title(title, pad=18)
    ax.set_xlabel(PICK_LABEL, labelpad=12)
    ax.set_ylabel(DEEP_LABEL, labelpad=12)
    ax.set_zlabel("")
    ax.view_init(elev=22, azim=-52)
    _ordered_legend(ax)
    fig.subplots_adjust(left=0.02, right=0.78, top=0.92, bottom=0.08)
    fig.text(0.93, 0.50, RP_LABEL, rotation=90, va="center", ha="center", fontsize=15)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)


def _save_rank_scatter(
    df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    published: pd.DataFrame | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    _scatter_ranks_2d(ax, df, x_col, y_col)
    if published is not None:
        _scatter_published_2d(ax, published, x_col, y_col)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#cfcfcf", linewidth=0.75, alpha=0.65)
    x_ref = published[x_col] if published is not None else None
    y_ref = published[y_col] if published is not None else None
    xlim = _padded_limits(df[x_col], x_ref)
    ylim = _padded_limits(df[y_col], y_ref)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _ordered_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_capacity_plot(
    df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    published: pd.DataFrame | None,
) -> int:
    points = df[["pick_faces", "storage_capacity"]].dropna().copy()
    envelope = _compute_storage_pick_envelope(points)

    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    ax.scatter(
        points["pick_faces"],
        points["storage_capacity"],
        marker="o",
        s=44,
        alpha=0.62,
        c=GENERATED_STORAGE_COLOR,
        edgecolors="none",
        linewidths=0,
        label="Generated layouts",
        zorder=3,
    )
    if not envelope.empty:
        ax.plot(
            envelope["pick_faces"],
            envelope["storage_capacity"],
            color=ENVELOPE_LINE_COLOR,
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=4.2,
            label="Non-dominated envelope",
            zorder=4,
        )
    if published is not None:
        _scatter_published_2d(ax, published, "pick_faces", "storage_capacity")

    ax.set_title(title, pad=12)
    ax.set_xlabel(PICK_LABEL)
    ax.set_ylabel(CAPACITY_LABEL)
    ax.grid(True, color="#cfcfcf", linestyle=":", linewidth=0.85, alpha=0.80)
    x_ref = published["pick_faces"] if published is not None else None
    y_ref = published["storage_capacity"] if published is not None else None
    xlim = _padded_limits(points["pick_faces"], x_ref)
    ylim = _padded_limits(points["storage_capacity"], y_ref)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _ordered_legend(ax, include_generated=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return int(len(envelope))


def _discover_generation_summaries(
    p11_nsga2_root: Path,
    instance: str,
) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(p11_nsga2_root.rglob("generation_summary.csv")):
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
        if seed not in SEEDS:
            continue
        if seed in mapping:
            raise ValueError(
                f"duplicate generation_summary.csv for {instance}/seed_{seed}"
            )
        mapping[seed] = path

    expected = set(SEEDS)
    if set(mapping) != expected:
        raise ValueError(
            f"generation-summary coverage mismatch for {instance}; "
            f"missing={sorted(expected-set(mapping))}, "
            f"extra={sorted(set(mapping)-expected)}"
        )
    return mapping


def _timing_summary(
    p11_nsga2_root: Path,
    instance: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = _discover_generation_summaries(p11_nsga2_root, instance)
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        path = files[seed]
        raw = pd.read_csv(path)
        required = {"generation", "runtime_seconds"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path} missing timing columns: {missing}")
        frame = raw[["generation", "runtime_seconds"]].copy()
        frame["generation"] = pd.to_numeric(frame["generation"], errors="raise").astype(int)
        frame["runtime_seconds"] = pd.to_numeric(
            frame["runtime_seconds"], errors="raise"
        ).astype(float)
        frame["seed"] = seed
        frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    summary = (
        all_rows.groupby("generation", as_index=False)
        .agg(
            n=("runtime_seconds", "size"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
        )
        .sort_values("generation")
        .reset_index(drop=True)
    )
    info = {
        "source_file_count": len(files),
        "seed_count": len(SEEDS),
        "generation_count": int(len(summary)),
        "generation_min": int(summary["generation"].min()),
        "generation_max": int(summary["generation"].max()),
        "all_generations_have_30_seeds": bool((summary["n"] == len(SEEDS)).all()),
    }
    if not info["all_generations_have_30_seeds"]:
        bad = summary.loc[summary["n"] != len(SEEDS), ["generation", "n"]]
        raise ValueError(
            "timing generations do not all contain 30 seeds: "
            + bad.to_dict(orient="records").__repr__()
        )
    return summary, info


def _save_timing_plot(
    summary: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.bar(
        summary["generation"],
        summary["mean_runtime_seconds"],
        color="#4c78a8",
        edgecolor="#2f4f73",
        linewidth=0.6,
    )
    ax.grid(True, axis="y", color="#cfcfcf", linewidth=0.75, alpha=0.65)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean computation time (s)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_audit(path: Path, audit: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Figure 8/9 Fresh Pareto Plot Audit",
        "",
        "## Provenance",
        f"- Results root: `{audit['results_root']}`",
        f"- Phase-11 Proposed root: `{audit['p11_nsga2_root']}`",
        "- Seeds: 101–130 (30 seeds).",
        "- Archive scope: final feasible Pareto ranks 0–3.",
        "- Deduplication: exact-grid `layout_signature` across seeds; best observed final within-seed rank retained.",
        "- Old logged rank-point CSVs are not used.",
        "",
        "## Fresh unique-layout manifests",
    ]
    for key in ("Atefeh", "Kov-1-O-w4"):
        item = audit["instances"][key]
        lines.append(
            f"- {key}: {item['archive_entry_count']} archive entries → "
            f"**{item['unique_signature_count']} unique signatures**; "
            f"rank counts {item['rank_counts']}."
        )
        r = item["ranges"]
        lines.append(
            f"  Ranges: SC {r['SC']['min']:g}–{r['SC']['max']:g}; "
            f"Npf {r['Npf']['min']:g}–{r['Npf']['max']:g}; "
            f"Nlocked {r['Nlocked']['min']:g}–{r['Nlocked']['max']:g}; "
            f"Rp {r['Rp']['min']:g}–{r['Rp']['max']:g}."
        )
        lines.append(
            f"  Derived SC–Npf non-dominated envelope points: "
            f"{item['sc_npf_envelope_point_count']}."
        )
        lines.append(
            f"  Timing: {item['timing']['source_file_count']} generation-summary files; "
            f"generations {item['timing']['generation_min']}–"
            f"{item['timing']['generation_max']}; 30 seeds per generation = "
            f"{item['timing']['all_generations_have_30_seeds']}."
        )

    pub = audit["published_reference"]
    pr = pub["ranges"]
    lines.extend(
        [
            "",
            "## Atefeh published reference",
            f"- Published layouts re-scored from AT_1--AT_13 masks: {pub['row_count']}.",
            f"- Status: {pub['scoring_status']}.",
            f"- Derived metrics CSV: `{pub['derived_metrics_csv']}`.",
            f"- Ranges: SC {pr['SC']['min']:g}–{pr['SC']['max']:g}; "
            f"Npf {pr['Npf']['min']:g}–{pr['Npf']['max']:g}; "
            f"Nlocked {pr['Nlocked']['min']:g}–{pr['Nlocked']['max']:g}; "
            f"Rp {pr['Rp']['min']:g}–{pr['Rp']['max']:g}.",
            "",
            "## Quantitative generated-vs-published dominance",
        ]
    )
    d = audit["atefeh_published_dominance"]
    lines.extend(
        [
            f"- Generated Pareto-rank 0 signatures: {d['generated_rank0_signature_count']}.",
            f"- Published layouts dominated by ≥1 generated rank-0 signature: "
            f"**{d['published_dominated_by_generated_rank0_count']}/"
            f"{d['published_layout_count']} "
            f"({100*d['published_dominated_by_generated_rank0_fraction']:.1f}%)**.",
            f"- Generated rank-0 signatures dominated by ≥1 published layout: "
            f"**{d['generated_rank0_dominated_by_published_count']}/"
            f"{d['generated_rank0_signature_count']} "
            f"({100*d['generated_rank0_dominated_by_published_fraction']:.1f}%)**.",
            f"- Pooled nondominated union: **{d['pooled_nondominated_design_count']} designs**, "
            f"representing **{d['pooled_nondominated_unique_objective_tuple_count']} "
            f"distinct objective tuples**.",
            f"- Pooled nondominated source split: generated "
            f"{d['pooled_nondominated_generated_count']}, published "
            f"{d['pooled_nondominated_published_count']}.",
            "",
            "## Figure terminology",
            "- Legends use `Pareto rank 0` … `Pareto rank 3`.",
            "- Published Atefeh overlay: `Digitized published layouts`.",
            "- SC–Npf line: `Non-dominated envelope` (derived descriptor space, not an optimized Pareto front).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_and_plot(
    *,
    results_root: Path,
    published_mask_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    results_root = results_root.resolve()
    p11_nsga2_root = results_root / "p11" / "nsga2"
    output_dir = output_dir.resolve()
    published_mask_dir = published_mask_dir.resolve()

    if not p11_nsga2_root.exists():
        raise FileNotFoundError(f"Phase-11 Proposed root not found: {p11_nsga2_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    published, published_info = _score_published_reference_masks(published_mask_dir)
    published_metrics_csv = output_dir / "atefeh_published_reference_metrics.csv"
    published.to_csv(published_metrics_csv, index=False)
    published_info["derived_metrics_csv"] = str(published_metrics_csv)
    published_info["derived_metrics_csv_sha256"] = _sha256(published_metrics_csv)

    generated_by_instance: dict[str, pd.DataFrame] = {}
    instance_audit: dict[str, dict[str, Any]] = {}

    for instance, spec in INSTANCE_SPECS.items():
        generated, info = _pool_unique_signatures(p11_nsga2_root, instance)
        manifest_path = output_dir / str(spec["manifest_name"])
        generated.to_csv(manifest_path, index=False)

        timing, timing_info = _timing_summary(p11_nsga2_root, instance)
        timing_csv = output_dir / f"{spec['short'].replace('-', '').replace(' ', '_')}_generation_timing_summary.csv"
        timing.to_csv(timing_csv, index=False)

        generated_by_instance[instance] = generated
        instance_audit[str(spec["short"])] = {
            **info,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "ranges": _ranges(generated),
            "timing": {
                **timing_info,
                "summary_csv": str(timing_csv),
            },
        }

    dominance, dominance_detail, pooled_union = _published_dominance_audit(
        generated_by_instance[ATEFEH_INSTANCE],
        published,
    )
    dominance_detail_path = output_dir / "atefeh_published_dominance_detail.csv"
    dominance_detail.to_csv(dominance_detail_path, index=False)
    pooled_union_path = output_dir / "atefeh_generated_published_pooled_union.csv"
    pooled_union.to_csv(pooled_union_path, index=False)

    # Generate objective-space and derived-capacity panels.
    for instance, spec in INSTANCE_SPECS.items():
        df = generated_by_instance[instance]
        short = str(spec["short"])
        files = PLOT_FILES[instance]
        ref = published if instance == ATEFEH_INSTANCE else None

        _save_3d(
            df,
            title=f"{short}: 3D objective-space view",
            output_path=output_dir / files["3d"],
            published=ref,
        )
        _save_rank_scatter(
            df,
            title=f"{short}: Retrieval penalty vs interior/deep storage",
            output_path=output_dir / files["rp_locked"],
            x_col="interior_storage",
            y_col="retrieval_penalty",
            xlabel=DEEP_LABEL,
            ylabel=RP_LABEL,
            published=ref,
        )
        _save_rank_scatter(
            df,
            title=f"{short}: Interior/deep storage vs pick faces",
            output_path=output_dir / files["pf_locked"],
            x_col="pick_faces",
            y_col="interior_storage",
            xlabel=PICK_LABEL,
            ylabel=DEEP_LABEL,
            published=ref,
        )
        envelope_count = _save_capacity_plot(
            df,
            title=f"{short}: Storage capacity vs pick faces",
            output_path=output_dir / files["capacity"],
            published=ref,
        )
        instance_audit[short]["sc_npf_envelope_point_count"] = envelope_count

        timing_csv = Path(instance_audit[short]["timing"]["summary_csv"])
        timing_df = pd.read_csv(timing_csv)
        _save_timing_plot(
            timing_df,
            title=f"{short}: Mean computation time by generation",
            output_path=output_dir / files["runtime"],
        )

    published_info["ranges"] = _ranges(published)
    audit = {
        "post_processing_only": True,
        "results_root": str(results_root),
        "p11_nsga2_root": str(p11_nsga2_root),
        "seed_range": [min(SEEDS), max(SEEDS)],
        "seed_count": len(SEEDS),
        "archive_scope": "final feasible Pareto ranks 0-3",
        "deduplication_rule": (
            "exact-grid layout_signature pooled across seeds; "
            "lowest/best observed final within-seed Pareto rank retained"
        ),
        "instances": instance_audit,
        "published_reference": published_info,
        "atefeh_published_dominance": dominance,
        "dominance_detail_csv": str(dominance_detail_path),
        "pooled_union_csv": str(pooled_union_path),
        "figure_output_dir": str(output_dir),
        "old_logged_rank_point_csvs_used": False,
        "old_prescored_published_reference_csv_used": False,
    }

    json_path = output_dir / "figure8_9_audit.json"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "figure8_9_audit.md"
    _write_markdown_audit(markdown_path, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate fresh IJPR Figures 8/9 from final-clean Phase-11 Proposed "
            "rank-0--3 archives."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Final-clean campaign root, e.g. results/revision_final_30seed_nofg",
    )
    parser.add_argument(
        "--published-mask-dir",
        type=Path,
        required=True,
        help="Directory containing published Atefeh masks AT_1.npz through AT_13.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fresh manifests, audit evidence, and Figure 8/9 panels",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analyze_and_plot(
        results_root=args.results_root,
        published_mask_dir=args.published_mask_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())