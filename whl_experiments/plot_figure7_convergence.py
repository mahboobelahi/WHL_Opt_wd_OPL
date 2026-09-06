from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INSTANCE_ORDER = ["Atefeh", "Kov-1-O-w4"]

PHASE_SPECS = [
    {
        "phase": "phase11",
        "filename": "Figure7a_phase11",
        "method_order": ["bs_only_direct", "proposed_nsga2_bs", "random_restart_bs"],
        "labels": {
            "bs_only_direct": "BS-only",
            "proposed_nsga2_bs": "Prop.",
            "random_restart_bs": "RRBS",
        },
        "styles": {
            "bs_only_direct": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
            "proposed_nsga2_bs": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
            "random_restart_bs": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
        },
        "legend_ncol": 3,
    },
    {
        "phase": "phase12c",
        "filename": "Figure7b_phase12c",
        "method_order": ["V0_full_proposed", "V6_depth15_beam_default", "V7_beam_plus1_depth_default"],
        "labels": {
            "V0_full_proposed": "Prop. (Reference)",
            "V6_depth15_beam_default": "Depth 15",
            "V7_beam_plus1_depth_default": "Beam +1",
        },
        "styles": {
            "V0_full_proposed": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
            "V6_depth15_beam_default": {"marker": "s", "linestyle": "--", "linewidth": 2.0, "markersize": 5},
            "V7_beam_plus1_depth_default": {"marker": "^", "linestyle": "-.", "linewidth": 2.0, "markersize": 6},
        },
        "legend_ncol": 3,
    },
    {
        "phase": "phase12b",
        "filename": "Figure7c_phase12b",
        "method_order": [
            "V0_full_proposed",
            "V1_fixed_sorting",
            "V2_fixed_weights",
            "V3_uniform_mutation",
            "V4_no_symmetry_breaking",
            "V5_random_feasible_start_spacing",
        ],
        "labels": {
            "V0_full_proposed": "Prop. (Reference)",
            "V1_fixed_sorting": "Fixed sorting rules",
            "V2_fixed_weights": "Fixed scalar weights",
            "V3_uniform_mutation": "Uniform mutation",
            "V4_no_symmetry_breaking": "No symmetry breaking",
            "V5_random_feasible_start_spacing": "Random feasible start",
        },
        "styles": {
            "V0_full_proposed": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
            "V1_fixed_sorting": {"marker": "s", "linestyle": "--", "linewidth": 2.0, "markersize": 5},
            "V2_fixed_weights": {"marker": "^", "linestyle": "-.", "linewidth": 2.0, "markersize": 5},
            "V3_uniform_mutation": {"marker": "D", "linestyle": ":", "linewidth": 2.0, "markersize": 5},
            "V4_no_symmetry_breaking": {"marker": "X", "linestyle": "--", "linewidth": 2.0, "markersize": 5},
            "V5_random_feasible_start_spacing": {"marker": "P", "linestyle": "-", "linewidth": 2.0, "markersize": 5},
        },
        "legend_ncol": 3,
    },
]


def _nice_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    pad = max(0.03, 0.08 * (vmax - vmin if vmax > vmin else 1.0))
    lower = max(0.0, vmin - pad)
    upper = vmax + pad
    return lower, upper


def _plot_one_phase(data: pd.DataFrame, spec: dict, output_dir: Path, dpi: int) -> None:
    phase = spec["phase"]
    phase_df = data[data["phase"] == phase].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), sharey=True)
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)

    legend_handles = []
    legend_labels = []
    y_values_for_limits: list[float] = []

    for ax, instance in zip(axes, INSTANCE_ORDER):
        ax.set_title(instance, fontsize=19, pad=8)

        instance_df = phase_df[phase_df["instance_label"] == instance].copy()

        for method in spec["method_order"]:
            group = instance_df[instance_df["method_or_variant"] == method].sort_values("index")
            if group.empty:
                continue

            label = spec["labels"][method]
            style = spec["styles"][method]

            x = group["index"].to_numpy()
            y = group["hv_mean"].to_numpy()
            sd = group["hv_std"].to_numpy()

            y_values_for_limits.extend((y - sd).tolist())
            y_values_for_limits.extend((y + sd).tolist())

            handle, = ax.plot(
                x,
                y,
                label=label,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
            )
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)

            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

        ax.set_xlabel("Generation / logging index", fontsize=16)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=13)

    axes[0].set_ylabel("Best-so-far HV", fontsize=18)

    ymin, ymax = _nice_limits(y_values_for_limits)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=spec["legend_ncol"],
        frameon=False,
        fontsize=15,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22, wspace=0.04)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{spec['filename']}.png"
    pdf_path = output_dir / f"{spec['filename']}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-style Figure 7 panels from convergence_summary.csv"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to convergence_summary.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output figure files")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300)")
    args = parser.parse_args()

    data = pd.read_csv(args.input)

    required = {
        "phase",
        "method_or_variant",
        "instance_label",
        "index",
        "hv_mean",
        "hv_std",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    for spec in PHASE_SPECS:
        _plot_one_phase(data, spec, args.output_dir, args.dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())