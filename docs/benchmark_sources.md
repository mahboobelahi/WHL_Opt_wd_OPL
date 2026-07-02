***TOC:***

- [1. Benchmark Sources](#1-benchmark-sources)
  - [2. Included methods](#2-included-methods)
  - [3. Layout groups](#3-layout-groups)
    - [Demo/example layouts](#demoexample-layouts)
    - [Optimization benchmark layouts](#optimization-benchmark-layouts)
    - [Reference/comparison-only layouts](#referencecomparison-only-layouts)
  - [4. Paper plotting inputs](#4-paper-plotting-inputs)
  - [5. Operational-layer diagnostic material](#5-operational-layer-diagnostic-material)
-----

# 1. Benchmark Sources

This repository contains the WHL structural optimization workflow, layout masks, paper plotting inputs, and the optional operational-layer diagnostic material used for the revised manuscript.

## 2. Included methods

The public optimization workflow includes:

- proposed NSGA-II + Beam Search;
- BS-only baseline;
- random-restart Beam Search baseline.

These methods are exposed through the public command-line workflow described in `docs/cli_commands.md`.

## 3. Layout groups

The repository contains three main layout groups.

### Demo/example layouts

These are small synthetic masks used for quick checks and examples. They are used by the optimization runners.

### Optimization benchmark layouts

These are the main layouts used by the structural optimization workflow. This group includes the Kovács-derived layouts, Answer Set layouts, and Atefeh(`AT_S_comercial_layout_AW_3.npz`).

### Reference/comparison-only layouts

`AT_1.npz` through `AT_13.npz` are included for reference, comparison, preview, and plotting workflows. They are not part of default optimization discovery and should not be treated as optimization instances.

See `docs/layout_data.md` for the layout classification and usage rules.

## 4. Paper plotting inputs

Paper-style Pareto plotting uses the prepared CSV inputs under:

```text
data/plot_inputs/paper/
```

The required paper Pareto plotting inputs are:

- `121_atefeh_published_reference_metrics.csv`
- `121_atefeh_rank03_points.csv`
- `121_kov1ow4_rank03_points.csv`

The plotting command and parameters are documented in `docs/cli_commands.md`.

## 5. Operational-layer diagnostic material

Optional operational-layer diagnostics for the paper L1-L4 layouts are included under:

```text
data/operational_layer/
```

This material is post-optimization only. It does not feed back into NSGA-II + Beam Search and is not part of the structural fitness evaluation.

See `docs/operational_layer.md` for details.
