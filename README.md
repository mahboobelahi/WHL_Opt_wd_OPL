### WHL_Opt_wd_OPL

***TOC:***
- [1. Overview](#1-overview)
- [2. Repository contents](#2-repository-contents)
- [3. Installation](#3-installation)
- [4. Quick validation](#4-quick-validation)
- [5. Layout editor](#5-layout-editor)
- [6. Plotting and rendering](#6-plotting-and-rendering)
- [7. Operational-layer diagnostics](#7-operational-layer-diagnostics)
- [8. Documentation](#8-documentation)
- [9. Reproducibility](#9-reproducibility)
- [10. Citation](#10-citation)
- [11. License](#11-license)
- [12. Known limitations](#12-known-limitations)


## 1. Overview

This repository contains code and data for warehouse layout optimization. The main method is NSGA-II + Beam Search, with BS-only and random-restart Beam Search baselines. The repository also includes layout editing, layout/archive rendering, paper-style Pareto plotting, and optional operational-layer diagnostics for the paper L1-L4 layouts.

## 2. Repository contents

- `whl_core/`: shared loading, grid, feasibility, and objective logic
- `whl_algorithms/`: structural optimization logic
- `whl_experiments/`: public runners and optional OPL helper scripts
- `whl_visualization/`: rendering and paper-style Pareto plotting
- `apps/`: Tkinter layout editor
- `configs/`: layout/configuration files
- `data/`: layout masks, paper plotting inputs, and OPL paper data
- `docs/`: detailed documentation

See `docs/architecture.md` for the architecture overview.

## 3. Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tested with Python 3.10.x.

## 4. Quick validation

Proposed NSGA-II + Beam Search:

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --output-dir results\quick_nsga2_bs
```

BS-only:

```powershell
python -m whl_experiments.run_experiment_manager --method bs_only --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --output-dir results\quick_bs_only
```

Random-restart Beam Search:

```powershell
python -m whl_experiments.run_experiment_manager --method random_restart_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --decode-budget 10 --output-dir results\quick_rrbs
```

Detailed CLI usage and parameter explanations are in `docs/cli_commands.md`.

## 5. Layout editor

The Tkinter editor can be used to inspect and edit grid mask files.

```powershell
python -m apps.layout_editor.launch_editor
```

## 6. Plotting and rendering

Layout/archive rendering commands are documented in `docs/cli_commands.md`. Paper-style Pareto plotting uses inputs in `data/plot_inputs/paper/`. Rendering and plotting are separate from the Tkinter editor preview.

## 7. Operational-layer diagnostics

Optional operational-layer diagnostics for the selected L1-L4 paper layouts are included under:

`data/operational_layer/`

and documented in:

`docs/operational_layer.md`

The operational layer is post-optimization only. It does not feed back into NSGA-II + Beam Search and is not part of the structural fitness evaluation.

## 8. Documentation

- Architecture: `docs/architecture.md`
- Inputs and outputs: `docs/input_output.md`
- CLI commands: `docs/cli_commands.md`
- Layout data: `docs/layout_data.md`
- Benchmark sources: `docs/benchmark_sources.md`
- Operational-layer diagnostics: `docs/operational_layer.md`

## 9. Reproducibility

Fixed seeds are supported. Quick validation uses explicit small parameters. Default proposed runs can use `auto_from_instance` / `auto_hyperparams` and may take longer. Generated results are written under `results/` and ignored by Git.

## 10. Citation

If you use this repository, please cite the associated manuscript once publication details are available.

## 11. License

See `LICENSE`.

## 12. Known limitations

- Structural optimization and diagnostic proxies only.
- No routed picker simulation.
- No throughput validation.
- OPL diagnostics are limited to the selected paper layouts and assumptions.