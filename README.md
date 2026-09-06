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

This repository contains code and data for warehouse layout optimization. The main method is NSGA-II + Beam Search, with BS-only and random-restart Beam Search baselines. The repository also includes the public 30-seed paper campaign wrapper, a unified final-clean revision-evidence analyzer, layout editing, layout/archive rendering, paper-style Pareto plotting, and optional operational-layer diagnostics for the paper L1-L4 layouts.

## 2. Repository contents

- `whl_core/`: shared loading, grid, feasibility, and objective logic
- `whl_algorithms/`: structural optimization logic
- `whl_experiments/`: experiment runners, paper campaign orchestration, revision-evidence post-processing, and optional OPL helper scripts
- `whl_visualization/`: rendering and paper-style Pareto plotting
- `apps/`: Tkinter layout editor
- `configs/`: layout/configuration files
- `data/`: layout masks, paper plotting inputs, reproducibility evidence, and OPL paper data
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
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --no-figures --output-dir results\quick_nsga2_bs
```

BS-only:

```powershell
python -m whl_experiments.run_experiment_manager --method bs_only --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --no-figures --output-dir results\quick_bs_only
```

Random-restart Beam Search:

```powershell
python -m whl_experiments.run_experiment_manager --method random_restart_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --decode-budget 10 --no-figures --output-dir results\quick_rrbs
```

Detailed CLI usage, paper campaign commands, and parameter explanations are in `docs/cli_commands.md`.

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
- CLI commands and paper campaign examples: `docs/cli_commands.md`
- Revision evidence and unified analyzer protocol: `docs/revision_evidence.md`
- Layout data: `docs/layout_data.md`
- Benchmark sources: `docs/benchmark_sources.md`
- Operational-layer diagnostics: `docs/operational_layer.md`

## 9. Reproducibility

Fixed seeds are supported. Individual validation runs can use explicit small parameters, while paper-scale structural experiments should use `whl_experiments.run_revision_30seed_campaign`.

The paper campaign wrapper uses `auto_from_instance`, saves both scientific layout archives, and disables optimization-time figure rendering. Its default structural campaign sets are:

- Phase 11: Proposed, random-restart BS, and BS-only;
- Phase 12B: ablations V1-V5 only;
- Phase 12C: V6 and V7.

Phase 12B V0 is not rerun: the Phase 11 `proposed_nsga2_bs` results for the same instances and seeds are reused as the full-proposed baseline. Phase 12C likewise reuses those raw Phase-11 Proposed archives as V0, but recomputes the indicators inside the separate V0/V6/V7 comparison union. The V6b binding-depth check is a separate matched Demo-only Phase-12C diagnostic stored under `results/revision_final_30seed_nofg_v6b/`.

After the final-clean Phase 11, Phase 12B, Phase 12C, and V6b campaigns are complete, rebuild the structural reviewer/manuscript evidence with the single post-processing command:

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --results-root results\revision_final_30seed_nofg `
  --v6b-results-root results\revision_final_30seed_nofg_v6b `
  --output-dir data\reproducibility\revision_final_30seed_nofg\structural
```

The analyzer does not invoke optimization and requires a fresh/empty output directory. It keeps four indicator/reference families separate: Phase 11 (Proposed/BS-only/RRBS), Phase 12B (V0-V5), Phase 12C (V0/V6/V7), and the Demo V0/V6b diagnostic. Therefore reused V0 raw archives can have different HV/IGD+/OSD values across comparison families.

Raw generated experiment output remains ignored by default. Compact files required to verify the final revision campaigns are selectively tracked under `results/revision_final_30seed_nofg/` and `results/revision_final_30seed_nofg_v6b/`. Post-processed manuscript evidence is stored under `data/reproducibility/revision_final_30seed_nofg/structural/`.

See `docs/revision_evidence.md` for the output inventory, statistical protocol, V6b safeguards, manuscript mapping, and the rule that this post-processing should be run only after the timed campaigns are complete.

## 10. Citation

If you use this repository, please cite the associated manuscript once publication details are available.

## 11. License

See `LICENSE`.

## 12. Known limitations

- Structural optimization and diagnostic proxies only.
- No routed picker simulation.
- No throughput validation.
- OPL diagnostics are limited to the selected paper layouts and assumptions.
