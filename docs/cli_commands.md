# CLI Commands

## 1. Before Running Commands

Run commands from the repository root:

```powershell
cd WHL_Opt_wd_OPL
```

Activate the virtual environment if your system Python does not already have the required packages:

```powershell
.\.venv\Scripts\Activate.ps1
```

Experiment outputs are written under `results/` by default, or under the folder passed with `--output-dir`. The `results/` folder is ignored by Git.

---
### Editor launch

```powershell
python -m apps.layout_editor.launch_editor
```


## 2. Available Public Runners

| Runner | Recommended use | Purpose |
|---|---|---|
| `whl_experiments.run_experiment_manager` | **Main public CLI** | Runs the documented workflows for the proposed NSGA-II + Beam Search method, BS-only baseline, and random-restart Beam Search baseline with consistent output folders and options. |
| `whl_experiments.render_saved_layouts` | **Public rendering utility** | Renders saved layout arrays or archive outputs after an optimization run. Use this for visual inspection of generated layouts. |
| `whl_experiments.render_experiment_archives` | **Public rendering utility** | Renders layouts from experiment archive folders, including filtered archive/rank outputs if supported by the script. |
| `whl_visualization.paper_pareto_plots` | **Paper-style plotting utility** | Regenerates manuscript-style Pareto plots from the prepared CSV inputs in `data/plot_inputs/paper/`. |

---

## 3. Recommended Public Entry Point

`run_experiment_manager.py` is the recommended public CLI for:

- `proposed_nsga2_bs`
- `bs_only`
- `random_restart_bs`

---

## 4. Method Commands

These are bounded test commands, not full paper-scale experiments.

### Proposed NSGA-II + Beam Search

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --output-dir results\quick_nsga2_bs
```

### BS-only

```powershell
python -m whl_experiments.run_experiment_manager --method bs_only --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --output-dir results\quick_bs_only
```

### Random-restart Beam Search

```powershell
python -m whl_experiments.run_experiment_manager --method random_restart_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --decode-budget 10 --output-dir results\quick_rrbs
```

### Default auto-budget proposed command

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --output-dir results\default_nsga2_bs
```

Do not use the default auto-budget command as a quick validation command. It may take longer because the default budget policy is `auto_from_instance`, which uses `auto_hyperparams`.

---

## 5. Parameter Reference

The original parameter tables were too wide. They are rewritten below as two-column tables so that each parameter has its details in one readable cell.

### 5.1 `run_experiment_manager`

| Parameter | Details |
|---|---|
| `--config` | **Meaning:** experiment plan config path.<br>**Type:** path.<br>**Default:** `configs/experiment_plan.yaml`.<br>**Example:** `--config configs/experiment_plan.yaml` |
| `--instances` | **Meaning:** instance names or paths.<br>**Type:** one or more strings. Comma-separated lists are also accepted.<br>**Default:** config/default discovery.<br>**Example:** `--instances Gyorgy-KOVACS_WH_Narrow_AW_4` |
| `--seeds` | **Meaning:** random seeds.<br>**Type:** comma-separated string.<br>**Default:** config/default seed.<br>**Example:** `--seeds 101,102` |
| `--method` | **Meaning:** method to run.<br>**Accepted:** `proposed_nsga2_bs`, `bs_only`, `random_restart_bs`.<br>**Default:** `proposed_nsga2_bs`.<br>**Example:** `--method bs_only` |
| `--experiment-id` | **Meaning:** explicit experiment folder name.<br>**Type:** string.<br>**Default:** auto timestamp.<br>**Example:** `--experiment-id test_run` |
| `--population-size` | **Meaning:** NSGA-II population size, or random-restart equivalent.<br>**Type:** integer.<br>**Default:** method/config/auto value.<br>**Example:** `--population-size 8` |
| `--generations` | **Meaning:** NSGA-II generation count, or random-restart equivalent.<br>**Type:** integer.<br>**Default:** method/config/auto value.<br>**Example:** `--generations 5` |
| `--decode-budget` | **Meaning:** total random restarts for `random_restart_bs`.<br>**Type:** integer.<br>**Default for RRBS:** population size × generations.<br>**Example:** `--decode-budget 10` |
| `--beam-width` | **Meaning:** Beam Search width.<br>**Type:** integer.<br>**Default:** method/config/auto value.<br>**Example:** `--beam-width 3` |
| `--max-depth` | **Meaning:** Beam Search maximum depth.<br>**Type:** integer.<br>**Default:** method/config/auto value.<br>**Example:** `--max-depth 8` |
| `--beam-width-delta` | **Meaning:** additive increase to auto beam width.<br>**Type:** integer.<br>**Default:** `0`.<br>**Example:** `--beam-width-delta 1` |
| `--output-dir` | **Meaning:** experiment output root.<br>**Type:** path.<br>**Default:** `results/experiments`.<br>**Example:** `--output-dir results\quick_nsga2_bs` |
| `--no-figures` | **Meaning:** disable selected layout figures.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--no-figures` |
| `--dry-run` | **Meaning:** plan runs without executing optimization.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--dry-run` |
| `--budget-policy` | **Meaning:** budget source policy.<br>**Accepted:** `fixed`, `auto_from_instance`.<br>**Default:** `auto_from_instance`.<br>**Example:** `--budget-policy fixed` |
| `--sorting-rule-mode` | **Meaning:** Beam sorting rule selection mode.<br>**Accepted:** `sampled_pool`, `fixed`.<br>**Default:** `sampled_pool`.<br>**Example:** `--sorting-rule-mode fixed` |
| `--sorting-rule` | **Meaning:** sorting rule name.<br>**Type:** string.<br>**Default:** `PF_LS_RP`.<br>**Example:** `--sorting-rule PF_LS_RP` |
| `--fixed-sorting-rule` | **Meaning:** alias for `--sorting-rule`.<br>**Type:** string.<br>**Default:** `PF_LS_RP`.<br>**Example:** `--fixed-sorting-rule PF_LS_RP` |
| `--adaptive-weight-mode` | **Meaning:** Beam weight mode.<br>**Accepted:** `adaptive`, `fixed`.<br>**Default:** `adaptive`.<br>**Example:** `--adaptive-weight-mode fixed` |
| `--fixed-beam-w1` | **Meaning:** fixed beam weight 1.<br>**Type:** float.<br>**Default:** `0.5`.<br>**Example:** `--fixed-beam-w1 0.5` |
| `--fixed-beam-w2` | **Meaning:** fixed beam weight 2.<br>**Type:** float.<br>**Default:** `0.5`.<br>**Example:** `--fixed-beam-w2 0.5` |
| `--fixed-beam-lambda` | **Meaning:** fixed beam lambda.<br>**Type:** float.<br>**Default:** `0.1`.<br>**Example:** `--fixed-beam-lambda 0.1` |
| `--mutation-mode` | **Meaning:** mutation policy.<br>**Accepted:** `weighted`, `uniform`, `weighted_no_symmetry_breaking`.<br>**Default:** `weighted`.<br>**Example:** `--mutation-mode uniform` |
| `--initialization-spacing-mode` | **Meaning:** initial population spacing policy.<br>**Accepted:** `feasible_start_adaptive_spacing`, `random_feasible_start_no_adaptive_spacing`.<br>**Default:** `feasible_start_adaptive_spacing`.<br>**Example:** `--initialization-spacing-mode feasible_start_adaptive_spacing` |
| `--ablation-variant` | **Meaning:** ablation label.<br>**Type:** string.<br>**Default:** `none`.<br>**Example:** `--ablation-variant none` |
| `--bs-rule-policy` | **Meaning:** BS-only sorting rule policy.<br>**Accepted:** `all_rules`, `fixed`.<br>**Default:** `all_rules`.<br>**Example:** `--bs-rule-policy fixed` |
| `--bs-weight-policy` | **Meaning:** BS-only weight policy.<br>**Accepted:** `fixed`.<br>**Default:** `fixed`.<br>**Example:** `--bs-weight-policy fixed` |
| `--archive-layouts` | **Meaning:** save layout archives.<br>**Accepted:** `none`, `generation_elites`, `final_ranked`, `both`, `all_debug`, `all_candidates_debug`.<br>**Default:** `none`.<br>**Example:** `--archive-layouts final_ranked` |
| `--archive-rank-max` | **Meaning:** maximum rank saved in eligible archives.<br>**Type:** integer.<br>**Default:** `3`.<br>**Example:** `--archive-rank-max 3` |
| `--profile-light` | **Meaning:** write lightweight runtime profile CSVs.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--profile-light` |
| `--save-generation-objectives` | **Meaning:** write rank-0 objective rows per generation.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--save-generation-objectives` |

### 5.2 `render_saved_layouts`

| Parameter | Details |
|---|---|
| `--archive` | **Meaning:** saved archive `.npz` path.<br>**Type:** path.<br>**Required:** yes.<br>**Example:** `--archive results\...\final_ranked_layouts.npz` |
| `--index` | **Meaning:** archive index JSON path.<br>**Type:** path.<br>**Required:** yes.<br>**Example:** `--index results\...\final_ranked_layouts_index.json` |
| `--output-dir` | **Meaning:** PNG output folder.<br>**Type:** path.<br>**Default:** inferred from archive path.<br>**Example:** `--output-dir results\rendered` |
| `--filter` | **Meaning:** archive metadata filter.<br>**Accepted:** `all`, `rank0`, `rank0_to_rank3`, `rank0_to_rank4`, `selected`.<br>**Default:** `all`.<br>**Example:** `--filter rank0_to_rank3` |
| `--max-layouts` | **Meaning:** limit rendered layout count.<br>**Type:** integer.<br>**Default:** no limit.<br>**Example:** `--max-layouts 20` |
| `--dpi` | **Meaning:** PNG resolution.<br>**Type:** integer.<br>**Default:** `150`.<br>**Example:** `--dpi 200` |
| `--title-fields` | **Meaning:** metadata fields in figure titles.<br>**Type:** comma-separated string.<br>**Default:** built-in field list.<br>**Example:** `--title-fields seed,rank,candidate_id` |
| `--title-format` | **Meaning:** title formatting mode.<br>**Accepted:** `fields`, `metrics_trace`.<br>**Default:** `fields`.<br>**Example:** `--title-format metrics_trace` |
| `--no-legend` | **Meaning:** hide legend.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--no-legend` |
| `--show-coords` | **Meaning:** show coordinate labels.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--show-coords` |

### 5.3 `render_experiment_archives`

| Parameter | Details |
|---|---|
| `--experiment-dir` | **Meaning:** completed experiment folder.<br>**Type:** path.<br>**Required:** yes.<br>**Example:** `--experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_...` |
| `--archive-type` | **Meaning:** archive type to render.<br>**Accepted:** `final_ranked`, `generation_elites`, `all_debug`, `all_candidates_debug`.<br>**Default:** `final_ranked`.<br>**Example:** `--archive-type final_ranked` |
| `--filter` | **Meaning:** archive metadata filter.<br>**Accepted:** `all`, `rank0`, `rank0_to_rank3`, `rank0_to_rank4`, `selected`.<br>**Default:** `rank0_to_rank3`.<br>**Example:** `--filter rank0` |
| `--output-dir` | **Meaning:** root output folder.<br>**Type:** path.<br>**Default:** inside each run folder.<br>**Example:** `--output-dir results\rendered_archives` |
| `--max-layouts` | **Meaning:** limit rendered layout count per job.<br>**Type:** integer.<br>**Default:** no limit.<br>**Example:** `--max-layouts 10` |
| `--dpi` | **Meaning:** PNG resolution.<br>**Type:** integer.<br>**Default:** `150`.<br>**Example:** `--dpi 200` |
| `--title-fields` | **Meaning:** metadata fields in figure titles.<br>**Type:** comma-separated string.<br>**Default:** built-in field list.<br>**Example:** `--title-fields seed,rank,candidate_id` |
| `--title-format` | **Meaning:** title formatting mode.<br>**Accepted:** `fields`, `metrics_trace`.<br>**Default:** `fields`.<br>**Example:** `--title-format fields` |
| `--no-legend` | **Meaning:** hide legend.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--no-legend` |
| `--show-coords` | **Meaning:** show coordinate labels.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--show-coords` |
| `--dry-run` | **Meaning:** discover render jobs without writing PNGs.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--dry-run` |
| `--overwrite` | **Meaning:** explicit overwrite intent.<br>**Type:** flag.<br>**Default:** false.<br>**Example:** `--overwrite` |

---

## 6. Archive Saving and Layout Rendering

Optimization runs write per-run CSV and JSON outputs under:

```text
<output-dir>\<experiment-id>\runs\<instance>\seed_<seed>\
```

Common outputs include:

- `run_metadata.json`
- `candidate_rows.csv`
- `generation_summary.csv`
- `experiment_metadata.json` at experiment root
- `experiment_summary.csv` at experiment root

Archive saving is controlled by:

```powershell
--archive-layouts {none,generation_elites,final_ranked,both,all_debug,all_candidates_debug}
--archive-rank-max 3
```

Archive files written by supported modes include:

- `generation_elites.npz` and `generation_elites_index.json`
- `final_ranked_layouts.npz` and `final_ranked_layouts_index.json`
- `all_debug_layouts.npz` and `all_debug_layouts_index.json`
- `all_candidates_debug_layouts.npz` and `all_candidates_debug_layouts_index.json`

Example archive-saving command:

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --archive-layouts final_ranked --archive-rank-max 3 --output-dir results\quick_nsga2_bs
```

---

## 7. Render Layouts After Optimization

### A. Render layouts after an optimization run

Use `render_experiment_archives` to batch-render archives under a completed experiment folder:

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked --filter rank0_to_rank3
```

### B. Render only selected ranks

Rank-specific rendering is exposed through `--filter`, not through `--rank` or `--ranks`.

Supported filters:

- `all`
- `rank0`
- `rank0_to_rank3`
- `rank0_to_rank4`
- `selected`

Example:

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked --filter rank0
```

### C. Render a specified result folder or archive

For a whole experiment folder:

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked
```

For one archive and index:

```powershell
python -m whl_experiments.render_saved_layouts --archive results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts.npz --index results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts_index.json --filter rank0_to_rank3
```

### D. Set an output folder

Both rendering CLIs support `--output-dir`:

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked --output-dir results\rendered_archives
```

### E. Coordinate, gridline, and pick-face controls

- `--show-coords` shows coordinate labels.
- `--no-legend` hides the legend.
- `--dpi` controls PNG resolution.
- Gridline and pick-face rendering behavior is implemented by the renderer and is not exposed as separate CLI switches.

---

## 8. Paper-Style Pareto Plotting

Paper-style Pareto plotting is separate from editor preview.

Command:

```powershell
python -m whl_visualization.paper_pareto_plots
```

Inputs are read from:

```text
data/plot_inputs/paper/
```

Required CSV inputs:


- `121_atefeh_published_reference_metrics.csv`
- `121_atefeh_rank03_points.csv`
- `121_kov1ow4_rank03_points.csv`

Outputs are written by the script to `whl_visualization/`, including manuscript-style PNG files and a plotting audit/report file if generated by the script.

---

## 9. Tkinter Editor Command

Launch the layout editor with:

```powershell
python -m apps.layout_editor.launch_editor
```

The editor is used to create, duplicate, edit, delete, and preview layout mask bundles. Saved masks are written under `data/instances/masks/` and registered through the layout registry. The launcher offers the original Tkinter canvas grid editor behavior and an optional Matplotlib editor backend. Paper-style Pareto plotting is separate from editor preview.

---

## 10. Layout Selection Rules

- `AT_1.npz` through `AT_13.npz` are reference/comparison-only layouts.
- Do not use `AT_1.npz` through `AT_13.npz` for optimization runs.
- `AT_S_comercial_layout_AW_3.npz` is an optimization layout.
- Demo layouts and literature/test layouts may be optimized.

---

## 11. Examples

### Proposed NSGA-II + Beam Search quick validation

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --output-dir results\quick_nsga2_bs
```

### BS-only

```powershell
python -m whl_experiments.run_experiment_manager --method bs_only --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --output-dir results\quick_bs_only
```

### Random-restart Beam Search

```powershell
python -m whl_experiments.run_experiment_manager --method random_restart_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --beam-width 3 --max-depth 8 --decode-budget 10 --output-dir results\quick_rrbs
```

### Dry-run

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --dry-run
```

### Archive-saving

```powershell
python -m whl_experiments.run_experiment_manager --method proposed_nsga2_bs --instances Gyorgy-KOVACS_WH_Narrow_AW_4 --seeds 101 --population-size 8 --generations 5 --beam-width 3 --max-depth 8 --archive-layouts final_ranked --archive-rank-max 3 --output-dir results\quick_nsga2_bs
```

### Render after optimization

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked --filter rank0_to_rank3
```

### Rank-specific render

```powershell
python -m whl_experiments.render_experiment_archives --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS --archive-type final_ranked --filter rank0
```

### Paper Pareto plot

```powershell
python -m whl_visualization.paper_pareto_plots
```

---

## 12. Optional Operational-layer Diagnostics

Operational-layer diagnostics are optional post-optimization reproduction helpers for the paper's fixed L1-L4 representative layout panel. They are not part of NSGA-II fitness evaluation and do not feed back into the optimizer.

The restored OPL scripts use fixed project-relative paths and do not expose public argparse options. See `docs/operational_layer.md` for the supported helper modules, input data, outputs, and limitations.
