# CLI Commands

This document provides the public command-line entry points needed to run structural experiments, reproduce paper campaigns, save layout archives, and render results.

## 1. Setup

Run commands from the repository root:

```powershell
cd WHL_Opt_wd_OPL
```

Activate the project environment if required:

```powershell
.\.venv\Scripts\Activate.ps1
```

Experiment outputs are written under `results/` by default, or under an explicitly supplied output directory. The `results/` directory is ignored by Git.

For the authoritative option list of any runner, use `--help`, for example:

```powershell
python -m whl_experiments.run_experiment_manager --help
python -m whl_experiments.run_revision_30seed_campaign --help
```

---

## 2. Public Runners

| Runner | Recommended use | Purpose |
|---|---|---|
| `whl_experiments.run_experiment_manager` | Individual or custom scientific experiments | Runs the proposed NSGA-II + Beam Search method, BS-only baseline, or random-restart Beam Search baseline. |
| `whl_experiments.run_revision_30seed_campaign` | Paper campaign reproduction | Orchestrates Phase 11/12 tasks by repeatedly calling `run_experiment_manager`, with worker control, dry-run, resume, and instance selection. |
| `whl_experiments.render_saved_layouts` | Render one saved archive | Renders layouts from one `.npz` archive and its JSON index. |
| `whl_experiments.render_experiment_archives` | Batch-render a completed experiment | Discovers and renders saved archives under a completed experiment directory. |
| `whl_visualization.paper_pareto_plots` | Paper-style plotting | Regenerates manuscript-style Pareto plots from prepared CSV inputs. |

`run_experiment_manager` contains the scientific experiment workflow. `run_revision_30seed_campaign` is an orchestration layer around that runner; it does not implement a separate optimization method.

---

## 3. Individual Scientific Experiments

Supported methods are:

- `proposed_nsga2_bs`
- `random_restart_bs`
- `bs_only`

The commands below are bounded smoke tests, not paper-scale experiments.

### 3.1 Proposed NSGA-II + Beam Search

```powershell
python -m whl_experiments.run_experiment_manager `
  --method proposed_nsga2_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --population-size 8 `
  --generations 5 `
  --beam-width 3 `
  --max-depth 8 `
  --no-figures `
  --output-dir results\quick_nsga2_bs
```

### 3.2 Random-restart Beam Search

```powershell
python -m whl_experiments.run_experiment_manager `
  --method random_restart_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --beam-width 3 `
  --max-depth 8 `
  --decode-budget 10 `
  --no-figures `
  --output-dir results\quick_rrbs
```

### 3.3 BS-only

```powershell
python -m whl_experiments.run_experiment_manager `
  --method bs_only `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --beam-width 3 `
  --max-depth 8 `
  --no-figures `
  --output-dir results\quick_bs_only
```

### 3.4 Automatic instance-based budget

If manual search-budget options are omitted, the default `auto_from_instance` policy resolves the search parameters from the instance geometry:

```powershell
python -m whl_experiments.run_experiment_manager `
  --method proposed_nsga2_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --no-figures `
  --output-dir results\auto_budget_example
```

This command can be substantially more expensive than the bounded smoke tests above.

### 3.5 Figure control

For individual manager runs, figure rendering is enabled when `--no-figures` is omitted.

**With figures:**

```powershell
python -m whl_experiments.run_experiment_manager `
  --method proposed_nsga2_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --population-size 8 `
  --generations 5 `
  --beam-width 3 `
  --max-depth 8 `
  --output-dir results\quick_with_figures
```

**Without figures:**

```powershell
python -m whl_experiments.run_experiment_manager `
  --method proposed_nsga2_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --population-size 8 `
  --generations 5 `
  --beam-width 3 `
  --max-depth 8 `
  --no-figures `
  --output-dir results\quick_without_figures
```

`--no-figures` suppresses layout PNG rendering only. It does not disable CSV outputs, profiling outputs, generation objectives, or requested `.npz`/JSON archives.

---

## 4. Paper Campaign Runner

Use `run_revision_30seed_campaign` for the replicated structural campaigns. Each task is one manager invocation for a selected method or variant, instance, and seed.

Campaign tasks automatically forward:

```text
--budget-policy auto_from_instance
--archive-layouts both
--no-figures
```

Therefore, paper campaigns save scientific archives and evidence files without rendering layout figures during optimization. Figures can be generated afterward from the saved archives.

### 4.1 Phase 11 core campaign

The core Phase 11 design contains 4 instances, 30 seeds (`101`-`130`), and 3 methods, giving 360 tasks.

Dry-run:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed `
  --phase phase11 `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3 `
  --dry-run
```

Expected:

```text
dry_run_tasks=360
```

Execution:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed `
  --phase phase11 `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

For a fresh reproduction, use a new campaign root or remove the previous campaign output first. If the same campaign is interrupted, rerun the same command with `--resume` to skip tasks already recorded as completed.

### 4.2 Explicit supplementary instance list

An explicit comma-separated list can be supplied with `--instance-list`. The following example uses one seed over 11 layouts and all three Phase 11 methods, giving 33 tasks:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_supplementary `
  --phase phase11 `
  --seed-start 101 `
  --seed-end 101 `
  --instance-list Answer_Set_layout_AW_1,Answer_Set_layout_AW_2,Answer_Set_layout_AW_3,demo_layout_door_bottom_AW_2,demo_layout_door_bottom_AW_3,demo_layout_door_left_AW_2,demo_layout_door_left_AW_3,demo_layout_door_UB_AW_2,demo_layout_door_UB_AW_3,Gyorgy-KOVACS_MWH_Narrow_AW_4,Gyorgy-KOVACS_MWH_Wide_AW_5 `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

Add `--dry-run` first when only task enumeration is required.

### 4.3 Predefined instance scopes

The campaign runner accepts:

- `core`: the four replicated inferential instances;
- `stress`: the predefined supplementary stress subset;
- `all`: `core + stress` presets.

`all` refers to all predefined campaign presets, not every mask stored in the repository. Use `--instance-list` for arbitrary repository masks.

---

## 5. Parameter Reference

### 5.1 `run_experiment_manager`

#### Run selection and output

| Option | Meaning |
|---|---|
| `--config PATH` | Experiment configuration file. Default: `configs/experiment_plan.yaml`. |
| `--instances ...` | One or more instance names or paths; comma-separated lists are also accepted. |
| `--seeds SEEDS` | Comma-separated random seeds. |
| `--method METHOD` | `proposed_nsga2_bs`, `random_restart_bs`, or `bs_only`. |
| `--experiment-id ID` | Explicit experiment folder identifier; otherwise a timestamped ID is created. |
| `--output-dir PATH` | Experiment output root. Default: `results/experiments`. |
| `--dry-run` | Build and print the experiment plan without running optimization. |

#### Search budget

| Option | Meaning |
|---|---|
| `--budget-policy {fixed,auto_from_instance}` | Search-budget source. Default: `auto_from_instance`. |
| `--population-size N` | Population size or RRBS population-equivalent value. |
| `--generations N` | Generation count or RRBS generation-equivalent value. |
| `--decode-budget N` | Total random restarts for `random_restart_bs`; otherwise derived from population size × generations. |
| `--beam-width N` | Beam Search width. |
| `--max-depth N` | Beam Search maximum depth. |
| `--beam-width-delta N` | Additive increase to the automatically resolved beam width. |

#### Method and ablation controls

| Option | Meaning |
|---|---|
| `--sorting-rule-mode {sampled_pool,fixed}` | Beam sorting-rule selection mode. |
| `--sorting-rule NAME` | Sorting rule name. |
| `--fixed-sorting-rule NAME` | Alias for `--sorting-rule`. |
| `--adaptive-weight-mode {adaptive,fixed}` | Beam weight policy. |
| `--fixed-beam-w1 FLOAT` | Fixed Beam Search weight `w1`. Default: `0.5`. |
| `--fixed-beam-w2 FLOAT` | Fixed Beam Search weight `w2`. Default: `0.5`. |
| `--fixed-beam-lambda FLOAT` | Fixed Beam Search multiplier `lambda`. Default: `0.1`. |
| `--mutation-mode MODE` | `weighted`, `uniform`, or `weighted_no_symmetry_breaking`. |
| `--initialization-spacing-mode MODE` | Initial-population spacing policy. |
| `--ablation-variant LABEL` | Label stored for an ablation run. |
| `--bs-rule-policy {all_rules,fixed}` | BS-only sorting-rule policy. |
| `--bs-weight-policy fixed` | BS-only weight policy. |

#### Evidence and figure outputs

| Option | Meaning |
|---|---|
| `--no-figures` | Disable layout figure rendering. |
| `--archive-layouts MODE` | `none`, `generation_elites`, `final_ranked`, `both`, `all_debug`, or `all_candidates_debug`. |
| `--archive-rank-max N` | Maximum rank retained by rank-filtered archives. Default: `3`. |
| `--profile-light` | Write lightweight runtime profiling CSVs. |
| `--save-generation-objectives` | Write per-generation rank-0 objective rows. |

### 5.2 `run_revision_30seed_campaign`

| Option | Meaning |
|---|---|
| `--campaign-root PATH` | Campaign output, log, and manifest root. Default: `results/revision_30seed_campaign`. |
| `--phase {phase11,phase12b,phase12c}` | Required campaign phase. |
| `--seed-start N` | First inclusive seed. Default: `101`. |
| `--seed-end N` | Last inclusive seed. Default: `130`. |
| `--instances {core,stress,all}` | Predefined instance scope. Default: `core`. |
| `--instance-list LIST` | Explicit comma-separated mask names; overrides `--instances`. Optional `.npz` suffixes are accepted. |
| `--only-instance NAME` | Restrict execution to one predefined instance. Do not combine with `--instance-list`. |
| `--only-variant NAME` | Restrict execution to one Phase 11 method or Phase 12 variant. |
| `--max-workers N` | Maximum number of manager tasks executed concurrently. Default: `3`. |
| `--dry-run` | Write the task manifest without executing optimization. |
| `--resume` | Skip tasks with an existing completed `experiment_summary.csv`. Use only when continuing the same campaign configuration. |
| `--archive-rank-max N` | Maximum archived rank forwarded to the manager. Default: `3`. |
| `--profile-light` | Enable lightweight runtime profiling for each task. |
| `--save-generation-objectives` | Save generation-level objective evidence for each task. |

---

## 6. Saved Outputs and Archives

A manager run writes per-run outputs under:

```text
<output-dir>\<experiment-id>\runs\<instance>\seed_<seed>\
```

Common per-run outputs include:

- `candidates.csv`
- `generation_summary.csv`
- `run_metadata.json`
- `runtime_profile_summary.csv` when `--profile-light` is enabled
- `generation_profile.csv` when `--profile-light` is enabled
- `generation_objectives.csv` when `--save-generation-objectives` is enabled

Experiment-level outputs include `experiment_metadata.json` and `experiment_summary.csv`.

Archive modes are controlled by `--archive-layouts`:

| Mode | Saved layout set |
|---|---|
| `none` | No layout archive. |
| `generation_elites` | Eligible generation-level layouts, filtered by `--archive-rank-max`. |
| `final_ranked` | Unique feasible layouts re-ranked for final archival, filtered by `--archive-rank-max`. |
| `both` | Both `generation_elites` and `final_ranked`. |
| `all_debug` | Unique feasible evaluated layouts for debugging. |
| `all_candidates_debug` | Unique evaluated layouts, including infeasible candidates, for debugging. |

Typical archive files are:

```text
generation_elites.npz
generation_elites_index.json
final_ranked_layouts.npz
final_ranked_layouts_index.json
```

Paper campaign tasks use `--archive-layouts both`.

---

## 7. Rendering Saved Layouts

Rendering is post-processing and does not rerun optimization.

### 7.1 Render one archive

Use `render_saved_layouts` when the archive and JSON index are known:

```powershell
python -m whl_experiments.render_saved_layouts `
  --archive results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts.npz `
  --index results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts_index.json `
  --filter rank0_to_rank3
```

Important options:

| Option | Meaning |
|---|---|
| `--archive PATH` | Archive `.npz` file. Required. |
| `--index PATH` | Archive JSON index. Required. |
| `--filter FILTER` | `all`, `rank0`, `rank0_to_rank3`, `rank0_to_rank4`, or `selected`. Default: `all`. |
| `--output-dir PATH` | PNG output directory. |
| `--max-layouts N` | Maximum number of layouts to render. |
| `--dpi N` | Output resolution. Default: `150`. |
| `--title-fields FIELDS` | Comma-separated metadata fields included in titles. |
| `--title-format {fields,metrics_trace}` | Figure-title format. |
| `--no-legend` | Hide the legend. |
| `--show-coords` | Show coordinate labels. |

### 7.2 Batch-render an experiment

Use `render_experiment_archives` to discover archives under a completed experiment directory:

```powershell
python -m whl_experiments.render_experiment_archives `
  --experiment-dir results\quick_nsga2_bs\proposed_nsga2_bs_YYYYMMDD_HHMMSS `
  --archive-type final_ranked `
  --filter rank0_to_rank3
```

Important options:

| Option | Meaning |
|---|---|
| `--experiment-dir PATH` | Completed experiment directory. Required. |
| `--archive-type TYPE` | `final_ranked`, `generation_elites`, `all_debug`, or `all_candidates_debug`. Default: `final_ranked`. |
| `--filter FILTER` | `all`, `rank0`, `rank0_to_rank3`, `rank0_to_rank4`, or `selected`. Default: `rank0_to_rank3`. |
| `--output-dir PATH` | Root PNG output directory. |
| `--max-layouts N` | Maximum layouts rendered per discovered archive. |
| `--dpi N` | Output resolution. Default: `150`. |
| `--title-fields FIELDS` | Comma-separated metadata fields included in titles. |
| `--title-format {fields,metrics_trace}` | Figure-title format. |
| `--no-legend` | Hide the legend. |
| `--show-coords` | Show coordinate labels. |
| `--dry-run` | Discover render jobs without writing PNGs. |
| `--overwrite` | Allow replacement of existing rendered outputs. |

---

## 8. Paper-Style Pareto Plots

Pareto plotting is separate from warehouse-layout rendering.

```powershell
python -m whl_visualization.paper_pareto_plots
```

The script reads prepared inputs from:

```text
data/plot_inputs/paper/
```

The documented paper inputs are:

```text
121_atefeh_published_reference_metrics.csv
121_atefeh_rank03_points.csv
121_kov1ow4_rank03_points.csv
```

---

## 9. Layout Editor and Instance Rules

Launch the layout editor with:

```powershell
python -m apps.layout_editor.launch_editor
```

The editor is used to create, duplicate, edit, delete, and preview layout mask bundles. Saved masks are stored under `data/instances/masks/` and registered through the layout registry.

Instance-use rules relevant to reproduction:

- `AT_1.npz` through `AT_13.npz` are reference/comparison-only layouts and must not be used as optimization instances.
- `AT_S_comercial_layout_AW_3.npz` is an optimization instance.
- Demo and literature/test masks may be supplied to the experiment manager where appropriate.

---

## 10. Operational-Layer Diagnostics

Operational-layer diagnostics are post-optimization reproduction helpers for the fixed L1-L4 representative layout panel. They are not part of NSGA-II fitness evaluation and do not feed back into structural optimization.

See `docs/operational_layer.md` for the supported operational-layer scripts, inputs, outputs, and limitations.
