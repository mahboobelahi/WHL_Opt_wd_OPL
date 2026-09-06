# CLI Commands

This document lists the public command-line entry points for structural optimization, paper campaign reproduction, unified revision-evidence post-processing, archive rendering, Pareto plotting, and the optional operational-layer diagnostics.

## 1. Setup

Run commands from the repository root:

```powershell
cd WHL_Opt_wd_OPL
```

Activate the environment if required:

```powershell
.\.venv\Scripts\Activate.ps1
```

Generated experiment outputs are written under `results/` unless another output root is supplied. The `results/` directory is ignored by Git.

For the authoritative option list:

```powershell
python -m whl_experiments.run_experiment_manager --help
python -m whl_experiments.run_revision_30seed_campaign --help
python -m whl_experiments.analyze_revision_campaign_evidence --help
```

---

## 2. Public Runners

| Runner | Recommended use | Purpose |
|---|---|---|
| `whl_experiments.run_experiment_manager` | Individual/custom experiment | Runs Proposed NSGA-II + Beam Search, BS-only, or random-restart Beam Search. |
| `whl_experiments.run_revision_30seed_campaign` | Paper campaign reproduction | Orchestrates Phase 11, Phase 12B, Phase 12C, and the targeted V6b task family by calling `run_experiment_manager`. |
| `whl_experiments.analyze_revision_campaign_evidence` | Final-clean structural evidence post-processing | Rebuilds the unified Phase11, Phase12B, Phase12C, and separate matched V6b reviewer/manuscript evidence without rerunning optimization. |
| `whl_experiments.render_saved_layouts` | Render one saved archive | Renders layouts from one `.npz` archive and JSON index. |
| `whl_experiments.render_experiment_archives` | Batch-render a completed experiment | Discovers and renders saved archives under a completed experiment directory. |
| `whl_visualization.paper_pareto_plots` | Paper-style Pareto plotting | Regenerates manuscript-style plots from prepared CSV inputs. |

`run_revision_30seed_campaign` is an orchestration layer only. It does not implement a separate optimization algorithm. `analyze_revision_campaign_evidence` is post-processing only and never invokes an optimizer.

---

## 3. Individual Scientific Experiments

Supported methods are:

- `proposed_nsga2_bs`
- `random_restart_bs`
- `bs_only`

The examples below are bounded smoke tests, not paper-scale experiments.

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

If manual search-budget options are omitted, `auto_from_instance` resolves the search parameters from the instance geometry:

```powershell
python -m whl_experiments.run_experiment_manager `
  --method proposed_nsga2_bs `
  --instances Gyorgy-KOVACS_WH_Narrow_AW_4 `
  --seeds 101 `
  --no-figures `
  --output-dir results\auto_budget_example
```

This can be substantially more expensive than the bounded smoke tests.

### 3.5 Figure control

Omit `--no-figures` to allow layout PNG rendering. Supply it to disable layout rendering.

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

`--no-figures` suppresses layout PNG rendering only. CSV outputs, profiling files, generation-objective files, and requested archives are unaffected.

---

## 4. Paper Campaign Runner

Use `run_revision_30seed_campaign` for the replicated structural experiments. Each task is one `run_experiment_manager` invocation for one method/variant, instance, and seed.

Campaign tasks automatically forward:

```text
--budget-policy auto_from_instance
--archive-layouts both
--no-figures
```

The campaign runner also supports `--dry-run`, `--resume`, worker control, predefined instance scopes, and explicit instance lists.

### 4.1 Phase definitions

| Phase | Runner members | Core task count |
|---|---|---:|
| Phase 11 | Proposed, random-restart BS, BS-only | 360 |
| Phase 12B | V1 fixed sorting, V2 fixed weights, V3 uniform mutation, V4 no symmetry breaking, V5 random feasible-start spacing | 600 |
| Phase 12C | V6 depth 15, V7 beam width +1 | 240 |

**Phase 12B does not execute V0.** The V0/full-proposed baseline is the Phase 11 `proposed_nsga2_bs` result for the same core instances and seeds. Post-processing labels/reuses that Phase 11 result as V0 when constructing the Phase 12B comparison.

**Phase 12C also does not execute V0.** The same Phase 11 Proposed raw archives are reused as V0, but Phase-12C indicators are recomputed inside the separate V0/V6/V7 normalization/reference union.

`V6b_binding_depth10` is a separate Phase 12C diagnostic restricted to `demo_layout_door_left_AW_2`; it is not part of the default 240-task Phase 12C set and is stored in the separate final-clean campaign root `results/revision_final_30seed_nofg_v6b`.

### 4.2 Instance scopes

- `core`: four replicated inferential instances;
- `stress`: predefined supplementary stress subset;
- `all`: `core + stress` presets.

`all` means all predefined campaign presets, not every repository mask. Use `--instance-list` for an arbitrary set of repository masks.

### 4.3 Fresh run and resume

For a fresh campaign, use a new/empty campaign root and do not add `--resume`.

If the same campaign is interrupted, rerun the identical command with:

```text
--resume
```

Only tasks with an existing completed `experiment_summary.csv` are skipped.

---

## 5. Parameter Reference

### 5.1 `run_experiment_manager`

#### Run selection and output

| Option | Meaning |
|---|---|
| `--config PATH` | Experiment configuration file. Default: `configs/experiment_plan.yaml`. |
| `--instances ...` | One or more instance names or paths; comma-separated lists are accepted. |
| `--seeds SEEDS` | Comma-separated random seeds. |
| `--method METHOD` | `proposed_nsga2_bs`, `random_restart_bs`, or `bs_only`. |
| `--experiment-id ID` | Explicit experiment folder identifier. |
| `--output-dir PATH` | Experiment output root. Default: `results/experiments`. |
| `--dry-run` | Build the experiment plan without running optimization. |

#### Search budget

| Option | Meaning |
|---|---|
| `--budget-policy {fixed,auto_from_instance}` | Search-budget source. Default: `auto_from_instance`. |
| `--population-size N` | Population size or RRBS population-equivalent value. |
| `--generations N` | Generation count or RRBS generation-equivalent value. |
| `--decode-budget N` | Total RRBS restarts; otherwise derived from population size × generations. |
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
| `--only-variant NAME` | Restrict execution to one Phase 11 method or runnable Phase 12 variant. Phase 12B/12C V0 is not runnable because it is reused from Phase 11 Proposed. |
| `--max-workers N` | Maximum concurrent manager tasks. Default: `3`. |
| `--dry-run` | Write the task manifest without executing optimization. |
| `--resume` | Skip tasks with an existing completed `experiment_summary.csv`. |
| `--archive-rank-max N` | Maximum archived rank forwarded to the manager. Default: `3`. |
| `--profile-light` | Enable lightweight runtime profiling for each task. |
| `--save-generation-objectives` | Save generation-level objective evidence for each task. |

### 5.3 `analyze_revision_campaign_evidence`

| Option | Meaning |
|---|---|
| `--results-root PATH` | Main final-clean campaign root. Default: `results/revision_final_30seed_nofg`. |
| `--v6b-results-root PATH` | Separate completed V6b campaign root. Default: `results/revision_final_30seed_nofg_v6b`. |
| `--output-dir PATH` | Fresh/empty evidence-package directory. Default: `data/reproducibility/revision_final_30seed_nofg/structural`. |
| `--phases ...` | Main completed comparison groups to analyze: `phase11`, `phase12b`, and/or `phase12c`. Default: all three. |
| `--skip-v6b` | Intentionally omit the separate matched V0-vs-V6b Demo diagnostic. |

The analyzer requires complete expected archive/manifest coverage for each selected family and refuses a non-empty output directory. V0 is reused from Phase 11 Proposed for both Phase 12B and Phase 12C. The V6b comparison uses its own Demo V0+V6b normalization/reference union.

---

## 6. Saved Outputs and Archives

A manager run writes per-run outputs under:

```text
<output-dir>\<experiment-id>\runs\<instance>\seed_<seed>\
```

Common outputs include:

- `candidates.csv`
- `generation_summary.csv`
- `run_metadata.json`
- `runtime_profile_summary.csv` when `--profile-light` is enabled
- `generation_profile.csv` when `--profile-light` is enabled
- `generation_objectives.csv` when `--save-generation-objectives` is enabled

Experiment-level outputs include `experiment_metadata.json` and `experiment_summary.csv`.

Archive modes:

| Mode | Saved layout set |
|---|---|
| `none` | No layout archive. |
| `generation_elites` | Eligible generation-level layouts filtered by `--archive-rank-max`. |
| `final_ranked` | Unique feasible final layouts filtered by `--archive-rank-max`. |
| `both` | Both generation-elites and final-ranked archives. |
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

```powershell
python -m whl_experiments.render_saved_layouts `
  --archive results\quick_nsga2_bs\<experiment-id>\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts.npz `
  --index results\quick_nsga2_bs\<experiment-id>\runs\Gyorgy-KOVACS_WH_Narrow_AW_4\seed_101\final_ranked_layouts_index.json `
  --filter rank0_to_rank3
```

Important options include `--filter`, `--output-dir`, `--max-layouts`, `--dpi`, `--title-fields`, `--title-format`, `--no-legend`, and `--show-coords`.

### 7.2 Batch-render a completed experiment

```powershell
python -m whl_experiments.render_experiment_archives `
  --experiment-dir results\quick_nsga2_bs\<experiment-id> `
  --archive-type final_ranked `
  --filter rank0_to_rank3
```

Use `--dry-run` to inspect discovered archives without rendering. Use `--overwrite` only when existing rendered outputs should be replaced.

---

## 8. Paper-Style Pareto Plotting

```powershell
python -m whl_visualization.paper_pareto_plots
```

Paper plotting inputs are stored under `data/plot_inputs/paper/`. This command is separate from optimization and layout archive rendering.

---

## 9. Layout Editor and Instance Rules

Launch the Tkinter editor with:

```powershell
python -m apps.layout_editor.launch_editor
```

`AT_1.npz` through `AT_13.npz` are reference/comparison layouts and should not be used as optimization instances. `AT_S_comercial_layout_AW_3.npz` is an optimization instance. See `docs/layout_data.md` for the complete classification.

---

## 10. Operational-Layer Diagnostics

The operational layer is an optional post-optimization diagnostic for the fixed paper layouts L1-L4. It does not contribute to NSGA-II/Beam Search fitness and does not feed back into structural optimization.

See:

```text
docs/operational_layer.md
```

for the checked-in reviewer evidence, occupancy workflow, and reproduction commands.

---

## 11. Example Commands

The following commands reproduce the structural campaigns used for the paper. Run the dry-run variant first when verifying task enumeration. Do not use `--resume` for a new/empty campaign root.

### 11.1 Phase 11 — three-method core campaign

Dry-run:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
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

Expected: `360` tasks.

Execution:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
  --phase phase11 `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

### 11.2 Phase 12B — V1 to V5 ablations

Phase 12B automatically runs V1-V5 only. It does **not** run V0. The Phase 11 Proposed results are reused as V0 during analysis.

Dry-run:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
  --phase phase12b `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3 `
  --dry-run
```

Expected: `600` tasks.

Execution:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
  --phase phase12b `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

### 11.3 Phase 12C — V6 and V7 sensitivity

V0 is not rerun; the Phase 11 Proposed raw archives are reused and re-evaluated inside the V0/V6/V7 comparison-specific indicator union.

Dry-run:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
  --phase phase12c `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3 `
  --dry-run
```

Expected: `240` tasks.

Execution:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg `
  --phase phase12c `
  --seed-start 101 `
  --seed-end 130 `
  --instances core `
  --max-workers 5 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

### 11.4 Phase 12C V6b — binding-depth diagnostic

Use a separate final-clean root so the V6b diagnostic cannot be confused with the four-instance V6/V7 Phase-12C campaign:

```powershell
python -m whl_experiments.run_revision_30seed_campaign `
  --campaign-root results/revision_final_30seed_nofg_v6b `
  --phase phase12c `
  --seed-start 101 `
  --seed-end 130 `
  --only-variant V6b_binding_depth10 `
  --only-instance demo_layout_door_left_AW_2 `
  --max-workers 3 `
  --profile-light `
  --save-generation-objectives `
  --archive-rank-max 3
```

Expected: `30` tasks.

V6b is a matched Demo-only binding-depth diagnostic. It is not part of the V0/V6/V7 four-instance Table-8 comparison.

### 11.5 Supplementary one-seed structural check

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

Expected: `33` tasks.

### 11.6 Unified final-clean evidence analysis

Run this only after Phase 11, Phase 12B, Phase 12C, and V6b are complete. Do not run post-processing concurrently with timed optimization campaigns when runtime evidence is still being collected.

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --results-root results\revision_final_30seed_nofg `
  --v6b-results-root results\revision_final_30seed_nofg_v6b `
  --output-dir data\reproducibility\revision_final_30seed_nofg\structural
```

The defaults select `phase11 phase12b phase12c` and include V6b. The output directory must be absent or empty.

The comparison-specific reference unions are intentionally separate:

- Phase 11 / Table 5: Proposed + BS-only + RRBS;
- Phase 12B / Table 7: V0--V5;
- Phase 12C / Table 8: V0 + V6 + V7;
- V6b: Demo V0 + V6b.

Therefore the same Phase-11 Proposed raw archive reused as V0 can receive different HV/IGD+/OSD values in different comparison families.

To omit V6b intentionally, add `--skip-v6b`. To analyze only selected completed main families, specify them with `--phases` and use a fresh output directory.

The unified analyzer produces the compact manuscript/reviewer evidence under `data/reproducibility/revision_final_30seed_nofg/structural/`. See `docs/revision_evidence.md` for the complete output inventory, statistical protocol, V6b safeguards, and manuscript mapping.
