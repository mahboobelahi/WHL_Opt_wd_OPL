# Revision campaign evidence analysis

The final-clean structural revision campaigns are post-processed by one unified analyzer:

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --results-root results\revision_final_30seed_nofg `
  --v6b-results-root results\revision_final_30seed_nofg_v6b `
  --output-dir data\reproducibility\revision_final_30seed_nofg\structural
```

The current defaults are the same main campaign root, V6b campaign root, output directory, and the three main comparison groups `phase11 phase12b phase12c`; therefore the command can also be run without arguments after the final-clean campaigns are available.

This command performs **post-processing only**. It does not invoke NSGA-II, Beam Search, RRBS, BS-only, or any other optimizer.

## Execution gate

Run the unified analyzer only after the required experiment evidence is complete:

- Phase 11: 360 completed core tasks;
- Phase 12B: 600 completed V1--V5 core tasks;
- Phase 12C: 240 completed V6/V7 core tasks;
- V6b: 30 completed `demo_layout_door_left_AW_2` tasks in the separate `results/revision_final_30seed_nofg_v6b` campaign root.

Do not run this post-processing concurrently with timed optimization campaigns when runtime measurements are being collected.

The output directory must be absent or empty. This prevents a new evidence package from being mixed with files from an earlier analysis.

## Comparison families

### Phase 11 — Proposed / BS-only / RRBS

The analyzer expects seeds 101--130 on the four core instances for:

- `proposed_nsga2_bs`;
- `random_restart_bs`;
- `bs_only` in the campaign manifest, reported as `bs_only_direct` in the analysis evidence.

The main statistical evidence uses matched seeds per instance, two-sided Wilcoxon signed-rank tests, Holm correction across the three method pairs within each metric/instance, and paired rank-biserial effects. The pooled 4 x 30 fixed-block analysis is secondary evidence, not 120 independent warehouse instances. OSD is descriptive and has no preferred direction.

### Phase 12B — V0--V5

Phase 12B executes V1--V5 only. `V0_full_proposed` is **not rerun**. The analyzer reuses the completed Phase-11 `proposed_nsga2_bs` raw final archives and recomputes V0 inside the Phase-12B V0--V5 comparison-specific normalization/reference union.

`phase12b_v0_pairwise_stats.csv` is supporting evidence; Table 7 remains a descriptive ablation summary unless those tests are explicitly cited.

### Phase 12C — V0 / V6 / V7

Phase 12C executes:

- `V6_depth15_beam_default`;
- `V7_beam_plus1_depth_default`.

V0 is again reused from the Phase-11 Proposed raw archives, but the analyzer recomputes all indicators inside a separate V0/V6/V7 comparison-specific normalization/reference union. Phase-12C values therefore must not be copied from the Phase-11 or Phase-12B indicator tables.

The analyzer also writes retained final-archive depth evidence for V0/V6/V7. This diagnostic is restricted to final feasible Pareto-rank 0--3 layouts and **must not be interpreted as an exact Beam Search stop-reason or frontier-level cap-hit log**.

### V6b — separate matched Demo diagnostic

`V6b_binding_depth10` is not folded into the Phase-12C V0/V6/V7 comparison. It is a separate matched diagnostic on `demo_layout_door_left_AW_2`, seeds 101--130, using:

- V0 from `results/revision_final_30seed_nofg/p11/nsga2`;
- V6b from `results/revision_final_30seed_nofg_v6b/p12c/V6b_d10`.

The analyzer constructs a **separate V0+V6b normalization/reference union** for this diagnostic. It checks stored scientific configuration fields, expected search parameters, exact-grid signatures against archived `.npz` layouts, retained structural depths, candidate rows above the configured depth cap, and the matched V0-vs-V6b statistical comparison.

Configured `Dmax=10` alone is not treated as evidence that the cap was binding; binding evidence is derived from observed retained/candidate depths.

OSD remains descriptive. HV, IGD+, unique-layout count per run, and runtime use the matched V0-vs-V6b paired comparison.

## Indicator protocol

For Phase 11, Phase 12B, Phase 12C, and V6b the structural indicator protocol uses:

- final feasible Pareto-rank 0--3 archive records;
- minimization vector `(N_locked, -N_pf, R_p)`;
- min--max normalization separately for each fixed instance and comparison family;
- zero observed ranges replaced by one;
- the unique nondominated normalized comparison union as the empirical reference front;
- HV reference point `(1.1,1.1,1.1)`;
- IGD+ positive-part minimization distance;
- descriptive OSD;
- saved exact-grid `layout_signature` for structural identity.

The normalization/reference unions are deliberately comparison specific:

| Manuscript/evidence family | Reference union |
|---|---|
| Phase 11 / Table 5 | Proposed + BS-only + RRBS |
| Phase 12B / Table 7 | V0--V5 |
| Phase 12C / Table 8 | V0 + V6 + V7 |
| V6b / Section 5.4 + Appendix C3 | Demo V0 + V6b |

Consequently, the same raw V0 archive can receive different HV, IGD+, and OSD values in different comparison families.

## Main outputs

The unified output directory is:

```text
data/reproducibility/revision_final_30seed_nofg/structural/
```

### Phase 11

```text
phase11_seed_level.csv
phase11_summary_by_instance.csv
phase11_summary_overall.csv
phase11_signature_summary.csv
phase11_seed_novelty.csv
phase11_stats_by_instance.csv
phase11_stats_pooled.csv
phase11_friedman_instance_means.csv
table5_phase11_manuscript_values.csv
```

### Phase 12B

```text
phase12b_seed_level.csv
phase12b_summary_by_instance.csv
phase12b_summary_overall.csv
phase12b_signature_summary.csv
phase12b_v0_pairwise_stats.csv
table7_phase12b_manuscript_values.csv
```

### Phase 12C

```text
phase12c_seed_level.csv
phase12c_summary_by_instance.csv
phase12c_summary_overall.csv
phase12c_signature_summary.csv
phase12c_v0_pairwise_stats.csv
phase12c_depth_by_seed.csv
phase12c_depth_summary.csv
table8_phase12c_manuscript_values.csv
```

### V6b

```text
v6b_binding_by_seed.csv
v6b_binding_summary.json
v6b_configuration_match_by_seed.csv
v6b_indicator_seed_level.csv
v6b_indicator_summary.csv
v6b_indicator_protocol.json
v6b_paired_statistics.csv
```

### Shared provenance

```text
indicator_reference_metadata.csv
input_manifest.csv
input_hashes.csv
analysis_summary.json
README.md
```

`input_manifest.csv` records the logical evidence inputs, including reused V0 runs. `indicator_reference_metadata.csv` records the comparison-specific normalization/reference definitions. `input_hashes.csv` provides SHA-256 hashes of the files read by the analyzer.

## Manuscript evidence mapping

The analyzer directly prepares the core numerical evidence for:

- Table 5 from `table5_phase11_manuscript_values.csv`;
- Table 7 from `table7_phase12b_manuscript_values.csv`;
- Table 8 from `table8_phase12c_manuscript_values.csv`;
- Phase-11 inferential claims from the Phase-11 statistics files;
- Phase-12C cross-seed structural identity evidence from `phase12c_signature_summary.csv`;
- the Section 5.4 / Appendix C3 V6b diagnostic from the V6b binding, indicator, and paired-statistics files.

Runtime and signature summaries generated here support reconciliation of the corresponding manuscript runtime/search-workload and structural-rediscovery entries. Generation-convergence tables/figures are **not regenerated by this analyzer**; they require the saved `generation_objectives.csv` evidence and the separate convergence post-processing workflow.

V6b remains a single-instance binding-depth diagnostic and must not be inserted into the Phase-12C V0/V6/V7 Table 8 comparison or treated as a four-instance sensitivity result.

## CLI variants

Analyze the three main families but intentionally omit V6b:

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --results-root results\revision_final_30seed_nofg `
  --output-dir data\reproducibility\revision_final_30seed_nofg\structural `
  --phases phase11 phase12b phase12c `
  --skip-v6b
```

Analyze only selected completed main families by changing `--phases`. For example:

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --phases phase11 phase12b `
  --skip-v6b `
  --output-dir data\reproducibility\revision_partial_check
```

Use a fresh/empty output directory for every such analysis run.

These compact files are intended for manuscript/reviewer verification and are tracked under `data/reproducibility/`. The raw campaign tree remains generated evidence and is selectively exposed only where required by `.gitignore`.
