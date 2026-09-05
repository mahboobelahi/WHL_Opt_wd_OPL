# Revision campaign evidence analysis

The completed structural revision campaigns are post-processed with:

```powershell
python -m whl_experiments.analyze_revision_campaign_evidence `
  --results-root results/revision_final_30seed_nofg `
  --output-dir data/reproducibility/revision_final_30seed_nofg/structural `
  --phases phase11 phase12b
```

This command performs **post-processing only**. It does not invoke NSGA-II, Beam Search, RRBS, BS-only, or any other optimization run.

## Phase 11

The analyzer expects the complete 30-seed campaign for the four core instances and the three methods:

- `proposed_nsga2_bs`
- `random_restart_bs`
- `bs_only` in the campaign manifest, reported as `bs_only_direct` in the analysis evidence

The indicator/statistical protocol follows Section 5 of the manuscript: final feasible Pareto-rank 0--3 archives, comparison-specific min--max normalization, empirical nondominated reference front, HV reference point `(1.1,1.1,1.1)`, IGD+, descriptive OSD, per-instance paired Wilcoxon tests, Holm correction, paired rank-biserial effects, and secondary fixed-block pooled results.

## Phase 12B

Phase 12B contains V1--V5 only. `V0_full_proposed` is **not rerun**. During post-processing the analyzer reuses the completed Phase-11 `proposed_nsga2_bs` raw archives and recomputes their indicators inside the V0--V5 comparison-specific normalization/reference union.

Consequently, the underlying V0 raw runs are identical to the Phase-11 Proposed runs, while their HV/IGD+/OSD values may differ between the Phase-11 and Phase-12B tables because the comparison-specific reference unions differ.

## Outputs

The output directory must be absent or empty. The command creates compact reviewer-facing CSV/JSON evidence, including:

```text
phase11_seed_level.csv
phase11_summary_by_instance.csv
phase11_summary_overall.csv
phase11_stats_by_instance.csv
phase11_stats_pooled.csv
phase11_friedman_instance_means.csv
phase11_signature_summary.csv
phase11_seed_novelty.csv
table5_phase11_manuscript_values.csv
phase12b_seed_level.csv
phase12b_summary_by_instance.csv
phase12b_summary_overall.csv
phase12b_signature_summary.csv
phase12b_v0_pairwise_stats.csv
table7_phase12b_manuscript_values.csv
indicator_reference_metadata.csv
input_manifest.csv
input_hashes.csv
analysis_summary.json
README.md
```

These compact files are intended for manuscript/reviewer verification and are tracked under `data/reproducibility/`. The raw campaign tree remains generated evidence and is selectively exposed only where required by `.gitignore`.
