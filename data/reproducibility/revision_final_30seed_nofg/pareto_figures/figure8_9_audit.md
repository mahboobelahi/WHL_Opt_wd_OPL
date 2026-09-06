# Figure 8/9 Fresh Pareto Plot Audit

## Provenance
- Results root: `C:\Users\elahi\Desktop\WHL_Opt_wd_OPL\results\revision_final_30seed_nofg`
- Phase-11 Proposed root: `C:\Users\elahi\Desktop\WHL_Opt_wd_OPL\results\revision_final_30seed_nofg\p11\nsga2`
- Seeds: 101–130 (30 seeds).
- Archive scope: final feasible Pareto ranks 0–3.
- Deduplication: exact-grid `layout_signature` across seeds; best observed final within-seed rank retained.
- Old logged rank-point CSVs are not used.

## Fresh unique-layout manifests
- Atefeh: 709 archive entries → **62 unique signatures**; rank counts {'0': 23, '1': 11, '2': 12, '3': 16}.
  Ranges: SC 195–264; Npf 87–127; Nlocked 90–144; Rp 69–344.
  Derived SC–Npf non-dominated envelope points: 3.
  Timing: 30 generation-summary files; generations 0–14; 30 seeds per generation = True.
- Kov-1-O-w4: 655 archive entries → **39 unique signatures**; rank counts {'0': 7, '1': 14, '2': 9, '3': 9}.
  Ranges: SC 560–684; Npf 232–270; Nlocked 290–424; Rp 341–616.
  Derived SC–Npf non-dominated envelope points: 3.
  Timing: 30 generation-summary files; generations 0–17; 30 seeds per generation = True.

## Atefeh published reference
- Published layouts re-scored from AT_1--AT_13 masks: 13.
- Status: recomputed from published-reference masks via load_mask -> mask_to_grid -> score_layout.
- Derived metrics CSV: `C:\Users\elahi\Desktop\WHL_Opt_wd_OPL\data\reproducibility\revision_final_30seed_nofg\pareto_figures\atefeh_published_reference_metrics.csv`.
- Ranges: SC 153–348; Npf 83–155; Nlocked 28–265; Rp 85–4324.

## Quantitative generated-vs-published dominance
- Generated Pareto-rank 0 signatures: 23.
- Published layouts dominated by ≥1 generated rank-0 signature: **6/13 (46.2%)**.
- Generated rank-0 signatures dominated by ≥1 published layout: **20/23 (87.0%)**.
- Pooled nondominated union: **5 designs**, representing **4 distinct objective tuples**.
- Pooled nondominated source split: generated 3, published 2.

## Figure terminology
- Legends use `Pareto rank 0` … `Pareto rank 3`.
- Published Atefeh overlay: `Digitized published layouts`.
- SC–Npf line: `Non-dominated envelope` (derived descriptor space, not an optimized Pareto front).
