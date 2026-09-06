# Figure 7 plotting note

Use `convergence_summary.csv` to rebuild Figure 7.

- Panel (a): `phase11`
- Panel (b): `phase12b`
- Panel (c): `phase12c`
- Use only the two representative instances:
  `AT_S_comercial_layout_AW_3` (Atefeh) and
  `Gyorgy-KOVACS_WH_Narrow_AW_4` (Kov-1-O-w4).
- Plot `hv_mean` against `index`.
- Shade `hv_mean ± hv_std` (one standard deviation across the 30 seeds).
- Do NOT compare numerical HV values across Phase11/Phase12B/Phase12C:
  each panel uses its own comparison-specific normalization.
- BS-only has only a single direct-search archive point and therefore no
  iterative outer-search trajectory.
