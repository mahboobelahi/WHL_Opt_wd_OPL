# Operational-layer reviewer evidence

This directory contains the checked-in data used for the paper's Section 7 operational diagnostics. The operational layer is post-optimization and does not feed back into NSGA-II or Beam Search.

## Canonical fixed-layout evidence

`paper_inputs/` contains the reviewer-facing canonical evidence for the fixed L1-L4 panel:

- `selected_layouts.csv`: layout identity, structural provenance, and pallet-slot capacity for L1-L4.
- `slot_metrics_by_layout.csv`: slot-level access, depth, level, normalized descriptors, and slot cost for all 19,776 pallet slots.
- `sku_catalog.csv`: deterministic 100-SKU ABC catalog and the Low=790 pallet inventory.
- `representative_access_assignment.csv`: 400 representative-access assignments (100 SKUs x 4 layouts).
- `reserve_pallet_assignment.csv`: 2,760 Low-load reserve assignments (690 x 4 layouts).
- `regime_A_metrics.csv`: occupancy-invariant representative-access diagnostics.
- `regime_B_metrics.csv`: Low-load reserve diagnostics and capacity/utilization measures.
- `reserve_fragmentation_summary.csv`: Low-load class-specific reserve fragmentation evidence.
- `synthetic_orders.csv`, `order_effort_by_seed.csv`, and `order_effort_summary.csv`: fixed synthetic workload evidence.
- `sensitivity/lambda_sensitivity_summary.csv` and `sensitivity/lambda_sensitivity_by_seed.csv`: canonical Scenario-A and Low Scenario-B weight sensitivity.

The locked capacities are L1=4,480, L2=5,056, L3=5,440, and L4=4,800 pallet slots.

## Occupancy-sensitivity evidence

`occupancy_sensitivity/` now contains the audited Round-2 Section-7 evidence transferred from the original WHLR result tree. It includes:

- compact manuscript-facing summaries at the directory root;
- `cases/low_790`, `cases/medium_2240`, and `cases/high_3584` with validated Scenario-B and fragmentation outputs plus path-sanitized validation summaries;
- `cross_occupancy/` with utilization, reserve-placement, fragmentation, and reserve-component decomposition outputs;
- `weight_sensitivity/` with Scenario-A and Scenario-B sensitivity evidence;
- `scaled_inventory_summary.csv` with the exact deterministic Low/Medium/High inventory vectors;
- `source_result_inventory.csv` with row counts, sizes, and SHA-256 hashes for all 33 files in the audited WHLR source result bundle.

The occupancy totals are Low=790, Medium=2,240, and High=3,584 pallets. Each case retains one representative pallet per SKU; reserve totals are 690, 2,140, and 3,484 respectively.

Large occupancy-specific per-pallet assignment CSVs are not duplicated because the public wrapper regenerates them deterministically; their source row counts and SHA-256 hashes are retained in `source_result_inventory.csv`. The canonical Low assignment remains checked in under `paper_inputs/`.

## Public reproduction modules

- `whl_experiments.build_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_weight_sensitivity_by_occupancy`

See `docs/operational_layer.md` for the exact commands and interpretation limits.
