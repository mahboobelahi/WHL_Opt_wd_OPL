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

`occupancy_sensitivity/` contains compact validated results retained for direct reviewer inspection and manuscript comparison:

- `occupancy_summary.csv`: Low/Medium/High x L1-L4 capacity, utilization, reserve deep-share, and reserve-access cost (12 rows).
- `occupancy_fragmentation_summary.csv`: layout-level SKU-count-weighted mean reserve groups per SKU (12 rows).
- `scenario_B_weight_sensitivity_summary.csv`: B1-B4 reserve-access costs and rankings at Low/Medium/High occupancy (12 rows).

The occupancy totals are Low=790, Medium=2,240, and High=3,584 pallets. Each case retains one representative pallet per SKU; reserve totals are 690, 2,140, and 3,484 respectively.

The compact CSVs reproduce the validated revision evidence used for the manuscript. Detailed Low/Medium/High case directories can be regenerated with the public occupancy scripts; generated case folders remain under `results/` and are not versioned.

## Public reproduction modules

- `whl_experiments.build_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_weight_sensitivity_by_occupancy`

See `docs/operational_layer.md` for the exact commands and interpretation limits.
