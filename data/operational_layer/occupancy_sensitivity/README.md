# Section 7 occupancy-sensitivity evidence

This directory exposes the reviewer-facing evidence audited from the original WHLR result tree `results/round2_section7_occupancy_sensitivity`.

The attached source bundle used for the transfer had SHA-256:

`b3479f28d0e5d50a4e81b3ed012cdbf0d846a83e5eadcf0b94bda692c47b80fc`

## Fixed structural/slot evidence

The fixed L1-L4 identities, capacities, and all 19,776 pallet-slot descriptors remain under `../paper_inputs/`:

- `selected_layouts.csv`
- `slot_metrics_by_layout.csv`
- `sku_catalog.csv`
- `representative_access_assignment.csv`
- `reserve_pallet_assignment.csv`
- `regime_A_metrics.csv`
- `sensitivity/`

L1-L4 capacities are 4,480, 5,056, 5,440, and 4,800 pallet slots.

## Occupancy case evidence

`cases/` contains the occupancy-sensitive Scenario-B metrics, class-specific reserve-fragmentation metrics, and path-sanitized validation summaries for:

- `low_790/`
- `medium_2240/`
- `high_3584/`

The Low reproduction gate passed against the canonical Round-1 OPL evidence. Medium and High passed all recorded assignment, accessibility, capacity, uniqueness, representative-invariance, Scenario-A-invariance, and source-integrity checks.

`scaled_inventory_summary.csv` records the deterministic Low/Medium/High SKU quantities used by the validated runs.

## Cross-occupancy evidence

`cross_occupancy/` contains the scientific rows and values audited from the WHLR result CSVs:

- `occupancy_summary.csv`
- `occupancy_fragmentation_by_class.csv`
- `occupancy_fragmentation_layout_summary.csv`
- `occupancy_reserve_components.csv`

These expose utilization, reserve deep placement, reserve-access cost, horizontal/depth/level decomposition, and fragmentation across all 12 occupancy-layout cases.

## Weight sensitivity

`weight_sensitivity/` contains the validated Scenario-A and Scenario-B sensitivity rows and values used for Appendix Table C1 support.

## Source-result inventory

`source_result_inventory.csv` records every one of the 33 original WHLR result files, including byte size, SHA-256, and CSV dimensions where applicable. This includes the original per-pallet representative/reserve assignment files and original manifests.

The source hashes in that inventory refer to the original WHLR files. Reviewer-facing CSV copies in this repository may have normalized text line endings; their scientific rows and values are preserved.

The large occupancy-specific per-pallet assignment CSVs are not duplicated here because the public wrapper regenerates them deterministically and the canonical Low assignment is already checked in under `../paper_inputs/`. Their original row counts and SHA-256 hashes remain recorded in `source_result_inventory.csv`.

## Reproduction code

Public modules:

- `whl_experiments.build_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_occupancy_sensitivity`
- `whl_experiments.analyze_operational_weight_sensitivity_by_occupancy`

See `docs/operational_layer.md` for commands and interpretation limits. The OPL remains a deterministic post-optimization diagnostic and must not be interpreted as routed travel time, throughput, congestion, batching, or replenishment simulation.
