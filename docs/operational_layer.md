# Operational-layer diagnostics

## 1. Scope

The operational layer (OPL) is a deterministic post-optimization diagnostic for the fixed L1-L4 layouts used in Section 7 of the paper. It is separate from structural optimization:

- OPL is not an NSGA-II or Beam Search objective.
- OPL does not alter or reselect the structural layouts.
- OPL does not feed results back into the optimizer.
- The diagnostics are static access/assignment proxies, not routed warehouse simulation or calibrated travel time.

The locked pallet-slot capacities are:

| Layout | Capacity |
|---|---:|
| L1 | 4,480 |
| L2 | 5,056 |
| L3 | 5,440 |
| L4 | 4,800 |

## 2. Checked-in reviewer evidence

The authoritative fixed-layout data are under `data/operational_layer/paper_inputs/`.

Key files are:

- `selected_layouts.csv` — L1-L4 identities, structural provenance, and capacities.
- `slot_metrics_by_layout.csv` — 19,776 slot rows with access distance, effective depth, vertical level, normalized descriptors, and slot cost.
- `sku_catalog.csv` — deterministic 100-SKU ABC catalog and Low=790 inventory.
- `representative_access_assignment.csv` — 400 representative assignments.
- `reserve_pallet_assignment.csv` — 2,760 Low-load reserve assignments.
- `regime_A_metrics.csv` — representative-access diagnostics.
- `regime_B_metrics.csv` — Low-load reserve/capacity diagnostics.
- `reserve_fragmentation_summary.csv` — class-specific reserve fragmentation.
- `synthetic_orders.csv`, `order_effort_by_seed.csv`, `order_effort_summary.csv` — fixed synthetic workload evidence.
- `sensitivity/lambda_sensitivity_summary.csv`, `sensitivity/lambda_sensitivity_by_seed.csv` — canonical weight-sensitivity evidence.

Compact validated Low/Medium/High results are checked in under `data/operational_layer/occupancy_sensitivity/` for direct reviewer inspection and manuscript comparison.

## 3. Occupancy protocol

The same L1-L4 layouts, slot geometry, 100 SKUs, ABC demand weights, SKU order, representative-assignment rule, and reserve-assignment rule are retained at every occupancy.

| Occupancy | Total pallets | Reserve pallets | Share of smallest capacity |
|---|---:|---:|---:|
| Low | 790 | 690 | 17.63% |
| Medium | 2,240 | 2,140 | 50.00% |
| High | 3,584 | 3,484 | 80.00% |

Inventory is scaled from the canonical 790-pallet vector by deterministic proportional largest remainder. Each SKU retains exactly one representative-access pallet. Equal remainders are resolved by ascending `global_sku_index`.

Expected scaled quantities are:

- Low: A `20 x 7`, B `30 x 15`, C `50 x 4`.
- Medium: A `20 x 20`, B `30 x 43`, C `50 x 11`.
- High: A `20 x 32`, B `30 x 68`, C `4 x 19` and `46 x 18`.

Representative assignments and Scenario-A diagnostics must remain invariant across occupancy. Reserve placement, utilization, reserve access, and reserve fragmentation may change.

## 4. Generate validated occupancy cases

Run from the repository root. The canonical source directory defaults to `data/operational_layer/paper_inputs`.

### Low reproduction gate

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\low_790 `
  --occupancy-label low `
  --inventory-total 790 `
  --verify-low-against data\operational_layer\paper_inputs
```

### Medium

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\medium_2240 `
  --occupancy-label medium `
  --inventory-total 2240
```

### High

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\high_3584 `
  --occupancy-label high `
  --inventory-total 3584
```

Each case writes:

```text
data/
  sku_catalog_scaled.csv
  representative_access_assignment.csv
  reserve_pallet_assignment.csv
  regime_A_metrics.csv
  regime_B_metrics.csv
  reserve_fragmentation_summary.csv
logs/
  validation_summary.json
  occupancy_manifest.json
```

## 5. Cross-occupancy analysis

After all three cases pass validation:

```powershell
python -m whl_experiments.analyze_operational_occupancy_sensitivity `
  --low-root results\section7_occupancy\low_790 `
  --medium-root results\section7_occupancy\medium_2240 `
  --high-root results\section7_occupancy\high_3584 `
  --output-root results\section7_occupancy\summary
```

The analysis checks representative and Scenario-A invariance, source hashes, and shortage-free assignments. It writes:

- `occupancy_summary.csv`
- `occupancy_fragmentation_by_class.csv`
- `occupancy_reserve_components.csv`
- `occupancy_fragmentation_layout_summary.csv`
- `summary_manifest.json`

The reserve-access decomposition is:

```text
mean(normalized_distance)
+ lambda_depth * mean(normalized_depth)
+ lambda_level * mean(normalized_level)
```

with the baseline Scenario-B weights `(0.1, 0.1)`.

## 6. Weight sensitivity by occupancy

```powershell
python -m whl_experiments.analyze_operational_weight_sensitivity_by_occupancy `
  --low-root results\section7_occupancy\low_790 `
  --medium-root results\section7_occupancy\medium_2240 `
  --high-root results\section7_occupancy\high_3584 `
  --output-root results\section7_occupancy\weight_sensitivity
```

The canonical Scenario-A and Low Scenario-B sensitivity files are read from `data/operational_layer/paper_inputs/sensitivity/` by default.

Scenario-B reserve weights are:

| Case | depth weight | level weight |
|---|---:|---:|
| B1 | 0.10 | 0.10 |
| B2 | 0.25 | 0.10 |
| B3 | 0.10 | 0.25 |
| B4 | 0.25 | 0.25 |

Outputs are:

- `scenario_B_weight_sensitivity_by_occupancy.csv`
- `scenario_B_weight_sensitivity_ranking.csv`
- `scenario_A_existing_sensitivity_summary.csv`
- `weight_sensitivity_manifest.json`

The analysis must reproduce the canonical Low Scenario-B evidence within absolute tolerance `1e-9` before accepting Medium/High weight-sensitivity results.

## 7. Checked-in compact occupancy results

`data/operational_layer/occupancy_sensitivity/` contains the compact validated evidence used for manuscript comparison:

- `occupancy_summary.csv`
- `occupancy_fragmentation_summary.csv`
- `scenario_B_weight_sensitivity_summary.csv`

Detailed generated case folders remain under `results/` and are intentionally not versioned. They can be regenerated with the commands above.

## 8. Interpretation limits

The OPL evidence supports deterministic comparisons of fixed-layout access, reserve placement, fragmentation, utilization, and sensitivity to inventory load and proxy weights.

It must not be interpreted as routed travel time, throughput, congestion, batching, replenishment dynamics, service rate, or calibrated forklift performance. Synthetic-order effort aggregates the same fixed representative-access components under sampled SKU mixes; it is not an independent routing model.
